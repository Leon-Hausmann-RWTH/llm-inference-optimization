#!/usr/bin/env python3
"""
Optuna quantisation search for HPC

Searches uniform and mixed precision quantisation configs for Qwen3-0.6B

terminal command: python optuna_search.py --trials 30

Output Directories:
    $HOME/ADL_Project/Models            # model checkpoints
    $HOME/ADL_Project/Logs              # output jsons
"""
# Imports
# Use conda actiavte llm_env
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import optuna
import psutil
import torch
import torch.distributed as dist
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

# ------------------------------------------------------

HOME = Path(os.environ["HOME"])
PROJECT_DIR = HOME / "ADL_Project"
SAVE_DIR = PROJECT_DIR / "Models"
LOG_DIR = PROJECT_DIR / "Logs"
TRIAL_LOG = LOG_DIR / "optuna_trials.jsonl"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------

# Hyperparameters 
MAX_SEQUENCE_LENGTH = 2048
NUM_CALIBRATION_SAMPLES = 512
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
MODEL_ID = "Qwen/Qwen3-0.6B"
GPU_MEM_UTIL = 0.8
EVAL_SAMPLES = 200
TOP_K_LOGPROBS = 20
MISSING_TOKEN_LOGPROB = -10.0
DAMPENING_FRAC = 1e-3

# TPS Eval prompts
SEARCH_PROMPTS = [
    "Explain what quantization does for transformer inference in two sentences.",
    "Write a short paragraph about the benefits of batching in LLM serving.",
    "Why can lower precision improve inference throughput?",
    "Summarize the tradeoff between speed and quality for quantized LLMs.",
    "What is the difference between BF16 and INT4 inference?",
    "Give a concise explanation of GPTQ quantization.",
    "Why does vLLM improve inference performance?",
    "What can go wrong when a model is quantized too aggressively?",
]
SEARCH_MAX_NEW_TOKENS = 128
SEARCH_WARMUP_RUNS = 1
SEARCH_MEASURE_RUNS = 3

SPEED_WEIGHT = 0.80
QUALITY_WEIGHT = 0.20

QWEN3_IGNORE = [
    "lm_head",  # Dont want to mess up the final predictions
    "re:.*embed_tokens.*",  # Dont need to quantise the embeddings mechanism
    "re:.*norm.*",  # Sensitive to quantisation
]

# Search space
quant_schemes = ["W4A16", "W8A16", "FP8_DYNAMIC", "FP8_BLOCK", "FP8"]

MIXED_PRECISION_SCHEMES = {
    "attention":     ["W8A16", "FP8", "FP8_BLOCK", "FP8_DYNAMIC"],
    "mlp_expansion": ["W4A16", "W8A16", "FP8_BLOCK"],
    "mlp_down":      ["W8A16", "FP8", "FP8_BLOCK", "FP8_DYNAMIC"],
}

search_space = {
    "schemes": quant_schemes,
    "targets": ["Linear"],
    "group_size": [32, 64, 128],
    "use_mixed_precision": [True, False],
    "mp_attention_scheme": MIXED_PRECISION_SCHEMES["attention"],
    "mp_mlp_expansion_scheme": MIXED_PRECISION_SCHEMES["mlp_expansion"],
    "mp_mlp_down_scheme": MIXED_PRECISION_SCHEMES["mlp_down"],
}

MIXED_PRECISION_TARGETS = {
    "attention": [
        "re:.*self_attn.q_proj.*",
        "re:.*self_attn.k_proj.*",
        "re:.*self_attn.v_proj.*",
        "re:.*self_attn.o_proj.*",
    ],
    "mlp_expansion": [
        "re:.*gate_proj.*",
        "re:.*up_proj.*",
    ],
    "mlp_down": [
        "re:.*down_proj.*",
    ],
}

FP8_SCHEMES_DICT = {"FP8", "FP8_BLOCK", "FP8_DYNAMIC"}
INT_SCHEMES_DICT  = {"W4A16", "W8A16"}



# Functions
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Prints CPU MEM not GPU MEM
def print_memory(tag: str):
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1024**3
    sys_mem = psutil.virtual_memory()
    avail_gb = sys_mem.available / 1024**3
    total_gb = sys_mem.total / 1024**3
    if avail_gb < 0.5:
        print("!!!!!!!!!!!!!!!!! WARNING MEGA MEMORY USAGE !!!!!!!!!!!!!!!!!!!!!!!!!")
    print(
        f"[{tag}] Process RSS: {rss_gb:.2f} GB | "
        f"System: {total_gb - avail_gb:.2f}/{total_gb:.2f} GB used"
    )



# To tackle the CPU and GPU memory leakage problem, associated with using vllm for search
def _kill_orphan_workers() -> None:
    current = psutil.Process(os.getpid())
    for child in current.children(recursive=True):
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(current.children(recursive=True), timeout=5)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass



# Call after each memory intensive part
def hard_cleanup():
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    _kill_orphan_workers()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()


# Configs taken from https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py#L370
def build_fp8_config_group(scheme: str, targets: list, block_size: int = 128) -> dict:
    if scheme == "FP8_DYNAMIC":
        return {
            "targets": targets,
            "weights": {
                "num_bits": 8, "type": "float", "strategy": "channel",
                "dynamic": False, "symmetric": True,
            },
            "input_activations": {
                "num_bits": 8, "type": "float", "strategy": "token",
                "dynamic": True, "symmetric": True,
            },
        }
    elif scheme == "FP8_BLOCK":
        return {
            "targets": targets,
            "weights": {
                "num_bits": 8, "type": "float", "strategy": "block",
                "dynamic": False, "symmetric": True,
                "block_structure": [block_size, block_size],
            },
            "input_activations": {
                "num_bits": 8, "type": "float", "strategy": "group",
                "dynamic": True, "symmetric": True, "group_size": 128,
            },
        }
    else:  # FP8_STATIC
        return {
            "targets": targets,
            "weights": {
                "num_bits": 8, "type": "float", "strategy": "tensor",
                "dynamic": False, "symmetric": True,
            },
            "input_activations": {
                "num_bits": 8, "type": "float", "strategy": "tensor",
                "dynamic": False, "symmetric": True,
            },
        }





def evaluate_qwen(
    model_path: str | Path,
    encodings: dict,
    num_samples: int = EVAL_SAMPLES,
    quant: str | None = "compressed-tensors",) -> float:
    input_ids_list = encodings["input_ids"][:num_samples]
    prompts = tokenizer.batch_decode(input_ids_list, skip_special_tokens=False)

    try:
        llm = LLM(
            model=str(model_path),
            dtype="bfloat16", # bf16 > fp16
            max_model_len=MAX_SEQUENCE_LENGTH,
            quantization=quant,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_logprobs=TOP_K_LOGPROBS,
            trust_remote_code=True,
            load_format="safetensors", # Forces direct-to-GPU memory mapping
        )

        sampling_params = SamplingParams(
            temperature=0, # Deterministic, greedy
            max_tokens=1,
            prompt_logprobs=TOP_K_LOGPROBS,
        )


        # Compute outputs
        outputs = llm.generate(prompts, sampling_params)

        total_nll = 0.0
        total_tokens = 0
        missing_tokens = 0

        for output, input_ids in zip(outputs, input_ids_list):
            prompt_logprobs = output.prompt_logprobs
            if not prompt_logprobs:
                continue

            # Position 0 has nothing before it → skip with [1:]
            for token_id, lp_dict in zip(input_ids[1:], prompt_logprobs[1:]):

                if lp_dict is None:
                    continue
                if token_id in lp_dict:
                    total_nll -= lp_dict[token_id].logprob
                else:
                    # Penalise tokens outside top-k rather than skipping
                    total_nll -= MISSING_TOKEN_LOGPROB # --10 = 10 penalty
                    missing_tokens += 1 
                total_tokens += 1

    finally:
        # Kill everything to recover the memory, runs fine a few times but memory begins stacking for no reason
        if llm is not None:
            try:
                if hasattr(llm, "llm_engine") and \
                   hasattr(llm.llm_engine, "engine_core"):
                    llm.llm_engine.engine_core.shutdown()
            except Exception as e:
                print(f" failed to llm.llm_engine.engine_core.shutdown()")
            finally:
                #hard_cleanup()
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                destroy_model_parallel()
                destroy_distributed_environment()
                del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    missing_pct = 100 * missing_tokens / max(total_tokens, 1)
    ppl = float(np.exp(total_nll / max(total_tokens, 1)))

    print(
        f"{Path(model_path).name}   |   "
        f"tokens={total_tokens} |   missing={missing_pct:.1f}%  |   PPL={ppl:.3f}"
    )
    return ppl


def benchmark_vllm_output_tps(model_ref: str, prompts: list[str], quantization: str | None,
    max_new_tokens: int = SEARCH_MAX_NEW_TOKENS,
    warmup_runs: int = SEARCH_WARMUP_RUNS,
    measure_runs: int = SEARCH_MEASURE_RUNS,) -> float:

    """
    Returns mean output tokens/sec over repeated vLLM runs
    """
    hard_cleanup()

    llm = LLM(
        model=model_ref,
        dtype="bfloat16",
        gpu_memory_utilization=0.75,
        max_model_len=1024,
        max_num_batched_tokens=2048,
        quantization=quantization,
        trust_remote_code=False,
        disable_log_stats=True,
        load_format="safetensors",
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        ignore_eos=True,
    )

    # Warmup to avoid first test outlier
    for _ in range(warmup_runs):
        _ = llm.generate(prompts, sp)

    tps_values = []
    for _ in range(measure_runs):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        t1 = time.perf_counter()
        wall = t1 - t0
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps_values.append(out_tokens / wall)

    del llm
    hard_cleanup()
    return float(np.mean(tps_values))


# Recipe builder for uniform and non uniform quantisation
def build_recipe_from_trial(trial) -> tuple:
    """
    Returns tuple(recipe_list, label, use_mp, group_size)

    """
    # group_size = trial.suggest_categorical("group_size", search_space["group_size"]) no point searching without custom compressed-tensor kernels each. cant even inference
    use_mp = trial.suggest_categorical("use_mixed_precision", search_space["use_mixed_precision"])

    # Uniform 
    if not use_mp:
        scheme = trial.suggest_categorical("scheme", search_space["schemes"])
        label = scheme
        if scheme in INT_SCHEMES_DICT:
            recipe = [GPTQModifier(
                scheme=scheme,
                targets=search_space["targets"],
                ignore=QWEN3_IGNORE,
                dampening_frac=DAMPENING_FRAC,
            )]
        else:
            recipe = [QuantizationModifier(
                config_groups={"group_0": build_fp8_config_group(scheme, ["Linear"])},
                ignore=QWEN3_IGNORE,
            )]
        return recipe, label, use_mp #, group_size

    # Mixed precision (non unfirm)
    attention_scheme = trial.suggest_categorical(
        "mp_attention_scheme", search_space["mp_attention_scheme"]
    )

    mlp_expansion_scheme = trial.suggest_categorical(
        "mp_mlp_expansion_scheme", search_space["mp_mlp_expansion_scheme"]
    )

    mlp_down_scheme = trial.suggest_categorical(
        "mp_mlp_down_scheme", search_space["mp_mlp_down_scheme"]
    )

    label = f"mp_{attention_scheme}_{mlp_expansion_scheme}_{mlp_down_scheme}"


    mp_groups = {
        "attention":     (attention_scheme,     MIXED_PRECISION_TARGETS["attention"]),
        "mlp_expansion": (mlp_expansion_scheme, MIXED_PRECISION_TARGETS["mlp_expansion"]),
        "mlp_down":      (mlp_down_scheme,      MIXED_PRECISION_TARGETS["mlp_down"]),
    }

    fp8_config_groups = {}
    int_groups = {}
    fp8_idx = 0
    recipe = []

    for group_name, (scheme, targets) in mp_groups.items():
        if scheme in FP8_SCHEMES_DICT:
            # build fp8 config
            fp8_config_groups[f"group_{fp8_idx}"] = build_fp8_config_group(
                scheme, targets, block_size=128 # keep for now static as vllm only supports
            )
            fp8_idx += 1
        else:
            int_groups.setdefault(scheme, []).extend(targets)


    # quantizationmod skips calibration using scaling from weights tensors

    if fp8_config_groups:
        recipe.append(QuantizationModifier(
            config_groups=fp8_config_groups,
            ignore=QWEN3_IGNORE,
        ))

    # gptq requires calibration data for calculating the most optimal scaling based on 
    if int_groups:
        for scheme, targets in int_groups.items():
            recipe.append(GPTQModifier(
                scheme=scheme,
                targets=targets,
                ignore=QWEN3_IGNORE,
                dampening_frac=DAMPENING_FRAC,
            ))

    return recipe, label, use_mp


# composite objective
def objective(trial: optuna.Trial) -> float:

    print_memory(f"Trial {trial.number} start") # Debug memory use

    recipe, label, use_mp = build_recipe_from_trial(trial)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    print_memory(f"Trial {trial.number}: After model load")

    has_gptq = any(isinstance(m, GPTQModifier) for m in recipe)

    # use sequential if mixed precision (eg mixed INT + FP8) or int schemes, otherwise data free for fp8 configs
    # sequential is used in multiple modifiers example in llm compressor, quantizationmodifier does not need data, so datafree
    pipeline = "sequential" if (len(recipe) > 1 or has_gptq) else "datafree"

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        pipeline=pipeline,
    )
    print_memory("After oneshot")

    MODEL_SAVE_DIR = SAVE_DIR / "base" / f"trial_{trial.number}_{label}"

    model.save_pretrained(
        MODEL_SAVE_DIR,
        save_compressed=True,
        quantization_format=None,
    )
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print(f"Model saved to {MODEL_SAVE_DIR}")


    model_size_bytes = sum(
        f.stat().st_size for f in MODEL_SAVE_DIR.glob("*.safetensors")
    )
    model_size_gb = model_size_bytes / (1024**3)

    del model, recipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_memory("After cleanup")

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning: {e}")

    # Quality
    eval_perplexity = evaluate_qwen(str(MODEL_SAVE_DIR), encodings=encodings)

    # Speed
    trial_tps = benchmark_vllm_output_tps(
        model_ref=str(MODEL_SAVE_DIR),
        prompts=SEARCH_PROMPTS,
        quantization="compressed-tensors",
    )

    # Normalize against BF16 baseline
    speed_gain = (trial_tps / BASELINE_TPS) - 1.0
    ppl_penalty = max(0.0, (eval_perplexity / BASELINE_PPL) - 1.0)
    objective_value = -(SPEED_WEIGHT * speed_gain) + (QUALITY_WEIGHT * ppl_penalty)

    print(
        f"Trial {trial.number} | MP={use_mp} | Scheme={label} | "
        f"PPL={eval_perplexity:.3f} | "
        f"TPS={trial_tps:.3f} | "
        f"SpeedGain={(100*speed_gain):+.2f}% | "
        f"PPLPenalty={(100*ppl_penalty):+.2f}% | "
        f"Obj={objective_value:.5f} | "
        f"Size={model_size_gb:.3f} GB"
    )

    trial.set_user_attr("scheme", label)
    trial.set_user_attr("use_mixed_precision", use_mp)
    trial.set_user_attr("dampening_frac", DAMPENING_FRAC)
    trial.set_user_attr("model_size_gb", model_size_gb)
    trial.set_user_attr("trial_tps", trial_tps)
    trial.set_user_attr("trial_ppl", eval_perplexity)
    trial.set_user_attr("speed_gain_pct", 100 * speed_gain)
    trial.set_user_attr("ppl_penalty_pct", 100 * ppl_penalty)

    gc.collect()
    torch.cuda.empty_cache()
    return objective_value


# Trial logging
def log_trial(trial: optuna.trial.FrozenTrial) -> None:
    row = {
        "number": trial.number,
        "state": str(trial.state),
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
    }
    with open(TRIAL_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


def logging_callback(study, trial):
    log_trial(trial)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna quantisation search")
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)",
    )
    args = parser.parse_args()

    set_seed(42)

    # Load tokenizer
    #print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, fix_mistral_regex=True)

    # Calibration Dataset (ultrachat's chat distribution)
    #print("Loading calibration dataset")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False,
            )
        }

    def tokenize_calibration(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(preprocess)
    ds = ds.map(tokenize_calibration, remove_columns=ds.column_names)
    #print(f"Calibration samples: {len(ds)}")

    # Eval dataset
    #print("Loading eval dataset")
    wikitext_eval = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # because we will evaluate on a split of a split we need filtering
    # Token based filter for to select samples with enough context
    eval_texts = []
    for text in wikitext_eval["text"]:
        if not text.strip():
            continue
        token_len = len(tokenizer.encode(text, add_special_tokens=False))
        if token_len >= 64: # At least 64 real tokens
            eval_texts.append(text)
        if len(eval_texts) == EVAL_SAMPLES: # Get 150 samples maybe 200 change later
            break

    del wikitext_eval
    gc.collect()

    # Verify 150 samples
    print(f"Eval samples after token filter: {len(eval_texts)}")

    encodings = tokenizer(
        eval_texts,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding=False,
        add_special_tokens=False,
        return_tensors=None,
    )
    del eval_texts
    gc.collect()

    # Verify 150 samples
    """   
    print(f"Encodings: {len(encodings['input_ids'])} sequences")
    print(
        f"Avg token length: "
        f"{sum(len(x) for x in encodings['input_ids']) / len(encodings['input_ids']):.1f}"
    )
    
    """

    # Calculate the baseline on the hpc
    print("Computing BF16 baselines:    ")
    hard_cleanup()
    BASELINE_TPS = benchmark_vllm_output_tps(
        model_ref=MODEL_ID,
        prompts=SEARCH_PROMPTS,
        quantization=None,
    )
    hard_cleanup()
    BASELINE_PPL = evaluate_qwen(MODEL_ID, encodings=encodings, quant=None)
    hard_cleanup()

    print(f"BASELINE_TPS = {BASELINE_TPS:.3f} output_tok/s")
    print(f"BASELINE_PPL = {BASELINE_PPL:.3f}")

    print("baseline complete !")


    # NAS
    print(f"Starting Optuna search")
    study = optuna.create_study(
        direction="minimize",
        study_name="gptq-hparam-search",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials, callbacks=[logging_callback])

    # SUmmary
    print("-------------------------------------------------------------")
    print("Best Trial")
    best = study.best_trial # Best trial assuming it computes all
    print(f"Number      : {best.number}")
    print(f"Objective   : {best.value:.5f}")
    print(f"Params      : {best.params}")
    print(f"User attrs  : {best.user_attrs}")
    print("-------------------------------------------------------------")

    # Save full study summary
    summary_path = LOG_DIR / "study_summary.json"
    summary = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStudy summary saved to {summary_path}")
    print(f"Trial log saved to {TRIAL_LOG}")

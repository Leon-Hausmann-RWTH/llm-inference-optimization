# LLM Inference Optimisation
This repository contains the implementation and evaluation code for our project on large language model inference optimization. The repository focuses on two complementary system components: an optimization pipeline that applies post-training quantization on top of vLLM and benchmarks the resulting speed–quality trade-off, and an Optuna-based search pipeline that explores efficient mixed-precision configurations under an explicit composite objective.

The project is motivated by a simple systems question: can LLM inference be made faster without incurring unacceptable degradation in predictive quality? Instead of treating serving runtime, numerical precision, and evaluation as separate issues, this repository studies them together as one inference system design problem.

## Purpose of the System

The purpose of this system is to design and evaluate an LLM inference pipeline that improves serving efficiency while preserving acceptable predictive quality. In modern LLM deployment, runtime efficiency is not determined by one factor alone. Performance depends on the interaction between the serving runtime, the numerical representation of model weights and activations, the workload shape, and the evaluation criteria used to judge quality loss. This repository therefore investigates two complementary optimization paths.

The first path uses vLLM for efficient inference asks whether an already optimized inference runtime can be improved further through post-training quantization. We start from vLLM as a strong serving baseline and test whether quantizing model weights yields additional gains in throughput and latency while keeping degradation in prediction quality under control. The second path asks whether manual selection of precision settings can be replaced or improved by automated search. Since mixed-precision design spaces are large and often expensive to explore exhaustively, we use Optuna to search for configurations that provide a better speed–quality trade-off than fixed hand-crafted settings. Together, these two components turn the repository into more than a benchmark collection: it is a practical framework for studying inference optimization as a systems problem.

## Overall Architecture

At a high level, the system is organized around two interacting pipelines. The first is the quantization and benchmarking pipeline. A pretrained model checkpoint is used as the reference baseline. The same model is then quantized using a post-training quantization method and benchmarked under the same runtime and workload conditions. This produces directly comparable measurements of throughput, latency, and quality. The logic of this pipeline is to load a pretrained baseline model, evaluate it under a fixed setup, quantize the same model into a lower-precision representation, evaluate the quantized variant under the same setup, and then compare the two. This pipeline answers the question of how much additional inference benefit can be obtained by compressing the model representation on top of a strong runtime baseline.

The second is the mixed-precision search pipeline. The Optuna component treats precision configuration as a search problem. Instead of evaluating every possible combination, the system samples candidate configurations, evaluates them, and scores them with a composite objective that balances speed against quality degradation. In practice, the pipeline defines a search space, samples a trial configuration, runs the configured model on a benchmark workload, measures throughput and perplexity-related quality, computes a scalar objective, logs the result, and continues the search. This pipeline answers the question of which configurations are meaningfully faster while remaining acceptably close to baseline quality.

## Optimisation Pipeline: Quantization on top of vLLM

We selected vLLM as the first layer of optimisation regarding runtime because it is specifically designed for efficient LLM serving. It improves throughput through runtime-level optimizations such as request scheduling and efficient KV-cache management. However, even a strong serving runtime still has to store and move large model weights. This means that runtime-level optimization alone does not eliminate the memory and bandwidth cost of the model representation itself. For this reason, we add a second optimization layer: post-training quantization.

The quantization pipeline compares a baseline FP16 or BF16 model and a quantized model derived from the same checkpoint. The repository uses GPTQ W4A16 as the main quantization method in the benchmarking workflow. This choice is deliberate because GPTQ is a widely used post-training quantization method for transformer models, it allows compression without retraining, and it provides a realistic low-bit baseline for deployment-oriented evaluation. To keep the comparison fair, the system controls the model family, the hardware, the runtime, the prompts, the workload settings, and the benchmark procedure across both baseline and quantized runs. Only the numerical representation of the model is changed. This design is important because it isolates the effect of quantization from unrelated sources of variation.

The quantization study is also run across multiple model sizes rather than only a single checkpoint because a method that helps only one scale is less convincing as a general inference optimization strategy. Evaluating multiple sizes helps answer whether the gain grows as models become larger and more memory-bound.

The quantization evaluation uses two levels of benchmarking. The first level is a controlled in-process benchmark. This benchmark varies workload parameters such as batch size, the number of generated tokens, and, depending on the experiment, the input/output workload shape. Its purpose is to isolate the direct effect of quantization on generation performance. Because it minimizes serving overhead, it is useful for studying the raw speed–quality trade-off of the numerical representation. The second level is a more realistic end-to-end evaluation through the serving stack. This benchmark measures runtime behavior under true request execution and captures effects such as request scheduling, server overhead, token streaming behavior, and time-to-first-token. This second benchmark is necessary because microbenchmarks may overestimate deployment gains. A quantized model may look very strong in a controlled loop, but the realized improvement in a real serving environment can be smaller. By combining both layers, the methodology avoids drawing conclusions from one evaluation regime alone.


## Performance Metrics

The system uses three classes of performance metrics. These are chosen because inference optimization should be evaluated as a trade-off rather than with a single number. The first class is throughput metrics, such as tokens per second and total token throughput. These metrics measure how much work the system can process per unit time. Since the central goal of the repository is to improve inference efficiency, throughput is the primary performance signal. A serving system that processes more tokens per second can support higher traffic, lower cost per request, or both, so throughput directly captures the main deployment benefit of optimization.

The second class is latency metrics, such as latency per request, time-to-first-token, and end-to-end response delay. These metrics measure responsiveness rather than aggregate capacity. A system that improves throughput but substantially harms latency may be unattractive in practice, especially in interactive settings. Latency therefore complements throughput and ensures that optimization is evaluated from the user-facing perspective as well.

The third class is quality metrics, such as perplexity, pseudo-perplexity proxies, or other comparative quality signals derived from evaluation outputs. These metrics measure how much predictive quality is lost after quantization or precision reduction. Lower precision can introduce numerical error, so without a quality metric, speed improvements would be difficult to interpret because they might simply reflect unacceptable degradation in model performance. Quality metrics therefore ensure that optimization is evaluated against model fidelity rather than speed alone.

### Microbenchmark summary

| Model | Mean throughput speedup | Mean latency improvement | Quality delta (Δ proxy PPL) |
|---|---:|---:|---:|
| Qwen3-0.6B | 1.26× | 20.5% | +1.73 |
| Qwen3-1.7B | 1.85× | 45.9% | +4.56 |
| Qwen3-4B | 2.27× | 55.7% | +0.16 |

### Representative server result

For Qwen3-1.7B, quantization increased average total token throughput from approximately **16.6k** to **20.2k** tokens per second, corresponding to an average serving-side gain of about **1.22×**, while also reducing mean TTFT by about **15.9%**. This confirms that the optimization remains beneficial in a realistic serving environment even after accounting for runtime overhead.

### Interpretation

These results suggest that quantization is not merely a marginal extension to vLLM, but a meaningful second optimization layer. The gain is strongest where model execution is most constrained by memory movement, which explains why the largest model benefits the most. At the same time, the serving benchmark demonstrates that inference evaluation should not rely on controlled throughput measurements alone. A complete methodology must include both controlled and deployment-like evaluation regimes.


## Composite Objective NAS
We employ the the composite objective as a loss to be minimised through optuna trials as:

$$
\mathcal{L} = -\lambda_s \cdot \Delta_s + \lambda_q \cdot \Delta_q
$$

$$
\Delta_s = \frac{\mathrm{TPS}_{\mathrm{trial}}}{\mathrm{TPS}_{\mathrm{BF16}}} - 1,
\qquad
\Delta_q = \max \left(0,\frac{\mathrm{PPL}_{\mathrm{trial}}}{\mathrm{PPL}_{\mathrm{BF16}}} - 1\right)
$$

with $\lambda_s = 0.80$ as `SPEED_WEIGHT` and $\lambda_q = 0.20$ as `QUALITY_WEIGHT`.

For these results, speed is prioritised over next token prediction quality retention. 

## Install Environment
```bash
conda env create -f llm-env.yml
conda activate llm-env
```
Then install a supported version of PyTorch
Ignore conflicts between compressed-tensors and llm-compressor. This is trivial. 

## Usage
```bash
python optuna_search.py --trials 30
```
The favour of the search can be changed by changing `SPEED_WEIGHT` and `QUALITY_WEIGHT`.

## Results

### NAS for Quality and Speed-Aware Composite Objective 
| Trial | Scheme | Loss ↓ | PPL | TPS | Speed Gain | PPL Penalty | Size vs Baseline |
|-------|--------|--------|-----|-----|------------|-------------|------------------|
| #15 | mp_W8A16_W8A16_W8A16 | −0.1665 | 40.53 | 3022 | +24.9% | 16.2% | −50.1% |
| #16 | W8A16 | −0.1640 | 40.33 | 3010 | +24.4% | 15.7% | −50.1% |
| #12 | FP8_DYNAMIC | −0.1354 | 40.54 | 2928 | +21.0% | 16.3% | −50.0% |
| #5 | mp_FP8_DYNAMIC_W8A16_FP8_DYNAMIC | −0.1268 | 40.66 | 2903 | +20.0% | 16.6% | −50.0% |
| #20 | FP8 | −0.1533 | 40.76 | 2985 | +23.4% | 16.9% | −50.0% |

A total of 21 trials were run on the Imperial HPC (L40S GPU + 100GB RAM + 4 Intel Xeon Platinum 8358 CPU Cores)

- **W8A16** — weights stored as INT8, dequantised back to BF16 before the matmul. Activations stay in BF16 during forward pass.
- **FP8** — weights stored as FP8, activations are also computed in FP8 during the forward pass.

Mixed-precision INT8 weights with BF16 activations across all three module groups achieves the lowest loss, and matches uniform W8A16 as expected. W8A16 is expected to perform well since activations are most sensitive to quantisation as they vary with input. 

FP8_DYNAMIC trails by ~4% in speed-up but maintains similar PPL penalty, with the overhead coming from on-the-fly scaling. In contrast, the uniform FP8 scheme recovers the speed-up at the cost of a higher PPL penalty. FP8_BLOCK consistently underperforms across all trials. Unlike NVIDIA Blackwell, which has silicon-level hardware support for block scales within the tensor instruction itself, the L40S (NVIDIA Ada Lovelace) implements block-wise scaling at the kernel level via CUTLASS: weights and activations are fed as FP8 into the Tensor Cores, matmul results are accumulated in FP32 registers, and then block-wise scaling are applied to the accumulators. However, because the model is already memory-bound and small matmuls completes almost instantaneously, overhead of added memory access outweighs the bandwidth savings from halving weight traffic.

## Interesting Results
| Trial | Scheme | PPL | Speed Gain (%) | Size vs Baseline | Comments |
|---|---|---|---|---|---|
| #9 | mp_FP8_DYNAMIC_FP8_BLOCK_FP8 | 3064 | +14.7% | −50.0% | Catastrophic perplexity increase indicating model collapse, |
| #2 | FP8 | 61.9 | +23.0% | −50.0% | Good speed-up. However significant decrease in prediction quality. |
| #4 | mp_FP8_BLOCK_W4A16_FP8_BLOCK | 40.95 | −11.7% | −55.7% | Second smallest model but slowest speed. |
| #5 | mp_FP8_DYNAMIC_W8A16_FP8_DYNAMIC | 40.66 | +20.0% | −50.0% | Best non-uniform modifier configuration. |

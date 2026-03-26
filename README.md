# LLM Inference Optimisation
Inference optimization for autoregressive LLMs through runtime-efficient serving, post-training quantization, and deployment-aware evaluation. Includes GPTQ on vLLM, benchmarking across Qwen3 model sizes, and Optuna-based search for strong latency-throughput-quality trade-offs.

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

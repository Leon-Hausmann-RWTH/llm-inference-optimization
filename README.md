# LLM Inference Optimisation
Inference optimization for autoregressive LLMs through runtime-efficient serving, post-training quantization, and deployment-aware evaluation. Includes GPTQ on vLLM, benchmarking across Qwen3 model sizes, and Optuna-based search for strong latency-throughput-quality trade-offs.

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
## Results

TBD

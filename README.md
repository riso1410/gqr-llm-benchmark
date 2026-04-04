# GQR LLM Benchmark

Benchmarking suite for evaluating LLMs on the [GQR](https://pypi.org/project/gqr/) (Generalization Quality Rating) text classification task. Models are tested across four prompt-engineering strategies and scored on both in-distribution and out-of-distribution accuracy.

## Task

Classify text passages into one of four domains: **law**, **finance**, **healthcare**, or **ood** (out-of-domain). The GQR score is the harmonic mean of ID and OOD accuracy, measuring how well a model generalizes beyond its training distribution.

## Models

| Model | Notes |
|---|---|
| `google/gemma-4-26B-A4B-it` | MoE, active 4B |
| `google/gemma-4-E4B-it` | Dense 4B |
| `Qwen/Qwen3.5-9B` | |
| `Qwen/Qwen3-14B` | |
| `microsoft/phi-4` | 14B |
| `ibm-granite/granite-4.0-tiny-preview` | |
| `mistralai/Mistral-7B-Instruct-v0.3` | |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Loaded in 4-bit |

## Strategies

- **baseline** -- Direct prompting via HuggingFace `generate()` with a system + user prompt.
- **dspy** -- Zero-shot [DSPy](https://dspy.ai/) `Predict` module with a typed signature.
- **gepa** -- [GEPA](https://dspy.ai/) (Grounded Example-based Prompt Augmentation) optimizer that compiles a prompt using training examples and reflective feedback.
- **fewshot** -- `BootstrapFewShotWithRandomSearch` from DSPy, selecting the best few-shot demonstrations from a candidate pool.

## Setup

Requires Python 3.13+ and a CUDA GPU (48 GB VRAM recommended).

```bash
uv sync
```

## Usage

```bash
# full benchmark (all models x all strategies)
uv run python bench.py

# quick sanity check with a handful of samples
uv run python bench.py --dry-run
```

Results are saved to `results/` as timestamped CSVs. Bar plots and latency-vs-accuracy scatter plots are generated automatically.

## Project Structure

```
bench.py          # single-file benchmark script
results/          # CSV results and plots
saves/            # saved DSPy programs (GEPA, fewshot)
pyproject.toml    # dependencies
```

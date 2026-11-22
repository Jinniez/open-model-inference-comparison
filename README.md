# Open-Model Inference Comparison (personal project, Nov 2026)

A small weekend experiment: run three open-weight models from the families
NVIDIA's Open Models team works with (Llama, Qwen, Gemma) on the same five
prompts through Ollama, and compare latency, throughput, and output quality.

## What it does

- Models: `llama3.1:8b`, `qwen2.5:7b`, `gemma2:9b`
- Five prompts: factual, summary, code, reasoning, creative
- Measures wall-clock seconds, tokens generated, tokens/sec per prompt
- Saves every raw output to `results.json` and prints a latency table

## How to run (Mac)

1. Install Ollama from https://ollama.com/download and start it.
2. Pull the models (one-time download, ~4–5 GB each):
   ```
   ollama pull llama3.1:8b
   ollama pull qwen2.5:7b
   ollama pull gemma2:9b
   ```
3. Run the script (Python 3, standard library only — nothing to pip install):
   ```
   python3 ollama-model-comparison.py
   ```
4. Total time: roughly 1–2 hours, mostly model downloads.

## What it is (and isn't)

- It IS a real, runnable experiment: local inference on open-weight models
  with measured latency and saved outputs.
- It is NOT production experience, fine-tuning, or evaluation at scale.
  The honest résumé line is: "Personal project (Nov 2026): compared
  open-weight models (Llama 3.1, Qwen 2.5, Gemma 2) on inference latency
  and output quality using Ollama."

## Publish it

Push this folder to GitHub (account: JinnieZ) so the project is visible:
`open-model-inference-comparison` — script + README + results.json.

## Results (Nov 22, 2026 — MacBook Pro, Ollama, single run)

Latency in seconds (lower is faster):

| prompt | llama3.1:8b | qwen2.5:7b | gemma2:9b |
| factual | 5.66 | 3.28 | 5.99 |
| summary | 2.93 | 2.26 | 2.41 |
| code | 10.63 | 13.99 | 16.76 |
| reasoning | 7.56 | 7.21 | 4.99 |
| creative | 1.53 | 1.18 | 1.66 |

Throughput once warmed up: ~20–26 tokens/sec for all three models.

Takeaways:
- All three answered the factual question correctly (Canberra).
- The time differences come mostly from verbosity, not speed: on the code prompt Gemma generated 445 tokens vs. Llama's 250, so it took longer.
- The first prompt per model is slow (3–6s for 8 tokens) because that call loads the model into memory — cold start, not generation speed.
- Single run (n=1) on a laptop; treat the ranking as directional, not a benchmark.

Full outputs: results.json.

# Open-Model Inference Comparison (personal project, Nov 2026)

A small weekend experiment: run open-weight models from the families
NVIDIA's Open Models team works with (Llama, Qwen, Gemma) on the same
kinds of prompts, two ways — through Ollama, and directly through
Hugging Face Transformers on PyTorch — and compare latency, throughput,
and output quality.

## What it does

### 1. Ollama comparison (`ollama-model-comparison.py`)
- Models: `llama3.1:8b`, `qwen2.5:7b`, `gemma2:9b`
- Five prompts: factual, summary, code, reasoning, creative
- Measures wall-clock seconds, tokens generated, tokens/sec per prompt
- Saves every raw output to `results.json` and prints a latency table

### 2. Hugging Face Transformers + PyTorch (`hf-transformers-inference.py`)
- Model: `Qwen/Qwen2.5-0.5B` (0.5B params, same Qwen family as the 7B above)
- Three prompts: factual, reasoning, creative
- Runs the model with `transformers.pipeline("text-generation")` on
  PyTorch (MPS on Mac, CPU otherwise), greedy decoding (`do_sample=False`)
- Measures wall-clock seconds, generated tokens, tokens/sec per prompt
- Saves every raw output to `hf-results.json`
- Notebook version: `hf-inference.ipynb` (same code, cell by cell)

## How to run (Mac)

### Ollama
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

### Hugging Face Transformers + PyTorch
1. Install (one time):
   ```
   pip3 install torch transformers accelerate
   ```
2. Run:
   ```
   python3 hf-transformers-inference.py
   ```
3. First run downloads Qwen2.5-0.5B (~1 GB) from Hugging Face Hub.

## What it is (and isn't)

- It IS a real, runnable experiment: local inference on open-weight models
  with measured latency and saved outputs, via two different stacks.
- It is NOT production experience, fine-tuning, or evaluation at scale.
  The honest résumé line is: "Personal project (Nov 2026): compared
  open-weight models (Llama 3.1, Qwen 2.5, Gemma 2) on inference latency
  and output quality using Ollama; ran Qwen2.5-0.5B locally with Hugging
  Face Transformers + PyTorch."

## Results — Ollama (Nov 22, 2026 — MacBook Pro, single run)

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

## Results — Hugging Face Transformers + PyTorch (Nov 22, 2026)

Model: Qwen/Qwen2.5-0.5B, greedy decoding, max 80 new tokens.
Stack: torch 2.5.0 + transformers 4.46.0, device MPS (MacBook Pro).
Model load: 328.8s first run (includes ~1 GB download from Hugging Face Hub).

| prompt | seconds | generated tokens | tokens/sec |
| factual | 12.44 | 7 | 0.6 |
| reasoning | 8.78 | 80 | 9.1 |
| creative | 3.48 | 18 | 5.2 |

Takeaways:
- Factual answer correct (Canberra), but very slow on the first generation (0.6 tok/s) — MPS warm-up / first-call overhead, not steady-state speed.
- Reasoning hit the 80-token cap and was cut off mid-sentence — the cap, not the model, stopped it. Raise `max_new_tokens` to finish it.
- Throughput is much lower than the Ollama run (~20–26 tok/s for 7–9B models) because this is a raw Transformers pipeline on MPS with no serving optimisation — exactly the gap vLLM / SGLang exist to close.
- Single run (n=1) on a laptop; treat the numbers as directional, not a benchmark.

Full outputs: hf-results.json.

## Files

- `ollama-model-comparison.py` + `results.json` — Ollama comparison
- `hf-transformers-inference.py` + `hf-inference.ipynb` + `hf-results.json` — HF Transformers + PyTorch inference
- `README.md` — this file

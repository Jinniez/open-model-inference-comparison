#!/usr/bin/env python3
"""Hugging Face Transformers + PyTorch inference (personal project, Nov 2026).

Second half of the open-model comparison: run a small open-weight model
directly through Hugging Face Transformers on PyTorch, instead of through
Ollama. Same idea as ollama-model-comparison.py - same kind of prompts,
measured latency + tokens/sec - so the two runs can be talked about together.

Model: Qwen/Qwen2.5-0.5B (0.5B params, ~1 GB download, runs on CPU/MPS).
Why this one: small enough to run on a laptop, same Qwen family as the
7B model in the Ollama comparison, and a real HF Hub model card.

Prerequisites (one time, ~10-20 min - torch is big):
  pip3 install torch transformers accelerate

Run:
  python3 hf-transformers-inference.py

Output: hf-results.json (full outputs) + a summary table printed here.
"""

import json
import time

MODEL_ID = "Qwen/Qwen2.5-0.5B"

PROMPTS = {
    "factual": "In one sentence, what is the capital of Australia?",
    "reasoning": "A store sells apples at $2 each and oranges at $3 each. If Maya buys 4 apples and 3 oranges and pays with a $20 bill, how much change does she get? Show the steps.",
    "creative": "Write a two-line poem about rain on a city street at night.",
}


def main():
    import torch
    from transformers import pipeline

    # Mac: MPS if available, otherwise CPU. No GPU needed for a 0.5B model.
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"torch {torch.__version__} | device: {device}")

    print(f"Loading {MODEL_ID} ... (first run downloads ~1 GB from Hugging Face)")
    t0 = time.time()
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.float32,
        device=device,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    tokenizer = pipe.tokenizer

    results = {}
    for name, prompt in PROMPTS.items():
        # Count prompt tokens so generated tokens = total - prompt.
        prompt_tokens = len(tokenizer.encode(prompt))
        start = time.time()
        out = pipe(
            prompt,
            max_new_tokens=80,
            do_sample=False,  # greedy = deterministic, same answer every run
            return_full_text=False,
        )
        seconds = time.time() - start
        text = out[0]["generated_text"].strip()
        gen_tokens = len(tokenizer.encode(text))
        tps = gen_tokens / seconds if seconds > 0 else 0
        results[name] = {
            "seconds": round(seconds, 2),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "tokens_per_sec": round(tps, 1),
            "output": text,
        }
        print(f"\n[{name}] {seconds:.1f}s | {gen_tokens} tok | {tps:.1f} tok/s")
        print(f"  -> {text[:200].replace(chr(10), ' ')}")

    with open("hf-results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nFull outputs saved to hf-results.json")

    print("\n## Latency summary (seconds, lower is faster)")
    for name, r in results.items():
        print(f"  {name}: {r['seconds']}s | {r['generated_tokens']} tok | {r['tokens_per_sec']} tok/s")


if __name__ == "__main__":
    main()

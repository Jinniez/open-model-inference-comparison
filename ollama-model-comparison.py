#!/usr/bin/env python3
"""Open-model inference comparison (personal project, Nov 2026).

Compares three open-weight models from the families NVIDIA's Open Models
team works with (Llama, Qwen, Gemma) on the same prompts, using Ollama.
Records latency + tokens/sec + raw outputs so results can be discussed
honestly: this is a small weekend experiment, not production experience.

Prerequisites:
  1. Install Ollama (Mac app from https://ollama.com/download) and start it.
  2. Pull the models (one-time download, a few GB each):
       ollama pull llama3.1:8b
       ollama pull qwen2.5:7b
       ollama pull gemma2:9b
  3. Run:  python3 ollama-model-comparison.py

Output: results.json (full outputs) + a summary table printed here.
Uses only the Python standard library; talks to Ollama on localhost:11434.
"""

import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = ["llama3.1:8b", "qwen2.5:7b", "gemma2:9b"]

PROMPTS = {
    "factual": "In one sentence, what is the capital of Australia?",
    "summary": "Summarize in two sentences: The city council approved a new light-rail line on Tuesday after three years of debate. Supporters say it will cut commute times by 20 minutes, while critics worry about the $2.1 billion cost.",
    "code": "Write a Python function that returns True if a string is a palindrome, ignoring spaces and case.",
    "reasoning": "A store sells apples at $2 each and oranges at $3 each. If Maya buys 4 apples and 3 oranges and pays with a $20 bill, how much change does she get? Show the steps.",
    "creative": "Write a two-line poem about rain on a city street at night.",
}


def run_prompt(model, prompt):
    """Send one prompt to one model; return (text, seconds, tokens, tok_per_sec)."""
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    seconds = time.time() - start
    tokens = data.get("eval_count", 0)
    tps = tokens / seconds if seconds > 0 else 0
    return data.get("response", "").strip(), seconds, tokens, tps


def main():
    results = {}
    for model in MODELS:
        print(f"\n=== {model} ===")
        results[model] = {}
        for name, prompt in PROMPTS.items():
            try:
                text, seconds, tokens, tps = run_prompt(model, prompt)
            except Exception as exc:  # noqa: BLE001 - surface the real error
                print(f"  [{name}] FAILED: {exc}")
                print(f"  (Did you run: ollama pull {model} ?)")
                results[model][name] = {"error": str(exc)}
                continue
            results[model][name] = {
                "seconds": round(seconds, 2),
                "tokens": tokens,
                "tokens_per_sec": round(tps, 1),
                "output": text,
            }
            print(f"  [{name}] {seconds:.1f}s | {tokens} tok | {tps:.1f} tok/s")
            print(f"    -> {text[:160].replace(chr(10), ' ')}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nFull outputs saved to results.json")

    print("\n## Latency summary (seconds, lower is faster)")
    header = "| prompt | " + " | ".join(MODELS) + " |"
    print(header)
    print("|" + "---|" * (len(MODELS) + 1))
    for name in PROMPTS:
        row = [name]
        for model in MODELS:
            entry = results[model].get(name, {})
            row.append(str(entry.get("seconds", "ERR")))
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()

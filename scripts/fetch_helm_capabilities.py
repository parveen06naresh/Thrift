"""Download per-question HELM Capabilities results (Stanford CRFM, release v1.15.0, Nov 2025):
the newest public set of prompts, answers, token counts and scores for frontier models
(GPT-5.1, Claude Sonnet 4.5, Gemini 3 Pro, Grok 4, DeepSeek, Kimi K2, ...).

Scenarios and the per-question quality score HELM reports (all in [0, 1]):
  GPQA (graduate science, CoT correctness), MMLU-Pro (CoT correctness), Omni-MATH (judged
  accuracy), IFEval (strict instruction following), WildBench (real user chats, judge score).

Output: data/helm_frontier.jsonl.gz, one row per question with every model's score and tokens.
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import os
import urllib.parse
import urllib.request

BASE = "https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output"
RELEASE = "v1.15.0"
MODELS = [
    "google_gemini-2.5-flash-lite",
    "openai_gpt-5-nano-2025-08-07",
    "xai_grok-3-mini-beta",
    "openai_gpt-5-mini-2025-08-07",
    "deepseek-ai_deepseek-v3",
    "moonshotai_kimi-k2-instruct",
    "anthropic_claude-haiku-4-5-20251001",
    "openai_gpt-5.1-2025-11-13",
    "google_gemini-3-pro-preview",
    "xai_grok-4-0709",
    "anthropic_claude-sonnet-4-5-20250929",
]
METRIC = {"gpqa": "chain_of_thought_correctness", "mmlu_pro": "chain_of_thought_correctness", "omni_math": "omni_math_accuracy",
          "ifeval": "ifeval_strict_accuracy", "wildbench": "wildbench_score_rescaled"}
OUT = "data/helm_cap"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.read()


def main() -> None:
    r2s = json.loads(get(f"{BASE}/releases/{RELEASE}/runs_to_run_suites.json"))
    jobs = []
    for run, suite in r2s.items():
        scen = run.split(":", 1)[0]
        model = run.split("model=", 1)[1].split(",", 1)[0]
        if scen not in METRIC or model not in MODELS:
            continue
        q = urllib.parse.quote(run, safe="=,:@")
        jobs.append((f"{BASE}/runs/{suite}/{q}/display_predictions.json", f"{OUT}/{model}/{scen}.json"))
        inst = f"{OUT}/instances/{scen}.json"
        if not any(j[1] == inst for j in jobs):
            jobs.append((f"{BASE}/runs/{suite}/{q}/instances.json", inst))

    def fetch(job):
        url, path = job
        if os.path.exists(path):
            return path, "cached"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = get(url)
        except Exception as e:  # noqa: BLE001
            return path, f"error {e}"
        with open(path, "wb") as f:
            f.write(data)
        return path, len(data)

    with cf.ThreadPoolExecutor(12) as ex:
        for path, res in ex.map(fetch, jobs):
            if isinstance(res, str) and res.startswith("error"):
                print(path, res)
    build()


def build() -> None:
    rows = {}
    for fn in sorted(os.listdir(f"{OUT}/instances")):
        scen = fn[:-5]
        for inst in json.load(open(f"{OUT}/instances/{fn}")):
            rows[(scen, inst["id"])] = {"scenario": scen, "id": inst["id"], "input": inst["input"]["text"], "models": {}}
    for model in MODELS:
        for scen in METRIC:
            p = f"{OUT}/{model}/{scen}.json"
            if not os.path.exists(p):
                print("missing", p)
                continue
            for pr in json.load(open(p)):
                row = rows.get((scen, pr["instance_id"]))
                s = pr["stats"]
                if row is None or METRIC[scen] not in s:
                    continue
                row["models"][model] = {"q": float(s[METRIC[scen]]), "in": int(s.get("num_prompt_tokens", 0)),
                                        "out": int(s.get("num_output_tokens", 0))}
    out = [r for r in rows.values() if len(r["models"]) == len(MODELS) and r["input"]]
    with gzip.open("data/helm_frontier.jsonl.gz", "wt") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in out:
        by[r["scenario"]] = by.get(r["scenario"], 0) + 1
    print(f"{len(out)} questions with all {len(MODELS)} models -> data/helm_frontier.jsonl.gz", by)


if __name__ == "__main__":
    main()

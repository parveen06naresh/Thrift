"""Download per-question HELM Lite results used by Thrift's router benchmark.

HELM (Stanford CRFM) publishes every prompt, every model completion, the token
counts and the per-question score. That gives us *real* outcomes for cheap and
strong models on the same questions, so routing can be evaluated offline,
without an API key, and without simulating anything.

Output: data/helm/<model>/<run>.json (raw display_predictions) and
data/helm/instances/<run>.json, plus data/helm_routing.jsonl.gz (compact table).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import urllib.parse
import urllib.request

BASE = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output"
RELEASE = "v1.13.0"

# (cheap, strong) pairs evaluated in the paper.
MODELS = [
    "openai_gpt-4o-mini-2024-07-18",
    "openai_gpt-4o-2024-08-06",
    "meta_llama-3.1-8b-instruct-turbo",
    "meta_llama-3.1-405b-instruct-turbo",
    "anthropic_claude-3-haiku-20240307",
    "anthropic_claude-3-5-sonnet-20241022",
]
# Scenario -> the per-question quality metric HELM reports for it (all in [0, 1]).
METRIC = {
    "mmlu": "exact_match",
    "commonsense": "exact_match",
    "med_qa": "quasi_exact_match",
    "legalbench": "quasi_exact_match",
    "gsm": "final_number_exact_match",
    "math": "math_equiv_chain_of_thought",
    "natural_qa": "f1_score",
    "narrative_qa": "f1_score",
}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def run_url(suite: str, run: str, name: str) -> str:
    return f"{BASE}/runs/{suite}/{urllib.parse.quote(run, safe='=,:@')}/{name}"


def strip_model(run: str) -> str:
    # "stop=none" is a decoding flag HELM added for some models; the questions are identical.
    parts = [p for p in run.split(":", 1)[1].split(",") if not p.startswith(("model=", "stop="))]
    return run.split(":", 1)[0] + ":" + ",".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/helm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r2s = json.loads(get(f"{BASE}/releases/{RELEASE}/runs_to_run_suites.json"))

    jobs = []  # (url, path)
    for run, suite in r2s.items():
        scen = run.split(":", 1)[0]
        model = run.split("model=", 1)[1].split(",", 1)[0]
        if scen not in METRIC or model not in MODELS:
            continue
        key = strip_model(run)
        safe = key.replace("/", "_")
        jobs.append((run_url(suite, run, "display_predictions.json"), f"{args.out}/{model}/{safe}.json"))
        inst = f"{args.out}/instances/{safe}.json"
        if not any(j[1] == inst for j in jobs):
            jobs.append((run_url(suite, run, "instances.json"), inst))
            jobs.append((run_url(suite, run, "display_requests.json"), f"{args.out}/requests/{safe}.json"))

    def fetch(job):
        url, path = job
        if os.path.exists(path):
            return path, "cached"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = get(url)
        except Exception as e:  # noqa: BLE001 - report and continue
            return path, f"error {e}"
        with open(path, "wb") as f:
            f.write(data)
        return path, len(data)

    with cf.ThreadPoolExecutor(16) as ex:
        for path, res in ex.map(fetch, jobs):
            if isinstance(res, str) and res.startswith("error"):
                print(path, res)
    print(f"{len(jobs)} files in {args.out}")
    build_table(args.out)


def build_table(root: str) -> None:
    """Join instances with every model's score and token counts into one row per question."""
    rows = {}
    for fn in sorted(os.listdir(f"{root}/instances")):
        run = fn[:-5]
        scen = run.split(":", 1)[0]
        for inst in json.load(open(f"{root}/instances/{fn}")):
            if inst.get("split") not in ("test", "valid"):  # HELM's evaluation split
                continue
            refs = [r["output"]["text"] for r in inst.get("references", []) if "correct" in r.get("tags", [])]
            rows[(run, inst["id"])] = {
                "run": run, "scenario": scen, "id": inst["id"],
                "input": inst["input"]["text"], "refs": refs, "models": {},
            }
    for model in MODELS:
        d = f"{root}/{model}"
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            run = fn[:-5]
            metric = METRIC[run.split(":", 1)[0]]
            for p in json.load(open(f"{d}/{fn}")):
                row = rows.get((run, p["instance_id"]))
                s = p["stats"]
                if row is None or metric not in s:
                    continue
                row["models"][model] = {
                    "q": float(s[metric]),
                    "in": int(s.get("num_prompt_tokens", 0)),
                    "out": int(s.get("num_output_tokens", 0)),
                }
    out = [r for r in rows.values() if len(r["models"]) == len(MODELS)]
    with gzip.open("data/helm_routing.jsonl.gz", "wt") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} questions with all {len(MODELS)} models -> data/helm_routing.jsonl.gz")


if __name__ == "__main__":
    main()

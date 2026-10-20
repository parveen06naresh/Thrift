"""Multi-provider routing ("ladder") on real HELM Lite outcomes from five vendors.

Same protocol as eval_router.py: random splits into train 50% / calibration 25% / test 25%.
The ladder routes each question to the cheapest model whose scorer says it will match the
strongest (reference) model; one threshold is certified with Learn-then-Test.

Writes results/ladder.json and thrift_llm/data/ladder_helm.json (a shippable ladder).
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thrift_llm.pricing import PRICES  # noqa: E402
from thrift_llm.router import Ladder, calibrate_ladder  # noqa: E402

# HELM model id -> price-table name; each ladder is ordered by real cost per question.
NAMES = {
    "google_gemini-1.5-flash-002": "gemini-1.5-flash-002",
    "openai_gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
    "deepseek-ai_deepseek-v3": "deepseek-v3",
    "anthropic_claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
    "google_gemini-1.5-pro-002": "gemini-1.5-pro-002",
    "mistralai_mistral-large-2407": "mistral-large-2407",
    "openai_gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
    "anthropic_claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
}
LADDERS = {
    "to_gpt4o": ["google_gemini-1.5-flash-002", "openai_gpt-4o-mini-2024-07-18", "deepseek-ai_deepseek-v3",
                 "anthropic_claude-3-5-haiku-20241022", "google_gemini-1.5-pro-002", "mistralai_mistral-large-2407",
                 "openai_gpt-4o-2024-08-06"],
    "to_claude_sonnet": ["google_gemini-1.5-flash-002", "openai_gpt-4o-mini-2024-07-18", "deepseek-ai_deepseek-v3",
                         "google_gemini-1.5-pro-002", "anthropic_claude-3-5-sonnet-20241022"],
}
EPS = [0.01, 0.02, 0.05]
DELTA = 0.1
SPLITS = int(os.environ.get("SPLITS", 20))

ROWS = [json.loads(line) for line in gzip.open("data/helm_multi.jsonl.gz", "rt")]
TEXT = [r["input"] for r in ROWS]


def arrays(models):
    Q = np.array([[r["models"][m]["q"] for m in models] for r in ROWS])
    C = np.array([[(r["models"][m]["in"] * PRICES[NAMES[m]][0] + r["models"][m]["out"] * PRICES[NAMES[m]][1]) / 1e6 for m in models]
                  for r in ROWS])
    return Q, C


def one_split(args):
    name, seed = args
    models = LADDERS[name]
    Q, C = arrays(models)
    idx = np.random.default_rng(seed).permutation(len(ROWS))
    n = len(idx)
    tr, ca, te = idx[: n // 2], idx[n // 2 : 3 * n // 4], idx[3 * n // 4 :]
    lad = Ladder(models).fit([TEXT[i] for i in tr], Q[tr], seed=seed)
    S = lad.scores(TEXT)
    out = []
    rows = np.arange(len(te))
    for eps in EPS:
        cert = calibrate_ladder(S[ca], Q[ca], eps, DELTA)
        k = Ladder.choose(S[te], cert.setting)
        q = Q[te][rows, k]
        cost = C[te][rows, k].sum()
        ref, ref_cost = Q[te, -1], C[te, -1].sum()
        clipped = float(np.clip(ref - q, 0, 1).mean())
        out.append({"eps": eps, "seed": seed, "threshold": cert.setting,
                    "share": np.bincount(k, minlength=len(models)).tolist(),
                    "quality_drop": float(ref.mean() - q.mean()), "clipped_drop": clipped, "violated": clipped > eps,
                    "cost_saved": float(1 - cost / ref_cost)})
    return name, seed, out


def main():
    jobs = [(name, seed) for name in LADDERS for seed in range(SPLITS)]
    with mp.Pool(min(4, os.cpu_count() or 2)) as pool:
        res = pool.map(one_split, jobs)
    summary = {}
    for name, models in LADDERS.items():
        Q, C = arrays(models)
        rows = [r for nm, _, out in res if nm == name for r in out]
        singles = [{"model": NAMES[m], "accuracy": float(Q[:, j].mean()), "cost_per_1k": float(C[:, j].mean() * 1000),
                    "drop_vs_ref": float(Q[:, -1].mean() - Q[:, j].mean()),
                    "clipped_drop_vs_ref": float(np.clip(Q[:, -1] - Q[:, j], 0, 1).mean()),
                    "cost_saved": float(1 - C[:, j].sum() / C[:, -1].sum())} for j, m in enumerate(models)]
        per = []
        for eps in EPS:
            rs = [r for r in rows if r["eps"] == eps]
            share = np.array([r["share"] for r in rs], dtype=float)
            share = (share / share.sum(axis=1, keepdims=True)).mean(axis=0)
            per.append({"eps": eps, "delta": DELTA, "splits": len(rs),
                        "share": {NAMES[m]: float(s) for m, s in zip(models, share)},
                        "quality_drop": float(np.mean([r["quality_drop"] for r in rs])),
                        "clipped_drop": float(np.mean([r["clipped_drop"] for r in rs])),
                        "cost_saved": float(np.mean([r["cost_saved"] for r in rs])),
                        "violation_rate": float(np.mean([r["violated"] for r in rs]))})
        summary[name] = {"models": [NAMES[m] for m in models], "singles": singles, "results": per}
        print(f"\n{name}: reference {NAMES[models[-1]]} acc {Q[:, -1].mean():.3f}, ${C[:, -1].mean() * 1000:.3f}/1k")
        for s in singles:
            print(f"  always {s['model']:28s} acc {s['accuracy']:.3f}  ${s['cost_per_1k']:.3f}/1k  clipped drop {100 * s['clipped_drop_vs_ref']:.1f}")
        for e in per:
            mix = ", ".join(f"{k.split('-20')[0]} {100 * v:.0f}%" for k, v in e["share"].items() if v >= 0.005)
            print(f"  eps={e['eps']:.2f}: drop {100 * e['quality_drop']:+.2f} pts (clipped {100 * e['clipped_drop']:.2f}), "
                  f"cost -{100 * e['cost_saved']:.1f}%, violated {100 * e['violation_rate']:.0f}%  [{mix}]")
    json.dump({"splits": SPLITS, "delta": DELTA, "n_questions": len(ROWS), "ladders": summary}, open("results/ladder.json", "w"), indent=1)

    # Ship the GPT-4o ladder trained on 75%, certified on 25% at eps = 0.02.
    models = LADDERS["to_gpt4o"]
    Q, _ = arrays(models)
    idx = np.random.default_rng(12345).permutation(len(ROWS))
    tr, ca = idx[: 3 * len(idx) // 4], idx[3 * len(idx) // 4 :]
    lad = Ladder([NAMES[m] for m in models]).fit([TEXT[i] for i in tr], Q[tr])
    cert = lad.calibrate([TEXT[i] for i in ca], Q[ca], 0.02, DELTA)
    lad.meta = {"trained_on": "HELM Lite v1.13.0, 8 scenarios", "n_train": len(tr), "n_cal": len(ca),
                "note": "Tier names must match these model names (see examples/multi_provider.toml). Re-calibrate on your own traffic."}
    lad.save("thrift_llm/data/ladder_helm.json")
    print("shipped ladder:", cert.to_dict())


if __name__ == "__main__":
    main()

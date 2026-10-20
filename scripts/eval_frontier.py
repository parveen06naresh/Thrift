"""Certified routing across 2025-generation frontier models from six vendors, on HELM
Capabilities v1.15.0 (Nov 2025): GPQA, MMLU-Pro, Omni-MATH and IFEval, 2,987 questions with
real answers and scores from every model.

Cost model. HELM's token counts are not usable for cost here: Gemini and DeepSeek runs report
no output tokens, and reasoning models report only visible tokens. So every model is charged
for the same tokens on a question (the prompt, counted with o200k, plus Claude Sonnet 4.5's
complete answer length) at its own list price. This compares prices, not verbosity; reasoning
models usually emit extra hidden tokens, so their real cost is higher than shown here.

Writes results/frontier.json and thrift_llm/data/ladder_frontier.json.
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thrift_llm import tokens as tok  # noqa: E402
from thrift_llm.pricing import PRICES  # noqa: E402
from thrift_llm.router import Ladder, calibrate_ladder  # noqa: E402

NAMES = {
    "openai_gpt-5-nano-2025-08-07": "gpt-5-nano",
    "google_gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "xai_grok-3-mini-beta": "grok-3-mini",
    "deepseek-ai_deepseek-v3": "deepseek-v3",
    "openai_gpt-5-mini-2025-08-07": "gpt-5-mini",
    "moonshotai_kimi-k2-instruct": "kimi-k2-instruct",
    "anthropic_claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "openai_gpt-5.1-2025-11-13": "gpt-5.1",
    "google_gemini-3-pro-preview": "gemini-3-pro-preview",
    "xai_grok-4-0709": "grok-4",
    "anthropic_claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
}
CHEAP = list(NAMES)[:7]
LADDERS = {
    "to_gemini_3_pro": CHEAP + ["openai_gpt-5.1-2025-11-13", "google_gemini-3-pro-preview"],
    "to_claude_sonnet_4_5": CHEAP + ["anthropic_claude-sonnet-4-5-20250929"],
    "to_grok_4": CHEAP + ["xai_grok-4-0709"],
}
EPS = [0.01, 0.02, 0.05]
DELTA = 0.1
SPLITS = int(os.environ.get("SPLITS", 20))
REF_LEN = "anthropic_claude-sonnet-4-5-20250929"

ROWS = [json.loads(line) for line in gzip.open("data/helm_frontier.jsonl.gz", "rt")]
TEXT = [r["input"] for r in ROWS]
TIN = np.array([tok.count(t) + 10 for t in TEXT], dtype=float)
TOUT = np.array([max(1, r["models"][REF_LEN]["out"]) for r in ROWS], dtype=float)


def arrays(models):
    Q = np.array([[r["models"][m]["q"] for m in models] for r in ROWS])
    C = np.stack([(TIN * PRICES[NAMES[m]][0] + TOUT * PRICES[NAMES[m]][1]) / 1e6 for m in models], axis=1)
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
        ref = Q[te, -1]
        clipped = float(np.clip(ref - q, 0, 1).mean())
        out.append({"eps": eps, "seed": seed, "share": np.bincount(k, minlength=len(models)).tolist(),
                    "quality_drop": float(ref.mean() - q.mean()), "clipped_drop": clipped, "violated": clipped > eps,
                    "cost_saved": float(1 - C[te][rows, k].sum() / C[te, -1].sum()),
                    "off_flagship": float(np.mean(k != len(models) - 1))})
    return name, seed, out


def main():
    jobs = [(name, seed) for name in LADDERS for seed in range(SPLITS)]
    with mp.Pool(min(4, os.cpu_count() or 2)) as pool:
        res = pool.map(one_split, jobs)
    scen = np.array([r["scenario"] for r in ROWS])
    summary = {}
    for name, models in LADDERS.items():
        Q, C = arrays(models)
        rows = [r for nm, _, out in res if nm == name for r in out]
        singles = [{"model": NAMES[m], "accuracy": float(Q[:, j].mean()), "cost_per_1k": float(C[:, j].mean() * 1000),
                    "by_scenario": {s: float(Q[scen == s, j].mean()) for s in sorted(set(scen))},
                    "drop_vs_ref": float(Q[:, -1].mean() - Q[:, j].mean()),
                    "clipped_drop_vs_ref": float(np.clip(Q[:, -1] - Q[:, j], 0, 1).mean()),
                    "cost_saved": float(1 - C[:, j].sum() / C[:, -1].sum())} for j, m in enumerate(models)]
        per = []
        for eps in EPS:
            rs = [r for r in rows if r["eps"] == eps]
            share = np.array([r["share"] for r in rs], dtype=float)
            share = (share / share.sum(axis=1, keepdims=True)).mean(axis=0)
            per.append({"eps": eps, "delta": DELTA, "splits": len(rs), "share": {NAMES[m]: float(s) for m, s in zip(models, share)},
                        "quality_drop": float(np.mean([r["quality_drop"] for r in rs])),
                        "clipped_drop": float(np.mean([r["clipped_drop"] for r in rs])),
                        "cost_saved": float(np.mean([r["cost_saved"] for r in rs])),
                        "off_flagship": float(np.mean([r["off_flagship"] for r in rs])),
                        "violation_rate": float(np.mean([r["violated"] for r in rs]))})
        summary[name] = {"models": [NAMES[m] for m in models], "singles": singles, "results": per}
        print(f"\n{name}: reference {NAMES[models[-1]]} acc {Q[:, -1].mean():.3f}, ${C[:, -1].mean() * 1000:.3f}/1k")
        for s in singles:
            print(f"  always {s['model']:22s} acc {s['accuracy']:.3f}  ${s['cost_per_1k']:.3f}/1k  clipped drop {100 * s['clipped_drop_vs_ref']:.1f}")
        for e in per:
            mix = ", ".join(f"{k} {100 * v:.0f}%" for k, v in e["share"].items() if v >= 0.01)
            print(f"  eps={e['eps']:.2f}: drop {100 * e['quality_drop']:+.2f} pts (clipped {100 * e['clipped_drop']:.2f}), cost -{100 * e['cost_saved']:.1f}%, "
                  f"off flagship {100 * e['off_flagship']:.0f}%, violated {100 * e['violation_rate']:.0f}%  [{mix}]")
    json.dump({"source": "HELM Capabilities v1.15.0", "splits": SPLITS, "delta": DELTA, "n_questions": len(ROWS),
               "scenarios": {s: int((scen == s).sum()) for s in sorted(set(scen))},
               "cost_model": "same tokens for every model: o200k prompt count + Claude Sonnet 4.5 answer length, at each model's list price",
               "prices": {NAMES[m]: PRICES[NAMES[m]] for m in NAMES}, "ladders": summary}, open("results/frontier.json", "w"), indent=1)

    models = LADDERS["to_gemini_3_pro"]
    Q, _ = arrays(models)
    idx = np.random.default_rng(12345).permutation(len(ROWS))
    tr, ca = idx[: 3 * len(idx) // 4], idx[3 * len(idx) // 4 :]
    lad = Ladder([NAMES[m] for m in models]).fit([TEXT[i] for i in tr], Q[tr])
    cert = lad.calibrate([TEXT[i] for i in ca], Q[ca], 0.02, DELTA)
    lad.meta = {"trained_on": "HELM Capabilities v1.15.0 (GPQA, MMLU-Pro, Omni-MATH, IFEval)", "n_train": len(tr), "n_cal": len(ca),
                "note": "Tier names must match these model names (see examples/frontier.toml). Re-calibrate on your own traffic."}
    lad.save("thrift_llm/data/ladder_frontier.json")
    print("shipped frontier ladder:", cert.to_dict())


if __name__ == "__main__":
    main()

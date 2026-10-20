"""Drift guard benchmark on real HELM Lite outcomes.

A ladder (Gemini 1.5 Flash -> GPT-4o-mini -> DeepSeek-V3 -> Claude 3.5 Haiku -> Gemini 1.5 Pro
-> Mistral Large -> GPT-4o) is trained and certified at epsilon = 0.02, then serves a stream of
held-out questions. Three worlds:

  no_drift     the stream looks like the calibration data (any alarm is a false alarm)
  model_swap   at request 2,000 the providers behind GPT-4o-mini and DeepSeek-V3 silently start
               answering like Llama-3.1-8B (a quiet downgrade; the prompts do not change)
  topic_shift  the ladder never saw maths (math, gsm); at request 2,000 traffic becomes all maths

The judge is HELM's own per-question metric (loss = how much worse the chosen model scored than
GPT-4o), i.e. a perfect judge; an LLM judge adds noise. Shadow rates 5%, 10%, 20%; 100 streams each.

Writes results/drift.json.
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_ladder import LADDERS, NAMES  # noqa: E402
from thrift_llm.drift import RiskMonitor  # noqa: E402
from thrift_llm.pricing import PRICES  # noqa: E402
from thrift_llm.router import Ladder, calibrate_ladder  # noqa: E402

EPS, DELTA, ALPHA = 0.02, 0.1, 0.05
T, SHIFT, RUNS = 6000, 2000, int(os.environ.get("RUNS", 100))
RATES = [0.05, 0.10, 0.20]
MODELS = LADDERS["to_gpt4o"]
SWAP = {"openai_gpt-4o-mini-2024-07-18": "meta_llama-3.1-8b-instruct-turbo", "deepseek-ai_deepseek-v3": "meta_llama-3.1-8b-instruct-turbo"}
MATHS = {"math", "gsm"}

ROWS = [json.loads(line) for line in gzip.open("data/helm_multi.jsonl.gz", "rt")]
TEXT = [r["input"] for r in ROWS]
Q = np.array([[r["models"][m]["q"] for m in MODELS] for r in ROWS])
QSWAP = np.array([[r["models"][SWAP.get(m, m)]["q"] for m in MODELS] for r in ROWS])
C = np.array([[(r["models"][m]["in"] * PRICES[NAMES[m]][0] + r["models"][m]["out"] * PRICES[NAMES[m]][1]) / 1e6 for m in MODELS] for r in ROWS])
SCEN = np.array([r["scenario"] for r in ROWS])


def build(train_mask, seed=0):
    """Train on 50% and certify on 25% of the allowed rows (the rest stays held out); choices for every row."""
    idx = np.flatnonzero(train_mask)
    idx = np.random.default_rng(seed).permutation(idx)
    tr, ca = idx[: len(idx) // 2], idx[len(idx) // 2 : 3 * len(idx) // 4]
    lad = Ladder(MODELS).fit([TEXT[i] for i in tr], Q[tr], seed=seed)
    S = lad.scores(TEXT)
    cert = calibrate_ladder(S[ca], Q[ca], EPS, DELTA)
    return Ladder.choose(S, cert.setting), set(tr) | set(ca), cert


def simulate(choice, pre_pool, post_pool, Qpost, rate, rng):
    """One stream: returns (alarm index or None, realised risk after the shift with and without the guard, shadow cost share)."""
    rows = np.concatenate([rng.choice(pre_pool, SHIFT), rng.choice(post_pool, T - SHIFT)])
    mon = RiskMonitor(EPS, ALPHA)
    ref = len(MODELS) - 1
    alarm = None
    loss_guard, loss_free, cost, shadow_cost = [], [], 0.0, 0.0
    for t, i in enumerate(rows):
        q = Qpost if t >= SHIFT else Q
        k = choice[i]
        loss = float(np.clip(q[i, ref] - q[i, k], 0, 1))
        loss_free.append(loss)
        served = ref if alarm is not None else k
        loss_guard.append(float(np.clip(q[i, ref] - q[i, served], 0, 1)))
        cost += C[i, served]
        if alarm is not None:
            continue
        if k == ref:
            z, judged = 0.0, False
        elif rng.random() < rate:
            z, judged = loss / rate, True
            shadow_cost += C[i, ref]
        else:
            z, judged = 0.0, False
        if mon.observe(z, judged):
            alarm = t
    post = slice(SHIFT, T)
    return alarm, float(np.mean(loss_free[post])), float(np.mean(loss_guard[post])), shadow_cost / cost


def main():
    out = {"epsilon": EPS, "delta": DELTA, "alpha": ALPHA, "stream_length": T, "shift_at": SHIFT, "runs": RUNS, "worlds": {}}
    allrows = np.ones(len(ROWS), bool)
    choice_all, used_all, cert_all = build(allrows)
    pool_all = np.setdiff1d(np.arange(len(ROWS)), list(used_all))  # held-out questions only
    nomath = ~np.isin(SCEN, list(MATHS))
    choice_nm, used_nm, cert_nm = build(nomath)
    pool_nm = np.setdiff1d(np.flatnonzero(nomath), list(used_nm))
    pool_math = np.flatnonzero(~nomath)
    worlds = {
        "no_drift": (choice_all, pool_all, pool_all, Q),
        "model_swap": (choice_all, pool_all, pool_all, QSWAP),
        "topic_shift": (choice_nm, pool_nm, pool_math, Q),
    }
    for name, (choice, pre, post, Qpost) in worlds.items():
        ref = len(MODELS) - 1
        true_post = float(np.mean(np.clip(Qpost[post, ref] - Qpost[post, choice[post]], 0, 1)))
        true_pre = float(np.mean(np.clip(Q[pre, ref] - Q[pre, choice[pre]], 0, 1)))
        res = {"true_risk_before": true_pre, "true_risk_after": true_post,
               "cheaper_share_after": float(np.mean(choice[post] != ref)), "by_rate": []}
        for rate in RATES:
            rng = np.random.default_rng(int(rate * 1000))
            sims = [simulate(choice, pre, post, Qpost, rate, rng) for _ in range(RUNS)]
            alarms = [a for a, *_ in sims]
            early = sum(a is not None and a < SHIFT for a in alarms)
            late = [a - SHIFT for a in alarms if a is not None and a >= SHIFT]
            res["by_rate"].append({
                "rate": rate,
                "alarm_rate": float(np.mean([a is not None for a in alarms])),
                "false_alarm_rate_before_shift": early / RUNS,
                "detected_after_shift": len(late) / RUNS,
                "median_delay": float(np.median(late)) if late else None,
                "p90_delay": float(np.percentile(late, 90)) if late else None,
                "risk_after_without_guard": float(np.mean([s[1] for s in sims])),
                "risk_after_with_guard": float(np.mean([s[2] for s in sims])),
                "shadow_cost_share": float(np.mean([s[3] for s in sims])),
            })
        out["worlds"][name] = res
        print(f"\n{name}: true risk before {true_pre:.4f}, after {true_post:.4f} (eps {EPS}); cheaper share after {res['cheaper_share_after']:.2f}")
        for b in res["by_rate"]:
            d = "-" if b["median_delay"] is None else f"{b['median_delay']:.0f}"
            print(f"  r={b['rate']:.2f}: alarms {100 * b['alarm_rate']:.0f}% (before shift {100 * b['false_alarm_rate_before_shift']:.0f}%), "
                  f"median delay {d}, risk after: {b['risk_after_without_guard']:.4f} -> {b['risk_after_with_guard']:.4f} with guard, "
                  f"shadow cost {100 * b['shadow_cost_share']:.1f}%")
    out["certificates"] = {"all": cert_all.to_dict(), "no_maths": cert_nm.to_dict()}
    json.dump(out, open("results/drift.json", "w"), indent=1)


if __name__ == "__main__":
    main()

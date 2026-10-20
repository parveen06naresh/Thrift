"""Router benchmark on real HELM Lite outcomes (no simulation, no API key).

For each (cheap, strong) pair: repeated random splits of the 8,021 questions into
train (50%) / calibration (25%) / test (25%). The router is trained, its threshold is
certified with Learn-then-Test, and then judged on the untouched test split:
  * cheap_share: fraction of test questions sent to the cheap model
  * quality_drop: accuracy(always strong) - accuracy(routed), in points
  * cost_saved: 1 - cost(routed) / cost(always strong), using each model's *real* token counts
  * violated: whether the certified bound (clipped drop <= epsilon) failed on test
Baselines at the same cheap share: random routing, and the oracle (perfect foresight).

Writes results/router.json and thrift_llm/data/router_helm.json (the router Thrift ships).
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
from thrift_llm.router import Router, calibrate_scores  # noqa: E402

PAIRS = [
    ("openai_gpt-4o-mini-2024-07-18", "openai_gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"),
    ("meta_llama-3.1-8b-instruct-turbo", "meta_llama-3.1-405b-instruct-turbo", "llama-3.1-8b-instruct-turbo", "llama-3.1-405b-instruct-turbo"),
    ("anthropic_claude-3-haiku-20240307", "anthropic_claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"),
]
EPS = [0.01, 0.02, 0.03, 0.05]
DELTA = 0.1
SPLITS = int(os.environ.get("SPLITS", 40))

ROWS = [json.loads(line) for line in gzip.open("data/helm_routing.jsonl.gz", "rt")]
TEXT = [r["input"] for r in ROWS]


def arrays(c, s, pc, ps):
    qc = np.array([r["models"][c]["q"] for r in ROWS])
    qs = np.array([r["models"][s]["q"] for r in ROWS])
    cc = np.array([(r["models"][c]["in"] * PRICES[pc][0] + r["models"][c]["out"] * PRICES[pc][1]) / 1e6 for r in ROWS])
    cs = np.array([(r["models"][s]["in"] * PRICES[ps][0] + r["models"][s]["out"] * PRICES[ps][1]) / 1e6 for r in ROWS])
    return qc, qs, cc, cs


def one_split(args):
    pair, seed = args
    c, s, pc, ps = pair
    qc, qs, cc, cs = arrays(c, s, pc, ps)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ROWS))
    n = len(idx)
    tr, ca, te = idx[: n // 2], idx[n // 2 : 3 * n // 4], idx[3 * n // 4 :]
    router = Router().fit([TEXT[i] for i in tr], qc[tr], qs[tr], seed=seed)
    sc = router.score(TEXT)
    out = []
    gap = qs[te] - qc[te]
    for eps in EPS:
        cert = calibrate_scores(sc[ca], qc[ca], qs[ca], eps, DELTA)
        cheap = sc[te] >= cert.setting
        k = int(cheap.sum())
        q_routed = np.where(cheap, qc[te], qs[te]).mean()
        cost_routed = np.where(cheap, cc[te], cs[te]).sum()
        # random routing with the same number of cheap calls (expected value)
        share = k / len(te)
        q_rand = qs[te].mean() - share * gap.mean()
        cost_rand = cs[te].sum() - share * (cs[te] - cc[te]).sum()
        # oracle: route the k questions with the smallest gap (ties: cheapest saving first)
        o = np.argsort(gap, kind="stable")[:k]
        q_orc = qs[te].mean() - gap[o].sum() / len(te)
        out.append({
            "eps": eps, "seed": seed, "threshold": cert.setting, "cheap_share": share,
            "quality_strong": float(qs[te].mean()), "quality_cheap": float(qc[te].mean()),
            "quality_drop": float(qs[te].mean() - q_routed),
            "clipped_drop": float((cheap * np.clip(gap, 0, 1)).mean()),
            "violated": bool((cheap * np.clip(gap, 0, 1)).mean() > eps),
            "cost_saved": float(1 - cost_routed / cs[te].sum()),
            "random_quality_drop": float(qs[te].mean() - q_rand), "random_cost_saved": float(1 - cost_rand / cs[te].sum()),
            "oracle_quality_drop": float(qs[te].mean() - q_orc),
            "cal_ucb": cert.risk_ucb,
        })
    # deferral curve on seed 0 only
    curve = None
    if seed == 0:
        curve = []
        order = np.argsort(-sc[te])
        for share in np.linspace(0, 1, 41):
            k = int(round(share * len(te)))
            cheap = np.zeros(len(te), bool)
            cheap[order[:k]] = True
            o = np.argsort(gap, kind="stable")[:k]
            curve.append({
                "share": float(share),
                "router": float(np.where(cheap, qc[te], qs[te]).mean()),
                "random": float(qs[te].mean() - share * gap.mean()),
                "oracle": float(qs[te].mean() - gap[o].sum() / len(te)),
                "router_cost": float(np.where(cheap, cc[te], cs[te]).sum() / cs[te].sum()),
            })
    return pair, seed, out, curve


def main():
    jobs = [(p, seed) for p in PAIRS for seed in range(SPLITS)]
    with mp.Pool(min(8, os.cpu_count() or 2)) as pool:
        res = pool.map(one_split, jobs)
    summary = []
    for pair in PAIRS:
        c, s, pc, ps = pair
        qc, qs, cc, cs = arrays(*pair)
        rows = [r for p, _, out, _ in res if p == pair for r in out]
        curve = next(cv for p, sd, _, cv in res if p == pair and sd == 0)
        per_eps = []
        for eps in EPS:
            rs = [r for r in rows if r["eps"] == eps]

            def m(k, rs=rs):
                v = np.array([r[k] for r in rs], dtype=float)
                return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}

            per_eps.append({
                "eps": eps, "delta": DELTA, "splits": len(rs),
                "cheap_share": m("cheap_share"), "quality_drop": m("quality_drop"), "clipped_drop": m("clipped_drop"),
                "cost_saved": m("cost_saved"), "random_quality_drop": m("random_quality_drop"), "oracle_quality_drop": m("oracle_quality_drop"),
                "violation_rate": float(np.mean([r["violated"] for r in rs])),
            })
        summary.append({
            "cheap": pc, "strong": ps, "n_questions": len(ROWS),
            "accuracy_cheap": float(qc.mean()), "accuracy_strong": float(qs.mean()),
            "cost_per_1k_questions_cheap": float(cc.mean() * 1000), "cost_per_1k_questions_strong": float(cs.mean() * 1000),
            "prices": {pc: PRICES[pc], ps: PRICES[ps]},
            "results": per_eps, "curve": curve,
        })
        print(f"\n{pc} -> {ps}  acc {qc.mean():.3f} / {qs.mean():.3f}")
        for e in per_eps:
            print(f"  eps={e['eps']:.2f}: cheap {100 * e['cheap_share']['mean']:.1f}%  drop {100 * e['quality_drop']['mean']:+.2f} pts "
                  f"(random {100 * e['random_quality_drop']['mean']:+.2f}, oracle {100 * e['oracle_quality_drop']['mean']:+.2f})  "
                  f"cost -{100 * e['cost_saved']['mean']:.1f}%  bound violated {100 * e['violation_rate']:.1f}% of splits (allowed {100 * DELTA:.0f}%)")
    json.dump({"splits": SPLITS, "delta": DELTA, "pairs": summary}, open("results/router.json", "w"), indent=1)

    # Ship a router: gpt-4o-mini -> gpt-4o, trained on 75%, certified on 25% at eps = 0.02.
    c, s, pc, ps = PAIRS[0]
    qc, qs, _, _ = arrays(*PAIRS[0])
    idx = np.random.default_rng(12345).permutation(len(ROWS))
    tr, ca = idx[: 3 * len(idx) // 4], idx[3 * len(idx) // 4 :]
    r = Router("cheap", "strong").fit([TEXT[i] for i in tr], qc[tr], qs[tr])
    cert = r.calibrate([TEXT[i] for i in ca], qc[ca], qs[ca], 0.02, DELTA)
    r.meta = {"trained_on": "HELM Lite v1.13.0, 8 scenarios", "cheap": pc, "strong": ps, "n_train": len(tr), "n_cal": len(ca),
              "note": "Calibrated on HELM-like traffic. Re-run `thrift-llm calibrate` on your own labelled prompts."}
    r.save("thrift_llm/data/router_helm.json")
    print("\nbundled router:", cert.to_dict(), os.path.getsize("thrift_llm/data/router_helm.json"), "bytes")


if __name__ == "__main__":
    main()

"""Semantic-cache benchmark on Quora Question Pairs (404k human-labelled pairs).

QQP is adversarial for a cache: its non-duplicate pairs were *selected* to look alike
("How do I learn Python?" vs "How do I learn Python fast?"). A cache that is safe here is
conservative on ordinary traffic.

Workload: cache the first question of N pairs, then ask the second question of each pair.
A query may hit *any* cached entry, not only its partner. A hit is correct when the two
questions are in the same duplicate cluster (transitive closure of QQP's duplicate labels).

The threshold is certified on one workload with Learn-then-Test (wrong-answer rate <= alpha
w.p. >= 1 - delta) and evaluated on a disjoint one, repeated over independent draws.

Writes results/cache.json.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thrift_llm.cache import SemanticCache, calibrate_threshold  # noqa: E402
from thrift_llm.text import HashingEmbedder, key_terms, normalize_query  # noqa: E402

URL = "https://raw.githubusercontent.com/MLDroid/quora_duplicate_challenge/master/data/quora_duplicate_questions.tsv"
PATH = "data/qqp.tsv"
N = 10_000  # pairs per workload
DRAWS = int(os.environ.get("DRAWS", 10))
ALPHAS = [0.01, 0.02, 0.05]
DELTA = 0.1


def load():
    if not os.path.exists(PATH):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL, PATH)
    csv.field_size_limit(10**9)
    rows = list(csv.DictReader(open(PATH, encoding="utf-8"), delimiter="\t"))
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        if r["is_duplicate"] == "1":
            parent[find(r["qid1"])] = find(r["qid2"])
    return rows, find


def workload(rows, find, emb, guard=True):
    cache = [(find(r["qid1"]), r["question1"]) for r in rows]
    qs = [(find(r["qid2"]), r["question2"]) for r in rows]
    C = emb.embed([t for _, t in cache])
    Q = emb.embed([t for _, t in qs])
    ct = [key_terms(t) for _, t in cache]
    cn = {}
    for j, (_, t) in enumerate(cache):
        cn.setdefault(normalize_query(t), j)
    sims = Q @ C.T
    top = np.argsort(-sims, axis=1)[:, :5]
    best = np.full(len(qs), -np.inf)
    correct = np.zeros(len(qs), bool)
    for i, (comp, t) in enumerate(qs):
        j = cn.get(normalize_query(t))
        if j is not None:  # exact tier
            best[i], correct[i] = 1.0, cache[j][0] == comp
            continue
        kt = key_terms(t)
        for j in top[i]:
            if guard and ct[j] != kt:
                continue
            best[i], correct[i] = sims[i, j], cache[j][0] == comp
            break
    avail = np.isin([c for c, _ in qs], [c for c, _ in cache])
    return best, correct, avail


def metrics(best, correct, avail, t):
    hit = best >= t
    return {
        "threshold": float(t), "hit_rate": float(hit.mean()), "wrong_rate": float((hit & ~correct).mean()),
        "precision": float(correct[hit].mean()) if hit.any() else 1.0,
        "recall_of_duplicates": float((hit & correct).sum() / max(1, avail.sum())),
    }


def main():
    rows, find = load()
    emb = HashingEmbedder()
    rng = np.random.default_rng(0)
    out = {"dataset": "Quora Question Pairs", "pairs_per_workload": N, "draws": DRAWS, "delta": DELTA, "embedder": emb.name, "runs": []}
    for d in range(DRAWS):
        perm = rng.permutation(len(rows))
        cal_rows = [rows[i] for i in perm[:N]]
        used = {find(r["qid1"]) for r in cal_rows} | {find(r["qid2"]) for r in cal_rows}
        test_rows = []
        for i in perm[N:]:
            r = rows[i]
            if find(r["qid1"]) not in used and find(r["qid2"]) not in used:
                test_rows.append(r)
                if len(test_rows) == N:
                    break
        cb, cc, _ = workload(cal_rows, find, emb)
        tb, tc, ta = workload(test_rows, find, emb)
        tb_ng, tc_ng, _ = workload(test_rows, find, emb, guard=False)
        run = {"draw": d, "dup_available": float(ta.mean()), "certified": [], "fixed": [], "exact_only": metrics(np.where(tb >= 1.0, tb, -np.inf), tc, ta, 1.0)}
        for a in ALPHAS:
            cert = calibrate_threshold(cb, cc, a, DELTA)
            m = metrics(tb, tc, ta, cert.setting)
            m.update({"alpha": a, "violated": m["wrong_rate"] > a, "cal_ucb": cert.risk_ucb})
            run["certified"].append(m)
        for t in (0.8, 0.85, 0.9, 0.92, 0.95):
            run["fixed"].append({**metrics(tb, tc, ta, t), "no_guard": metrics(tb_ng, tc_ng, ta, t)})
        out["runs"].append(run)
        c2 = run["certified"][1]
        print(f"draw {d}: alpha=2% -> t={c2['threshold']:.3f} hit {100 * c2['hit_rate']:.1f}% wrong {100 * c2['wrong_rate']:.2f}% "
              f"precision {100 * c2['precision']:.1f}% (exact-only hit {100 * run['exact_only']['hit_rate']:.2f}%)")

    # Real code path + latency: the shipped SemanticCache at the default threshold.
    cache = SemanticCache()
    scope = {"tier": "x"}
    t0 = time.perf_counter()
    for r in test_rows:
        cache.put(scope, r["question1"], {"a": r["qid1"]})
    put_ms = (time.perf_counter() - t0) * 1000 / len(test_rows)
    hits = wrong = 0
    lat = []
    for r in test_rows:
        t1 = time.perf_counter()
        h = cache.get(scope, r["question2"])
        lat.append((time.perf_counter() - t1) * 1000)
        if h is not None:
            hits += 1
            wrong += find(h.response["a"]) != find(r["qid2"])
    out["shipped_cache"] = {
        "threshold": cache.threshold, "entries": len(cache), "hit_rate": hits / len(test_rows), "wrong_rate": wrong / len(test_rows),
        "lookup_ms_p50": float(np.percentile(lat, 50)), "lookup_ms_p99": float(np.percentile(lat, 99)), "insert_ms_mean": put_ms,
    }
    print("shipped cache:", out["shipped_cache"])
    for a_i, a in enumerate(ALPHAS):
        cs = [r["certified"][a_i] for r in out["runs"]]
        print(f"alpha={a}: hit {100 * np.mean([c['hit_rate'] for c in cs]):.1f}%  wrong {100 * np.mean([c['wrong_rate'] for c in cs]):.2f}%  "
              f"violations {sum(c['violated'] for c in cs)}/{len(cs)}")
    import json

    json.dump(out, open("results/cache.json", "w"), indent=1)


if __name__ == "__main__":
    main()

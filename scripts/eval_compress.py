"""Compression benchmark: does the answer survive when the context shrinks?

Task (multi-document QA, the RAG setting): each Natural Questions question from HELM Lite is
given its gold passage plus 9 distractor passages from other questions, shuffled, the way a
retriever returns 10 chunks. A real reader model (BERT-base fine-tuned on SQuAD, see
reader.py) answers from the full context and from compressed contexts. Quality = SQuAD F1
against the human reference answers.

Methods at the same token budget (keep_ratio of context tokens, counted with o200k_base):
  * thrift: query-aware sentence selection (thrift_llm.compress)
  * truncate: keep the first keep_ratio of the context (what overflowing a window does)
  * random: keep a random subset of sentences
Per-question loss = max(0, F1(full) - F1(compressed)). Learn-then-Test picks the most
aggressive ratio whose mean loss is certified <= epsilon (w.p. >= 1 - delta) on a calibration
half, and we report what happens on the other half, over many random halvings.

Also reports the lossless normalisation stage on every HELM Lite prompt (no reader needed).
Writes results/compress.json.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from reader import Reader, f1  # noqa: E402

from thrift_llm import risk  # noqa: E402
from thrift_llm import tokens as tok  # noqa: E402
from thrift_llm.compress import Compressor, _paragraphs, _units, normalize_text  # noqa: E402

N = int(os.environ.get("N", 240))
RATIOS = [0.6, 0.4, 0.25]
EPS = [0.02, 0.05]
DELTA = 0.1
SPLITS = 200
K_DOCS = 10


def build_tasks():
    inst = json.load(open("data/helm/instances/natural_qa:mode=openbook_longans.json"))
    items = []
    for x in inst:
        passage, q = x["input"]["text"].rsplit("Question:", 1)
        refs = [r["output"]["text"] for r in x["references"] if "correct" in r["tags"]]
        items.append((normalize_text(passage), q.strip(), refs))
    rng = random.Random(0)
    tasks = []
    for i in rng.sample(range(len(items)), N):
        p, q, refs = items[i]
        others = [items[j][0] for j in rng.sample(range(len(items)), K_DOCS + 1) if j != i][: K_DOCS - 1]
        docs = others + [p]
        rng.shuffle(docs)
        # cap each distractor so one giant table doesn't dominate the context
        docs = [d if d == p else " ".join(d.split()[:400]) for d in docs]
        tasks.append({"q": q, "refs": refs, "context": "\n\n".join(docs), "gold": p})
    return tasks


def truncate(text: str, ratio: float) -> str:
    budget = ratio * tok.count(text)
    out, used = [], 0
    for par in _paragraphs(text):
        for u in _units(par):
            n = tok.count(u)
            if used + n > budget:
                return "\n\n".join(out)
            out.append(u)
            used += n
    return "\n\n".join(out)


def random_drop(text: str, ratio: float, rng: random.Random) -> str:
    units = [u for par in _paragraphs(text) for u in _units(par)]
    sizes = [tok.count(u) for u in units]
    budget = ratio * sum(sizes)
    order = list(range(len(units)))
    rng.shuffle(order)
    keep, used = set(), 0
    for j in order:
        if used + sizes[j] <= budget:
            keep.add(j)
            used += sizes[j]
    return "\n\n".join(units[j] for j in sorted(keep))


def main():
    t0 = time.time()
    tasks = build_tasks()
    reader = Reader(threads=os.cpu_count())
    comp = Compressor()
    rows = []
    ckpt = "results/.compress_rows.jsonl"  # resumable: one finished task per line
    if os.path.exists(ckpt):
        rows = [json.loads(line) for line in open(ckpt)]
    for k, t in enumerate(tasks):
        if k < len(rows):
            continue
        ctx = t["context"]
        n_ctx = tok.count(ctx)
        row = {"tokens": n_ctx, "full": f1(reader.answer(t["q"], ctx), t["refs"]), "thrift": {}, "truncate": {}, "random": {}, "kept_tokens": {}}
        for r in RATIOS:
            c = comp.select_text(ctx, t["q"], r)
            row["kept_tokens"][str(r)] = tok.count(c)
            row["thrift"][str(r)] = f1(reader.answer(t["q"], c), t["refs"])
            row["truncate"][str(r)] = f1(reader.answer(t["q"], truncate(ctx, r)), t["refs"])
            row["random"][str(r)] = f1(reader.answer(t["q"], random_drop(ctx, r, random.Random(1000 * k + int(100 * r)))), t["refs"])
        rows.append(row)
        with open(ckpt, "a") as f:
            f.write(json.dumps(row) + "\n")
        if k % 20 == 0:
            print(f"{k}/{len(tasks)} {time.time() - t0:.0f}s", file=sys.stderr)

    full = np.array([r["full"] for r in rows])
    out = {"n": len(rows), "k_docs": K_DOCS, "reader": "bertsquad-12 (BERT-base, SQuAD v1.1)", "tokenizer": tok.backend(),
           "context_tokens_mean": float(np.mean([r["tokens"] for r in rows])), "f1_full": float(full.mean()), "by_ratio": [], "certified": []}
    for r in RATIOS:
        kept = np.array([row["kept_tokens"][str(r)] / row["tokens"] for row in rows])
        e = {"keep_ratio": r, "actual_kept": float(kept.mean())}
        for m in ("thrift", "truncate", "random"):
            v = np.array([row[m][str(r)] for row in rows])
            e[m] = {"f1": float(v.mean()), "retained": float(v.mean() / full.mean()), "mean_loss": float(np.clip(full - v, 0, 1).mean())}
        out["by_ratio"].append(e)
        print(f"keep {r:.1f}: F1 thrift {e['thrift']['f1']:.3f}  truncate {e['truncate']['f1']:.3f}  random {e['random']['f1']:.3f}  (full {full.mean():.3f})")

    loss = {r: np.clip(full - np.array([row["thrift"][str(r)] for row in rows]), 0, 1) for r in RATIOS}
    srng = np.random.default_rng(0)
    for eps in EPS:
        chosen, test_loss, viol = [], [], []
        for _ in range(SPLITS):
            idx = srng.permutation(len(rows))
            ca, te = idx[: len(idx) // 2], idx[len(idx) // 2 :]
            cert = risk.learn_then_test([1.0] + RATIOS, lambda r, ca=ca: np.zeros(len(ca)) if r == 1.0 else loss[r][ca], eps, DELTA)
            chosen.append(cert.setting)
            tl = 0.0 if cert.setting == 1.0 else float(loss[cert.setting][te].mean())
            test_loss.append(tl)
            viol.append(tl > eps)
        vals, cnt = np.unique(chosen, return_counts=True)
        out["certified"].append({"eps": eps, "delta": DELTA, "splits": SPLITS, "chosen_ratio_counts": {str(v): int(c) for v, c in zip(vals, cnt)},
                                 "median_ratio": float(np.median(chosen)), "test_loss_mean": float(np.mean(test_loss)), "violation_rate": float(np.mean(viol))})
        print(f"eps={eps}: median keep {np.median(chosen):.2f}, test loss {np.mean(test_loss):.4f}, violated {100 * np.mean(viol):.1f}% of splits")

    # Lossless stage on every HELM Lite prompt
    before = after = 0
    for f in glob.glob("data/helm/requests/*.json"):
        for req in json.load(open(f)):
            p = req["request"]["prompt"]
            before += tok.count(p)
            after += tok.count(normalize_text(p))
    out["normalize_helm"] = {"prompts_tokens_before": before, "after": after, "saved_pct": 100 * (1 - after / before)}
    print("normalize on HELM prompts:", out["normalize_helm"])
    out["reader_windows"] = reader.windows_run
    out["runtime_s"] = time.time() - t0
    json.dump(out, open("results/compress.json", "w"), indent=1)


if __name__ == "__main__":
    main()

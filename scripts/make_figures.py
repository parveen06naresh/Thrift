"""Figures for the README and paper, drawn only from results/*.json."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, ACC, BLUE, WARN, GRID = "#14181d", "#6b7682", "#0f9d63", "#2563eb", "#c2410c", "#e5e8ec"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "figure.dpi": 160,
                     "font.family": "DejaVu Sans"})


def routing():
    R = json.load(open("results/router.json"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
    for ax, p in zip(axes, R["pairs"]):
        c = p["curve"]
        x = [100 * (1 - e["router_cost"]) for e in c]
        share = [100 * e["share"] for e in c]
        ax.plot(share, [100 * e["oracle"] for e in c], color=MUTED, ls=":", lw=1.5, label="oracle (perfect foresight)")
        ax.plot(share, [100 * e["random"] for e in c], color=WARN, ls="--", lw=1.5, label="random routing")
        ax.plot(share, [100 * e["router"] for e in c], color=ACC, lw=2.2, label="Thrift router")
        for r in p["results"]:
            if r["eps"] in (0.02, 0.05):
                s = 100 * r["cheap_share"]["mean"]
                q = 100 * (p["curve"][0]["router"] - r["quality_drop"]["mean"])
                ax.scatter([s], [q], color=INK, zorder=5, s=18)
                ax.annotate(f"ε={int(100 * r['eps'])}%: −{100 * r['cost_saved']['mean']:.0f}% cost", (s, q), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color=INK)
        ax.set_title(f"{p['cheap'].split('-2024')[0]} → {p['strong'].split('-2024')[0]}", fontsize=9.5, color=INK)
        ax.set_xlabel("% of questions sent to the cheap model")
        del x
    axes[0].set_ylabel("accuracy on test split (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("docs/figures/routing.png", bbox_inches="tight")


def compression():
    C = json.load(open("results/compress.json"))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    br = C["by_ratio"]
    xs = [100 * b["keep_ratio"] for b in br]
    for m, col, lab in (("thrift", ACC, "Thrift (query-aware)"), ("truncate", WARN, "truncate"), ("random", MUTED, "random sentences")):
        ax.plot(xs, [100 * b[m]["f1"] for b in br], marker="o", ms=4, color=col, lw=2 if m == "thrift" else 1.5, label=lab)
    ax.axhline(100 * C["f1_full"], color=INK, lw=1, ls=":", label="full context")
    ax.set_xlabel("% of context tokens kept")
    ax.set_ylabel("reader F1 (%)")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/figures/compression.png", bbox_inches="tight")


def cache():
    K = json.load(open("results/cache.json"))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    runs = K["runs"]
    import numpy as np

    ts = [f["threshold"] for f in runs[0]["fixed"]]
    hit = [np.mean([r["fixed"][i]["hit_rate"] for r in runs]) * 100 for i in range(len(ts))]
    wrong = [np.mean([r["fixed"][i]["wrong_rate"] for r in runs]) * 100 for i in range(len(ts))]
    hit_ng = [np.mean([r["fixed"][i]["no_guard"]["hit_rate"] for r in runs]) * 100 for i in range(len(ts))]
    wrong_ng = [np.mean([r["fixed"][i]["no_guard"]["wrong_rate"] for r in runs]) * 100 for i in range(len(ts))]
    ax.plot(wrong_ng, hit_ng, marker="o", ms=4, color=MUTED, ls="--", label="cosine only")
    ax.plot(wrong, hit, marker="o", ms=4, color=ACC, lw=2, label="cosine + key-term guard")
    for t, w, h in zip(ts, wrong, hit):
        ax.annotate(f"{t:.2f}", (w, h), textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED)
    for i, a in enumerate([0.01, 0.02, 0.05]):
        cs = [r["certified"][i] for r in runs]
        ax.scatter([100 * np.mean([c["wrong_rate"] for c in cs])], [100 * np.mean([c["hit_rate"] for c in cs])], color=INK, zorder=5, s=22)
        ax.annotate(f"certified α={int(100 * a)}%", (100 * np.mean([c["wrong_rate"] for c in cs]), 100 * np.mean([c["hit_rate"] for c in cs])),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("% of queries given a wrong cached answer")
    ax.set_ylabel("% of queries served from cache")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/figures/cache.png", bbox_inches="tight")


if __name__ == "__main__":
    for f in (routing, cache, compression):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)

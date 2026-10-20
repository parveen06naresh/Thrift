"""thrift command line: serve, demo, calibrate, compress, init."""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from importlib import resources

EXAMPLE_CONFIG = '''# Thrift gateway config. Tiers are ordered cheapest -> strongest.
# Clients call model "thrift-auto" to let Thrift route, or a tier name to pin one.

[server]
host = "127.0.0.1"
port = 8787
# api_keys = ["sk-thrift-change-me"]   # require clients to send one of these as Bearer token

[[tier]]
name = "cheap"
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[[tier]]
name = "strong"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

# Self-hosted example (vLLM / SGLang / Ollama all speak this API):
# [[tier]]
# name = "cheap"
# model = "meta-llama/Llama-3.1-8B-Instruct"
# base_url = "http://localhost:8000/v1"
# api_key_env = "VLLM_API_KEY"

[router]
enabled = true
# path = "router.json"      # from `thrift-llm calibrate`; default is the bundled HELM-calibrated router

[cache]
enabled = true
semantic = true
threshold = 0.92            # certified <= 2% wrong-answer rate on Quora Question Pairs
ttl_s = 604800
# path = "thrift-cache.sqlite"

[compress]
enabled = true
keep_ratio = 0.5            # share of long-context tokens kept by query-aware selection
min_tokens = 500            # only messages longer than this are candidates

[ledger]
# path = "thrift-ledger.sqlite"

[shadow]
rate = 0.0                  # e.g. 0.02: also send 2% of cheap-routed requests to the strong tier, for recalibration

[prices]                    # USD per 1M tokens [input, output]; extends the built-in table
# "my-finetune" = [0.30, 1.20]
'''


def _serve(cfg, open_browser: bool = False) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(cfg)
    url = f"http://{cfg.host}:{cfg.port}"
    print(f"Thrift listening on {url}/v1   dashboard: {url}/dashboard", file=sys.stderr)
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url + "/dashboard")).start()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


def cmd_serve(a) -> None:
    from . import config

    cfg = config.demo_config() if a.demo else config.load(a.config)
    if a.port:
        cfg.port = a.port
    if a.host:
        cfg.host = a.host
    _serve(cfg)


def demo_prompts() -> list[str]:
    with resources.files("thrift_llm").joinpath("data/demo_prompts.json").open() as f:
        return json.load(f)


def cmd_demo(a) -> None:
    """Start mock tiers + the gateway, replay a realistic mix of traffic, keep serving the dashboard."""
    import httpx

    from . import config

    cfg = config.demo_config()
    cfg.port = a.port
    prompts = demo_prompts()
    rng = random.Random(0)
    # Traffic with the shape real assistants see: a long tail of unique questions plus a head of
    # repeated ones (Zipf), some with long pasted context.
    weights = [1 / (i + 1) ** 1.1 for i in range(len(prompts))]

    def traffic():
        time.sleep(1.5)
        url = f"http://{cfg.host}:{cfg.port}/v1/chat/completions"
        with httpx.Client(timeout=30) as c:
            for _ in range(a.requests):
                p = rng.choices(prompts, weights)[0]
                try:
                    c.post(url, json={"model": "thrift-auto", "messages": [{"role": "user", "content": p}]})
                except httpx.HTTPError:
                    return
                time.sleep(a.interval)
        s = httpx.get(f"http://{cfg.host}:{cfg.port}/thrift/stats").json()
        d = s["derived"]
        print(f"\n{a.requests} requests replayed: saved {d['savings_pct']:.1f}% of spend "
              f"(cache hit {100 * d['cache_hit_rate']:.0f}%, cheap model {100 * d['cheap_route_rate']:.0f}%, "
              f"prompt tokens -{d['prompt_tokens_saved_pct']:.0f}%), overhead p50 {d['overhead_ms_p50']:.2f} ms.\n"
              f"(Synthetic repeat-heavy traffic on mock models: a demo, not a benchmark. See results/ for measured numbers.) "
              f"Dashboard still live; Ctrl-C to stop.", file=sys.stderr)

    threading.Thread(target=traffic, daemon=True).start()
    _serve(cfg, open_browser=not a.no_browser)


def cmd_calibrate(a) -> None:
    """Fit + calibrate a router from JSONL rows {"prompt", "q_cheap", "q_strong"} (quality in [0, 1])."""
    import numpy as np

    from .router import Router

    rows = [json.loads(line) for line in open(a.data) if line.strip()]
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(rows))
    cut = int(len(rows) * (1 - a.cal_frac))
    tr, ca = idx[:cut], idx[cut:]
    T = [r["prompt"] for r in rows]
    qc = np.array([float(r["q_cheap"]) for r in rows])
    qs = np.array([float(r["q_strong"]) for r in rows])
    r = Router(a.cheap, a.strong).fit([T[i] for i in tr], qc[tr], qs[tr])
    cert = r.calibrate([T[i] for i in ca], qc[ca], qs[ca], a.epsilon, a.delta)
    r.meta = {"source": a.data, "n_train": len(tr), "n_cal": len(ca)}
    r.save(a.out)
    share = float(np.mean(r.score([T[i] for i in ca]) >= r.threshold))
    print(json.dumps({"out": a.out, "threshold": r.threshold, "cheap_share_on_cal": share, **cert.to_dict()}, indent=2))


def cmd_compress(a) -> None:
    from .compress import Compressor

    text = sys.stdin.read()
    res = Compressor(min_tokens=0).compress([{"role": "user", "content": text}], a.keep)
    sys.stdout.write(res.messages[0]["content"] + "\n")
    print(f"tokens {res.tokens_before} -> {res.tokens_after} ({100 * (1 - res.ratio):.1f}% saved) {res.stages}", file=sys.stderr)


def cmd_init(a) -> None:
    with open(a.out, "x") as f:
        f.write(EXAMPLE_CONFIG)
    print(f"wrote {a.out}; edit tiers, then: thrift-llm serve --config {a.out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="thrift-llm", description="Thrift: same answers, fewer tokens. An OpenAI-compatible LLM gateway.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the gateway")
    s.add_argument("--config", default="thrift.toml")
    s.add_argument("--demo", action="store_true", help="use offline mock tiers (no API key needed)")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)
    d = sub.add_parser("demo", help="mock tiers + replayed traffic + live dashboard, fully offline")
    d.add_argument("--port", type=int, default=8787)
    d.add_argument("--requests", type=int, default=400)
    d.add_argument("--interval", type=float, default=0.02)
    d.add_argument("--no-browser", action="store_true")
    d.set_defaults(fn=cmd_demo)
    c = sub.add_parser("calibrate", help="train + certify a router on your own labelled prompts")
    c.add_argument("--data", required=True, help='JSONL: {"prompt": ..., "q_cheap": 0..1, "q_strong": 0..1}')
    c.add_argument("--out", default="router.json")
    c.add_argument("--epsilon", type=float, default=0.02, help="max expected quality drop (absolute, e.g. 0.02 = 2 points)")
    c.add_argument("--delta", type=float, default=0.1, help="allowed probability that the guarantee fails")
    c.add_argument("--cal-frac", type=float, default=0.3)
    c.add_argument("--cheap", default="cheap")
    c.add_argument("--strong", default="strong")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_calibrate)
    k = sub.add_parser("compress", help="compress text from stdin")
    k.add_argument("--keep", type=float, default=0.5)
    k.set_defaults(fn=cmd_compress)
    i = sub.add_parser("init", help="write an example thrift.toml")
    i.add_argument("--out", default="thrift.toml")
    i.set_defaults(fn=cmd_init)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

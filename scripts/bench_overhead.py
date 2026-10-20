"""How much latency does Thrift add? Measured in-process on the real pipeline with zero-latency
mock upstreams, so every millisecond reported is Thrift's own work (cache lookup, compression,
routing, accounting). Also measures HTTP throughput through uvicorn.

Writes results/overhead.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
import sys
import threading
import time

import httpx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thrift_llm import config, tokens  # noqa: E402
from thrift_llm.gateway import Gateway  # noqa: E402


def cfg0():
    c = config.demo_config()
    c.tiers[0].base_url = "mock://cheap"
    c.tiers[1].base_url = "mock://strong"
    return c


async def pipeline_overhead():
    prompts = json.load(open("thrift_llm/data/demo_prompts.json"))
    short = [p for p in prompts if len(p) < 700]
    long = [p for p in prompts if len(p) > 2500]
    gw = Gateway(cfg0())
    out = {}

    async def run(name, texts, n, model="thrift-auto"):
        before = len(gw.ledger.overheads)
        for i in range(n):
            await gw.chat({"model": model, "messages": [{"role": "user", "content": texts[i % len(texts)] + ("" if name == "cache_hit" else f" #{i}")}]})
        ov = np.array(list(gw.ledger.overheads)[before:])
        out[name] = {"n": n, "p50_ms": float(np.percentile(ov, 50)), "p99_ms": float(np.percentile(ov, 99)), "mean_ms": float(ov.mean())}
        print(name, out[name])

    await run("short_prompt_miss", short, 2000)
    await run("long_context_compress", long, 300)
    gw.cache.clear()
    for p in short:
        await gw.chat({"model": "strong", "messages": [{"role": "user", "content": p}]})
    await run("cache_hit", short, 2000, model="strong")
    # cache lookup cost as the cache grows
    from thrift_llm.cache import SemanticCache

    c = SemanticCache()
    sizes = {}
    scope = {"tier": "x"}
    for n_target in (1_000, 10_000, 50_000):
        while len(c) < n_target:
            k = len(c)
            c.put(scope, f"question number {k} about topic {k % 977} and item {k * 31 % 1013}", {"a": k})
        lat = []
        for i in range(300):
            t = time.perf_counter()
            c.get(scope, f"an unseen question {i} about something else entirely")
            lat.append((time.perf_counter() - t) * 1000)
        sizes[n_target] = {"p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99))}
        print("cache size", n_target, sizes[n_target])
    out["semantic_lookup_by_size"] = sizes
    await gw.aclose()
    return out


def http_throughput(n=3000, conc=64):
    import uvicorn

    from thrift_llm.server import create_app

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(create_app(cfg0()), host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)

    async def go():
        lat = []
        async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=conc)) as cl:
            sem = asyncio.Semaphore(conc)

            async def one(i):
                async with sem:
                    t = time.perf_counter()
                    r = await cl.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                      json={"model": "thrift-auto", "messages": [{"role": "user", "content": f"What is {i} times {i + 1}?"}]})
                    assert r.status_code == 200
                    lat.append((time.perf_counter() - t) * 1000)

            t0 = time.perf_counter()
            await asyncio.gather(*(one(i) for i in range(n)))
            return n / (time.perf_counter() - t0), lat

    rps, lat = asyncio.run(go())
    server.should_exit = True
    th.join(timeout=5)
    res = {"requests": n, "concurrency": conc, "rps": rps, "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)),
           "note": "single uvicorn worker, mock upstream with zero latency, client on the same machine"}
    print("http", res)
    return res


if __name__ == "__main__":
    out = asyncio.run(pipeline_overhead())
    out["http"] = http_throughput()
    out["machine"] = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "python": platform.python_version(),
                      "tokenizer": tokens.backend()}
    json.dump(out, open("results/overhead.json", "w"), indent=1)

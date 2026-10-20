"""Drift guard: the e-process keeps its false-alarm promise, catches real drift, and the
gateway falls back to the strongest model when it fires. Also the multi-model ladder."""
import asyncio

import numpy as np

from thrift_llm import config
from thrift_llm.drift import RiskMonitor, agreement_loss, parse_judgement
from thrift_llm.gateway import Gateway
from thrift_llm.router import Ladder


def stream(monitor, risk, rate, n, rng):
    """Routed requests whose true risk is `risk`, shadow-sampled at `rate`, loss in {0, 1}."""
    for _ in range(n):
        if rng.random() < rate:
            loss = float(rng.random() < risk / 1.0)
            if monitor.observe(loss / rate, judged=True):
                return monitor.n
        elif monitor.observe(0.0):
            return monitor.n
    return None


def test_false_alarm_rate_is_at_most_alpha():
    rng = np.random.default_rng(0)
    alarms = sum(stream(RiskMonitor(0.02, alpha=0.05), 0.02, 0.2, 4000, rng) is not None for _ in range(200))
    assert alarms / 200 <= 0.08  # alpha = 0.05 plus Monte-Carlo slack; the risk sits exactly at epsilon


def test_detects_risk_three_times_epsilon():
    rng = np.random.default_rng(1)
    delays = [stream(RiskMonitor(0.02, alpha=0.05), 0.06, 0.2, 20000, rng) for _ in range(50)]
    assert all(d is not None for d in delays)
    assert np.median(delays) < 3000


def test_judgement_parsing_and_agreement():
    assert parse_judgement("0.3") == 0.3 and parse_judgement("Score: 1") == 1.0 and parse_judgement("no idea") is None
    assert agreement_loss("Paris", "paris") == 0.0 and agreement_loss("Paris", "Berlin") == 1.0


def test_gateway_enters_safe_mode_and_resets():
    cfg = config.from_dict({
        "tier": [{"name": "cheap", "model": "gpt-4o-mini", "base_url": "mock://cheap"},
                 {"name": "strong", "model": "gpt-4o", "base_url": "mock://strong"}],
        "router": {"threshold": 0.0},  # send everything cheap
        "shadow": {"rate": 1.0, "path": "/dev/null"},
        "cache": {"enabled": False},
    })
    gw = Gateway(cfg)

    async def always_wrong(send, ref, cand):
        return 1.0

    gw._judge = always_wrong

    async def run():
        tiers = []
        for i in range(60):
            r = await gw.chat({"model": "thrift-auto", "messages": [{"role": "user", "content": f"question {i}?"}]})
            tiers.append(r.headers["x-thrift-tier"])
            await asyncio.gather(*list(gw._tasks))
        return tiers, r

    tiers, last = asyncio.run(run())
    assert tiers[0] == "cheap" and tiers[-1] == "strong"
    assert last.headers.get("x-thrift-safe-mode") == "1"
    st = gw.stats()["drift"]
    assert st["safe_mode"] and st["e_value"] >= 20
    gw.reset_drift()
    assert not gw.safe_mode


def test_ladder_learns_who_is_good_at_what(tmp_path):
    rng = np.random.default_rng(0)
    topics = ["capital city geography", "integral calculus proof", "poem about the sea"]
    texts, Q = [], []
    for i in range(900):
        t = i % 3
        texts.append(f"{topics[t]} question number {rng.integers(1e6)}")
        # tiny model good only at geography, mid model good at geography + poems, big model good at all
        Q.append([float(t == 0), float(t in (0, 2)), 1.0])
    Q = np.array(Q)
    lad = Ladder(["tiny", "mid", "big"]).fit(texts[:600], Q[:600], epochs=20)
    cert = lad.calibrate(texts[600:], Q[600:], epsilon=0.05, delta=0.1)
    assert cert.setting <= 1.0
    assert lad.route("capital city geography question 5")[0] == "tiny"
    assert lad.route("poem about the sea question 7")[0] == "mid"
    assert lad.route("integral calculus proof question 9")[0] == "big"
    p = tmp_path / "ladder.json"
    lad.save(str(p))
    back = Ladder.load(str(p))
    assert back.models == ["tiny", "mid", "big"] and back.route("poem about the sea question 1")[0] == "mid"

    cfg = config.from_dict({"tier": [{"name": n, "model": m, "base_url": f"mock://{n}"} for n, m in
                                     [("tiny", "gemini-1.5-flash-002"), ("mid", "gpt-4o-mini"), ("big", "gpt-4o")]],
                            "router": {"path": str(p)}, "cache": {"enabled": False}})
    gw = Gateway(cfg)
    r = asyncio.run(gw.chat({"model": "auto", "messages": [{"role": "user", "content": "poem about the sea question 3"}]}))
    assert r.headers["x-thrift-tier"] == "mid"
    assert gw.stats()["totals"]["routed_cheap"] == 1

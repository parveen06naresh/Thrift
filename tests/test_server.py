import json

from starlette.testclient import TestClient

from thrift_llm import config
from thrift_llm.server import create_app


def client(**over):
    cfg = config.demo_config()
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.tiers[0].base_url = "mock://cheap"
    cfg.tiers[1].base_url = "mock://strong"
    return TestClient(create_app(cfg))


def ask(c, text, **kw):
    return c.post("/v1/chat/completions", json={"model": kw.pop("model", "thrift-auto"), "messages": [{"role": "user", "content": text}], **kw},
                  headers=kw.pop("headers", {}))


def test_chat_roundtrip_and_headers():
    with client() as c:
        r = ask(c, "What is the boiling point of water at sea level?")
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"].startswith("[")
        assert r.headers["x-thrift-cache"] == "miss"
        assert r.headers["x-thrift-tier"] in ("cheap", "strong")
        assert float(r.headers["x-thrift-baseline-usd"]) >= float(r.headers["x-thrift-cost-usd"])


def test_cache_hit_costs_nothing():
    with client() as c:
        ask(c, "Name the largest planet in the solar system.", model="strong")
        r = ask(c, "name the largest planet in the solar system", model="strong")
        assert r.headers["x-thrift-cache"] == "exact"
        assert r.json()["usage"]["total_tokens"] == 0
        s = c.get("/thrift/stats").json()
        assert s["totals"]["cache_exact"] == 1 and s["derived"]["savings_usd"] > 0


def test_pinned_model_and_bypass():
    with client() as c:
        r = ask(c, "Pin me to the cheap tier", model="cheap")
        assert r.headers["x-thrift-tier"] == "cheap"
        ask(c, "same question twice", model="strong")
        r = ask(c, "same question twice", model="strong", headers={"x-thrift-bypass": "1"})
        assert r.headers["x-thrift-cache"] == "off"


def test_streaming_and_streamed_cache_replay():
    with client() as c:
        for expect in ("miss", "exact"):
            with c.stream("POST", "/v1/chat/completions", json={"model": "strong", "stream": True,
                                                                  "messages": [{"role": "user", "content": "Stream a short haiku"}]}) as r:
                assert r.headers["x-thrift-cache"] == expect or expect == "exact"
                lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
            assert lines[-1] == "data: [DONE]"
            text = "".join((json.loads(ln[6:])["choices"] or [{}])[0].get("delta", {}).get("content") or "" for ln in lines[:-1])
            assert "haiku" in text


def test_compression_applies_to_long_context():
    filler = "\n\n".join(f"Section {i}: notes on irrigation schedules and pump maintenance for orchards." for i in range(120))
    doc = f"{filler}\n\nThe access code for the vault is 4471.\n\n{filler}\n\nWhat is the access code for the vault?"
    with client() as c:
        r = ask(c, doc, model="strong")
        assert int(r.headers["x-thrift-prompt-tokens-sent"]) < 0.7 * int(r.headers["x-thrift-prompt-tokens-original"])


def test_fallback_when_cheap_tier_fails():
    cfg = config.demo_config()
    cfg.tiers[0].base_url = "mock://cheap?fail=1"
    cfg.tiers[1].base_url = "mock://strong"
    with TestClient(create_app(cfg)) as c:
        r = ask(c, "Fallback please", model="cheap")
        assert r.status_code == 200 and r.headers["x-thrift-tier"] == "strong" and r.headers["x-thrift-fallback"] == "1"


def test_auth_and_validation():
    with client(api_keys=["sk-test"]) as c:
        assert ask(c, "hi").status_code == 401
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}, headers={"authorization": "Bearer sk-test"})
        assert r.status_code == 200
        assert c.post("/v1/chat/completions", json={"model": "x"}, headers={"authorization": "Bearer sk-test"}).status_code == 400


def test_models_metrics_dashboard():
    with client() as c:
        ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        assert "thrift-auto" in ids and "cheap" in ids
        ask(c, "count me")
        assert "thrift_requests_total 1" in c.get("/metrics").text
        assert "Thrift" in c.get("/dashboard").text

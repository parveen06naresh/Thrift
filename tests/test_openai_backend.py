"""The real HTTP backend, against a fake OpenAI-compatible server (httpx.MockTransport)."""
import asyncio
import json

import httpx

from thrift_llm import config
from thrift_llm.backends import OpenAIBackend
from thrift_llm.gateway import Gateway


def fake_openai(fail_first: int = 0):
    calls = {"n": 0, "bodies": [], "auth": []}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(req.content)
        calls["bodies"].append(body)
        calls["auth"].append(req.headers.get("authorization"))
        if calls["n"] <= fail_first:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        text = f"hello from {body['model']}"
        usage = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
        if body.get("stream"):
            chunks = [
                {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
                {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": body["model"], "choices": [], "usage": usage},
            ]
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "c1", "object": "chat.completion", "created": 1, "model": body["model"],
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                         "usage": usage})

    return handler, calls


def gateway(handler):
    cfg = config.from_dict({"tier": [
        {"name": "cheap", "model": "gpt-4o-mini", "base_url": "https://example.test/v1", "api_key": "sk-cheap"},
        {"name": "strong", "model": "gpt-4o", "base_url": "https://example.test/v1", "api_key": "sk-strong"},
    ]})
    gw = Gateway(cfg)
    for t in cfg.tiers:
        gw.backends[t.name] = OpenAIBackend(t, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return gw


def test_forwards_model_key_and_strips_thrift_fields():
    handler, calls = fake_openai()
    gw = gateway(handler)
    r = asyncio.run(gw.chat({"model": "strong", "thrift": {"cache": False}, "messages": [{"role": "user", "content": "hi"}]}))
    assert r.response["choices"][0]["message"]["content"] == "hello from gpt-4o"
    assert calls["bodies"][0]["model"] == "gpt-4o" and "thrift" not in calls["bodies"][0]
    assert calls["auth"][0] == "Bearer sk-strong"
    assert r.headers["x-thrift-prompt-tokens-sent"] == "11"


def test_retries_on_429():
    handler, calls = fake_openai(fail_first=1)
    gw = gateway(handler)
    r = asyncio.run(gw.chat({"model": "cheap", "messages": [{"role": "user", "content": "retry me"}]}))
    assert r.status == 200 and calls["n"] == 2


def test_streaming_through_real_backend_then_cached():
    handler, calls = fake_openai()
    gw = gateway(handler)

    async def run():
        out = []
        for _ in range(2):
            r = await gw.chat({"model": "strong", "stream": True, "messages": [{"role": "user", "content": "stream it"}]})
            text = ""
            async for ch in r.stream:
                for c in ch.get("choices") or []:
                    text += (c.get("delta") or {}).get("content") or ""
            out.append(text)
        return out

    a, b = asyncio.run(run())
    assert a == b == "hello from gpt-4o"
    assert calls["n"] == 1  # second request served from cache
    assert calls["bodies"][0]["stream_options"] == {"include_usage": True}

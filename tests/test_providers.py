"""Provider presets, Claude's native API translation, and cached-token pricing."""
import asyncio
import json

import httpx
import pytest

from thrift_llm import anthropic as ant
from thrift_llm import config
from thrift_llm.backends import AnthropicBackend, OpenAIBackend, make_backend
from thrift_llm.gateway import Gateway
from thrift_llm.pricing import PriceBook


def test_presets_fill_endpoint_and_key():
    cfg = config.from_dict({"tier": [
        {"name": "grok", "model": "grok-4", "provider": "grok"},
        {"name": "sonar", "model": "sonar-pro", "provider": "perplexity"},
        {"name": "gemini", "model": "gemini-2.5-flash", "provider": "google"},
        {"name": "claude", "model": "claude-sonnet-4-5", "provider": "claude"},
    ]})
    g, p, ge, c = cfg.tiers
    assert (g.provider, g.base_url, g.api_key_env) == ("xai", "https://api.x.ai/v1", "XAI_API_KEY")
    assert p.base_url == "https://api.perplexity.ai" and p.cache is False  # live web answers are never cached
    assert ge.base_url.startswith("https://generativelanguage.googleapis.com") and ge.cache is True
    assert c.api_style == "anthropic" and isinstance(make_backend(c), AnthropicBackend)
    assert isinstance(make_backend(g), OpenAIBackend) and not isinstance(make_backend(g), AnthropicBackend)


def test_config_errors():
    with pytest.raises(ValueError, match="unknown provider"):
        config.from_dict({"tier": [{"name": "x", "model": "m", "provider": "nope"}]})
    with pytest.raises(ValueError, match="fallback"):
        config.from_dict({"tier": [{"name": "x", "model": "m", "fallback": ["y"]}]})


def test_openai_to_anthropic_translation():
    long_sys = "You are a careful assistant. " * 400
    body = {
        "messages": [
            {"role": "system", "content": long_sys},
            {"role": "user", "content": [{"type": "text", "text": "What is in this image?"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "cat"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a cat"},
            {"role": "user", "content": "Thanks, and the colour?"},
        ],
        "temperature": 1.7, "stop": "END", "max_tokens": 50,
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
        "tool_choice": "required",
    }
    r = ant.to_anthropic(body, "claude-sonnet-4-5")
    assert r["model"] == "claude-sonnet-4-5" and r["max_tokens"] == 50 and r["temperature"] == 1.0
    assert r["stop_sequences"] == ["END"] and r["tool_choice"] == {"type": "any"}
    assert r["tools"][0]["input_schema"]["properties"]["q"]["type"] == "string"
    assert r["system"][-1]["cache_control"] == {"type": "ephemeral"}  # long system prompt is cacheable
    roles = [m["role"] for m in r["messages"]]
    assert roles == ["user", "assistant", "user"]  # tool result + next user turn merged: roles alternate
    assert r["messages"][0]["content"][1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    tu = {k: v for k, v in r["messages"][1]["content"][0].items() if k != "cache_control"}
    assert tu == {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "cat"}}
    assert r["messages"][2]["content"][0]["type"] == "tool_result"
    assert r["messages"][1]["content"][-1]["cache_control"] == {"type": "ephemeral"}  # conversation prefix too
    short = ant.to_anthropic({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]}, "m")
    assert "cache_control" not in short["system"][-1]


def test_anthropic_response_and_usage():
    resp = {"id": "msg_1", "model": "claude-x", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "Let me check."}, {"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {"q": "x"}}],
            "usage": {"input_tokens": 10, "output_tokens": 7, "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0}}
    o = ant.from_anthropic(resp)
    ch = o["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == "Let me check."
    assert json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}
    assert o["usage"]["prompt_tokens"] == 2010 and o["usage"]["prompt_tokens_details"]["cached_tokens"] == 2000


def test_cached_tokens_are_priced_at_the_discount():
    pb = PriceBook()
    full = pb.cost("claude-3-5-sonnet-20241022", 2010, 7)
    cached = pb.cost("claude-3-5-sonnet-20241022", 2010, 7, {"prompt_tokens_details": {"cached_tokens": 2000}})
    assert cached == pytest.approx((10 * 3 + 2000 * 0.3 + 7 * 15) / 1e6) and cached < full / 5
    assert pb.cost("gpt-4o", 1000, 0, {"prompt_tokens_details": {"cached_tokens": 1000}}) == pytest.approx(1000 * 1.25 / 1e6)
    assert pb.cost("deepseek-chat", 1000, 0, {"prompt_cache_hit_tokens": 1000}) == pytest.approx(1000 * 0.027 / 1e6)


EVENTS = [
    {"type": "message_start", "message": {"id": "msg_s", "model": "claude-x", "usage": {"input_tokens": 12, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Claude"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
    {"type": "message_stop"},
]


def fake_anthropic():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append((str(req.url), dict(req.headers), body))
        if body.get("stream"):
            sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in EVENTS)
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "msg_1", "model": body["model"], "stop_reason": "end_turn",
                                         "content": [{"type": "text", "text": "Hello Claude"}],
                                         "usage": {"input_tokens": 12, "output_tokens": 3}})

    return handler, seen


def test_claude_through_the_gateway():
    handler, seen = fake_anthropic()
    cfg = config.from_dict({"tier": [{"name": "claude", "model": "claude-3-5-sonnet-20241022", "provider": "anthropic", "api_key": "sk-ant"}]})
    gw = Gateway(cfg)
    gw.backends["claude"] = AnthropicBackend(cfg.tiers[0], client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def run():
        r = await gw.chat({"model": "claude", "messages": [{"role": "user", "content": "hi there"}]})
        s = await gw.chat({"model": "claude", "stream": True, "thrift": {"cache": False}, "messages": [{"role": "user", "content": "stream"}]})
        chunks = [c async for c in s.stream]
        return r, chunks

    r, chunks = asyncio.run(run())
    url, headers, body = seen[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-ant" and headers["anthropic-version"] == "2023-06-01" and "authorization" not in headers
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]
    assert r.response["choices"][0]["message"]["content"] == "Hello Claude"
    assert r.headers["x-thrift-prompt-tokens-sent"] == "12"
    text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))
    assert text == "Hello Claude"
    assert chunks[-1]["usage"]["completion_tokens"] == 3
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "stop" for c in chunks if c.get("choices"))


def test_fallback_chain_across_providers():
    def down(req):
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    def ok(req):
        b = json.loads(req.content)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 1, "model": b["model"],
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "from " + b["model"]},
                                                      "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    cfg = config.from_dict({"tier": [
        {"name": "grok", "model": "grok-3-mini", "provider": "xai", "api_key": "k", "fallback": ["gemini", "gpt"]},
        {"name": "gemini", "model": "gemini-2.5-flash", "provider": "gemini", "api_key": "k"},
        {"name": "gpt", "model": "gpt-4o", "provider": "openai", "api_key": "k"},
    ]})
    gw = Gateway(cfg)
    handlers = {"grok": down, "gemini": down, "gpt": ok}
    for t in cfg.tiers:
        gw.backends[t.name] = OpenAIBackend(t, client=httpx.AsyncClient(transport=httpx.MockTransport(handlers[t.name])))
    r = asyncio.run(gw.chat({"model": "grok", "messages": [{"role": "user", "content": "hello"}]}))
    assert r.status == 200 and r.headers["x-thrift-tier"] == "gpt" and r.headers["x-thrift-fallback"] == "1"

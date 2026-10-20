"""OpenAI chat-completions <-> Anthropic Messages API translation.

Clients keep speaking the OpenAI format; this module turns a request into a native Messages
call and the answer (or its event stream) back into OpenAI chunks. Going native rather than
through Anthropic's OpenAI-compatibility layer buys two things that matter for cost: prompt
caching (`cache_control`, cached reads are billed at 10% of the input price) and exact usage
including cached tokens.
"""
from __future__ import annotations

import json
import time

from . import tokens as tok

STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter",
        "pause_turn": "stop"}
DEFAULT_MAX_TOKENS = 4096
MIN_CACHE_TOKENS = 1024  # Anthropic ignores cache breakpoints on shorter prefixes


def _image(url: str) -> dict:
    if url.startswith("data:") and ";base64," in url:
        head, data = url.split(",", 1)
        return {"type": "image", "source": {"type": "base64", "media_type": head[5:].split(";")[0], "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _blocks(content) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    out = []
    for p in content:
        t = p.get("type")
        if t == "text" and p.get("text"):
            out.append({"type": "text", "text": p["text"]})
        elif t == "image_url":
            iu = p.get("image_url")
            out.append(_image(iu["url"] if isinstance(iu, dict) else iu))
    return out


def _tool_choice(tc):
    if tc in (None, "auto"):
        return {"type": "auto"}
    if tc == "required":
        return {"type": "any"}
    if tc == "none":
        return {"type": "none"}
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "tool", "name": tc["function"]["name"]}
    return {"type": "auto"}


def to_anthropic(body: dict, model: str, prompt_cache: bool = True) -> dict:
    """OpenAI request body -> Anthropic Messages request body."""
    system: list[dict] = []
    msgs: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if msgs and msgs[-1]["role"] == role:  # Anthropic requires alternating roles
            msgs[-1]["content"].extend(blocks)
        else:
            msgs.append({"role": role, "content": list(blocks)})

    for m in body.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            system.extend(_blocks(m.get("content")))
        elif role == "user":
            push("user", _blocks(m.get("content")))
        elif role == "assistant":
            blocks = _blocks(m.get("content"))
            for c in m.get("tool_calls") or []:
                f = c.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except ValueError:
                    args = {"_raw": f.get("arguments")}
                blocks.append({"type": "tool_use", "id": c.get("id"), "name": f.get("name"), "input": args})
            push("assistant", blocks)
        elif role == "tool":
            content = m.get("content")
            push("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
                           "content": content if isinstance(content, str) else _blocks(content)}])

    out: dict = {"model": model, "messages": msgs,
                 "max_tokens": int(body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS)}
    if system:
        out["system"] = system
    if body.get("temperature") is not None:
        out["temperature"] = min(1.0, float(body["temperature"]))  # OpenAI allows up to 2
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if body.get("tools"):
        out["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                         "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                        for t in body["tools"] if t.get("type") == "function"]
        out["tool_choice"] = _tool_choice(body.get("tool_choice"))
    if body.get("user"):
        out["metadata"] = {"user_id": str(body["user"])}
    if prompt_cache:
        _mark_cacheable(out)
    return out


def _mark_cacheable(req: dict) -> None:
    """Put cache breakpoints at the end of the system prompt and at the end of the conversation
    before the newest user turn, when those prefixes are long enough to be cached. A follow-up
    request in the same chat then re-reads that prefix at 10% of the price."""
    sys_tokens = sum(tok.count(b.get("text", "")) for b in req.get("system", []))
    if req.get("system") and sys_tokens >= MIN_CACHE_TOKENS:
        req["system"][-1]["cache_control"] = {"type": "ephemeral"}
    msgs = req["messages"]
    if len(msgs) >= 2:
        prefix = sys_tokens + sum(tok.count(b.get("text", "")) for m in msgs[:-1] for b in m["content"] if b.get("type") == "text")
        last = msgs[-2]["content"]
        if prefix >= MIN_CACHE_TOKENS and last:
            last[-1]["cache_control"] = {"type": "ephemeral"}


def usage(u: dict | None) -> dict:
    """Anthropic usage -> OpenAI usage. input_tokens excludes cached tokens, so add them back."""
    u = u or {}
    read = int(u.get("cache_read_input_tokens") or 0)
    write = int(u.get("cache_creation_input_tokens") or 0)
    p = int(u.get("input_tokens") or 0) + read + write
    c = int(u.get("output_tokens") or 0)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
            "prompt_tokens_details": {"cached_tokens": read}, "thrift_cache_write_tokens": write}


def from_anthropic(resp: dict) -> dict:
    """Anthropic Messages response -> OpenAI chat.completion."""
    text, calls = [], []
    for b in resp.get("content") or []:
        if b.get("type") == "text":
            text.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            calls.append({"id": b.get("id"), "type": "function",
                          "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {})}})
    msg: dict = {"role": "assistant", "content": "".join(text) if text or not calls else None}
    if calls:
        msg["tool_calls"] = calls
    return {"id": resp.get("id"), "object": "chat.completion", "created": int(time.time()), "model": resp.get("model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": STOP.get(resp.get("stop_reason"), "stop")}],
            "usage": usage(resp.get("usage"))}


class StreamTranslator:
    """Anthropic SSE events -> OpenAI chat.completion.chunk dicts (feed events in order)."""

    def __init__(self):
        self.base: dict = {}
        self.u: dict = {}
        self.tool_index: dict[int, int] = {}  # Anthropic block index -> OpenAI tool_calls index

    def _chunk(self, delta: dict, finish=None) -> dict:
        return {**self.base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def feed(self, ev: dict) -> list[dict]:
        t = ev.get("type")
        if t == "message_start":
            m = ev.get("message") or {}
            self.base = {"id": m.get("id"), "object": "chat.completion.chunk", "created": int(time.time()), "model": m.get("model")}
            self.u = dict(m.get("usage") or {})
            return [self._chunk({"role": "assistant", "content": ""})]
        if t == "content_block_start":
            b = ev.get("content_block") or {}
            if b.get("type") == "tool_use":
                i = self.tool_index.setdefault(ev.get("index", 0), len(self.tool_index))
                return [self._chunk({"tool_calls": [{"index": i, "id": b.get("id"), "type": "function",
                                                     "function": {"name": b.get("name"), "arguments": ""}}]})]
            if b.get("type") == "text" and b.get("text"):
                return [self._chunk({"content": b["text"]})]
            return []
        if t == "content_block_delta":
            d = ev.get("delta") or {}
            if d.get("type") == "text_delta":
                return [self._chunk({"content": d.get("text", "")})]
            if d.get("type") == "input_json_delta":
                i = self.tool_index.get(ev.get("index", 0), 0)
                return [self._chunk({"tool_calls": [{"index": i, "function": {"arguments": d.get("partial_json", "")}}]})]
            return []
        if t == "message_delta":
            self.u.update({k: v for k, v in (ev.get("usage") or {}).items() if v is not None})
            reason = (ev.get("delta") or {}).get("stop_reason")
            return [self._chunk({}, STOP.get(reason, "stop"))] if reason else []
        if t == "message_stop":
            return [{**self.base, "choices": [], "usage": usage(self.u)}]
        return []  # ping, content_block_stop

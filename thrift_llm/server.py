"""OpenAI-compatible HTTP server. Point any OpenAI SDK at http://host:8787/v1 and use model
"thrift-auto" to let Thrift route, or a tier name/model to pin it."""
from __future__ import annotations

import contextlib
import hmac
import json
from importlib import resources

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from . import __version__
from . import tokens as tok
from .config import Config
from .gateway import Gateway


def create_app(cfg: Config, gateway: Gateway | None = None) -> Starlette:
    gw = gateway or Gateway(cfg)

    def authorized(request: Request) -> bool:
        if not cfg.api_keys:
            return True
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return any(hmac.compare_digest(got, k) for k in cfg.api_keys)

    def unauthorized() -> JSONResponse:
        return JSONResponse({"error": {"message": "invalid Thrift API key", "type": "invalid_request_error"}}, status_code=401)

    async def chat(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": {"message": "body must be JSON", "type": "invalid_request_error"}}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse({"error": {"message": "'messages' is required", "type": "invalid_request_error"}}, status_code=400)
        res = await gw.chat(body, dict(request.headers))
        if res.stream is not None:

            async def sse():
                async for chunk in res.stream:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream", headers=res.headers)
        return JSONResponse(res.response, status_code=res.status, headers=res.headers)

    async def models(request: Request) -> Response:
        data = [{"id": m, "object": "model", "owned_by": "thrift"} for m in cfg.router.auto_models[:1]]
        data += [{"id": t.name, "object": "model", "owned_by": "thrift", "upstream_model": t.model} for t in cfg.tiers]
        return JSONResponse({"object": "list", "data": data})

    async def passthrough(request: Request) -> Response:
        """Everything else under /v1 (embeddings, completions, ...) goes to the strongest tier untouched."""
        if not authorized(request):
            return unauthorized()
        r = await gw.backends[cfg.strong.name].passthrough(request.method, "/" + request.path_params["path"], await request.body(), dict(request.headers))
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    async def stats(request: Request) -> Response:
        s = gw.ledger.stats()
        s["config"] = {
            "tiers": [{"name": t.name, "model": t.model, "price": gw.prices.lookup(t.model)} for t in cfg.tiers],
            "router": None if gw.router is None else {"threshold": gw.router.threshold, "certificate": gw.router.certificate, "meta": gw.router.meta},
            "cache": None if gw.cache is None else {"entries": len(gw.cache), "threshold": gw.cache.threshold, "semantic": gw.cache.semantic},
            "compress": None if gw.compressor is None else {"keep_ratio": gw.compressor.keep_ratio, "min_tokens": gw.compressor.min_tokens},
            "tokenizer": tok.backend(),
            "unpriced_models": sorted(gw.prices.unknown),
            "version": __version__,
        }
        return JSONResponse(s)

    async def metrics(request: Request) -> Response:
        return PlainTextResponse(gw.ledger.prometheus(), media_type="text/plain; version=0.0.4")

    async def dashboard(request: Request) -> Response:
        html = resources.files("thrift_llm").joinpath("dashboard.html").read_text()
        return HTMLResponse(html)

    async def clear_cache(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        if gw.cache is not None:
            gw.cache.clear()
        return JSONResponse({"ok": True})

    async def health(request: Request) -> Response:
        return JSONResponse({"ok": True, "version": __version__})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        await gw.aclose()

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/chat/completions", chat, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/{path:path}", passthrough, methods=["GET", "POST"]),
            Route("/thrift/stats", stats),
            Route("/thrift/cache/clear", clear_cache, methods=["POST"]),
            Route("/metrics", metrics),
            Route("/healthz", health),
            Route("/", dashboard),
            Route("/dashboard", dashboard),
        ],
        lifespan=lifespan,
    )
    app.state.gateway = gw
    return app

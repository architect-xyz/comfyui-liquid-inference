"""HTTP routes for the Settings panel. No route returns the key."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from functools import cache
from typing import Any

from .config import Config
from .oauth import SignIn

log = logging.getLogger(__name__)

Answer = tuple[int, dict[str, Any]]


def status(config: Config) -> Answer:
    return 200, config.status()


def store_key(config: Config, content_type: str, body: bytes) -> Answer:
    return _store(config, content_type, body, "api_key", config.set_api_key)


def store_base_url(config: Config, content_type: str, body: bytes) -> Answer:
    return _store(config, content_type, body, "base_url", config.set_base_url)


def start_sign_in(
    sign_in: SignIn, content_type: str, origin: str, expected_origin: str, peer: str
) -> Answer:
    if content_type != "application/json":
        return 415, {"error": "send application/json"}
    if origin != expected_origin:
        return 403, {"error": "Start sign-in from the ComfyUI Settings panel."}
    try:
        return 200, {"url": sign_in.start(origin, peer)}
    except ValueError as error:
        return 400, {"error": str(error)}


def _store(
    config: Config, content_type: str, body: bytes, field: str, setter: Callable[[str], None]
) -> Answer:
    # A web page on another site can send a form to ComfyUI, but it cannot
    # send JSON without a CORS preflight, which ComfyUI does not allow.
    if content_type != "application/json":
        return 415, {"error": "send application/json"}
    try:
        value = json.loads(body).get(field)
    except (ValueError, AttributeError):
        value = None
    if not isinstance(value, str):
        return 400, {"error": f"send a JSON object with a string `{field}`"}
    try:
        setter(value)
    except ValueError as error:
        return 400, {"error": str(error)}
    return 200, config.status()


def register(config: Callable[[], Config]) -> None:
    """Adds the routes to the ComfyUI server."""
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    @cache
    def account() -> SignIn:
        return SignIn(config())

    def answer(result: Answer) -> Any:
        code, payload = result
        return web.json_response(payload, status=code)

    async def get_prices(request: Any) -> Any:
        import asyncio
        from dataclasses import asdict

        from .nodes import _catalog, _video_catalog

        images, videos = await asyncio.gather(
            asyncio.to_thread(_catalog().models), asyncio.to_thread(_video_catalog().models)
        )
        return web.json_response(
            {"images": [asdict(m) for m in images], "videos": [asdict(m) for m in videos]}
        )

    async def get_status(request: Any) -> Any:
        code, payload = status(config())
        return answer((code, {**payload, **account().status()}))

    async def post_sign_in(request: Any) -> Any:
        return answer(
            start_sign_in(
                account(),
                request.content_type,
                request.headers.get("Origin", ""),
                f"{request.scheme}://{request.host}",
                request.remote or "",
            )
        )

    async def callback(request: Any) -> Any:
        import asyncio
        import ipaddress

        headers = {
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        }
        try:
            if not ipaddress.ip_address(request.remote or "").is_loopback:
                raise ValueError("Sign-in needs ComfyUI on this machine.")
            await asyncio.to_thread(
                account().finish, request.query, f"{request.scheme}://{request.host}"
            )
        except (ValueError, OSError) as error:
            return web.Response(text=str(error), status=400, headers=headers)
        return web.Response(
            text="Sign-in complete. Close this page and return to ComfyUI.", headers=headers
        )

    async def post_key(request: Any) -> Any:
        return answer(store_key(config(), request.content_type, await request.read()))

    async def post_base_url(request: Any) -> Any:
        return answer(store_base_url(config(), request.content_type, await request.read()))

    routes.get("/liquid-inference/prices")(get_prices)
    routes.get("/liquid-inference/status")(get_status)
    routes.post("/liquid-inference/sign-in")(post_sign_in)
    routes.get("/liquid-inference/callback")(callback)
    routes.post("/liquid-inference/key")(post_key)
    routes.post("/liquid-inference/base-url")(post_base_url)

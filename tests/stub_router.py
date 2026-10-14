"""A local stand-in for the router, for the unit tests and the smoke test."""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PIL import Image


def png(width: int = 8, height: int = 6, color: tuple[int, ...] = (255, 0, 0)) -> bytes:
    mode = "RGBA" if len(color) == 4 else "RGB"
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


IMAGE_CARDS: list[dict[str, Any]] = [
    {
        "id": "blackforestlabs/flux-schnell",
        "object": "model",
        "kind": "image_generation",
        "image_sizes": ["1024x1024", "512x512"],
    },
    {"id": "bytedance/seedream-4.5", "object": "model", "kind": "image_generation"},
    {
        "id": "openai/gpt-image",
        "object": "model",
        "kind": "image_generation",
        "image_sizes": ["1536x1024", "1024x1024"],
    },
    {"id": "liquid.auto", "object": "model", "match_type": "router"},
]


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class Stub:
    """What the stub answers. Tests change these fields."""

    base: str = ""
    cards: list[dict[str, Any]] = field(default_factory=lambda: list(IMAGE_CARDS))
    # "b64", "url" or "error".
    mode: str = "b64"
    status: int = 200
    error: dict[str, Any] = field(default_factory=dict)
    delay: float = 0.0
    image: bytes = field(default_factory=png)
    requests: list[Recorded] = field(default_factory=list)

    def calls(self, path: str) -> list[Recorded]:
        return [r for r in self.requests if r.path == path]


def _handler(stub: Stub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _record(self) -> Recorded:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {k.lower(): v for k, v in self.headers.items()}
            recorded = Recorded(self.command, self.path, headers, body)
            stub.requests.append(recorded)
            return recorded

        def _send(self, status: int, body: bytes, kind: str, extra: dict[str, str]) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            for name, value in extra.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._record()
            if self.path == "/v1/models":
                body = json.dumps({"object": "list", "data": stub.cards}).encode()
                self._send(200, body, "application/json", {})
            elif self.path == "/files/image.png":
                self._send(200, stub.image, "image/png", {})
            else:
                self._send(404, b"{}", "application/json", {})

        def do_POST(self) -> None:
            request = self._record()
            if stub.delay:
                time.sleep(stub.delay)
            if self.path != "/v1/images/generations":
                self._send(404, b"{}", "application/json", {})
                return
            job = {"x-liquid-job-id": "job-stub-1"}
            if stub.mode == "error":
                self._send(stub.status, json.dumps(stub.error).encode(), "application/json", job)
                return
            count = int(request.json().get("n", 1))
            if stub.mode == "url":
                item = {"url": f"{stub.base}/files/image.png", "mime_type": "image/png"}
            else:
                encoded = base64.b64encode(stub.image).decode()
                item = {"b64_json": encoded, "mime_type": "image/png"}
            answer = {
                "created": 1,
                "data": [item] * count,
                "exchange": {
                    "job_id": "job-stub-1",
                    "provider_slug": "stub-seller",
                    "cap_usd": 0.04,
                    "buyer_charge_usd": 0.04,
                    "settlement_status": "settled",
                    "images": count,
                    "addressing": {"served_model": "stub/served-model"},
                },
            }
            self._send(200, json.dumps(answer).encode(), "application/json", job)

    return Handler


@contextmanager
def serve(stub: Stub | None = None) -> Iterator[Stub]:
    stub = stub or Stub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(stub))
    server.daemon_threads = True
    stub.base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        server.shutdown()
        server.server_close()


def dead_base() -> str:
    """A base URL with no server behind it."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{reserved.getsockname()[1]}/v1"

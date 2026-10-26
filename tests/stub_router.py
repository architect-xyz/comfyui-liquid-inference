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

# The entries of `GET /v1/exchange/models?kind=video`, keyed by `model`.
VIDEO_ENTRIES: list[dict[str, Any]] = [
    {
        "model": "google/veo-3.1",
        "kind": "video_generation",
        "input_modalities": [{"type": "image", "formats": ["image/png"]}],
        "output_modalities": [
            {"type": "video", "sizes": ["1920x1080", "1280x720"], "seconds": [8, 4]}
        ],
    },
    {
        "model": "kling/kling-2.5",
        "kind": "video_generation",
        "input_modalities": [],
        "output_modalities": [{"type": "video", "seconds": [5, 10]}],
    },
    {"model": "openai/gpt-image", "kind": "image_generation"},
]

VIDEO_JOB = "job_video_1"
# Not a real MP4. The unit tests only pass the bytes along.
CLIP = b"\x00\x00\x00\x18ftypmp42stub-clip"


def video_status(status: str = "completed", **fields: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "id": VIDEO_JOB,
        "object": "video",
        "model": "google/veo-3.1",
        "status": status,
        "progress": 100 if status == "completed" else 0,
        "seconds": "4",
        "size": "1280x720",
        "expires_at": 1_900_000_000 if status == "completed" else None,
        "error": None,
    }
    if status != "in_progress":
        answer["exchange"] = {
            "job_id": VIDEO_JOB,
            "provider_slug": "stub-seller",
            "status": "settled",
            "charge": {"cap_usd": 0.8, "total_usd": 0.8 if status == "completed" else 0.0},
            "addressing": {"served_model": "google/veo-3.1-fast"},
        }
    answer.update(fields)
    return answer


def video_error(code: str, status: int = 503) -> tuple[int, bytes]:
    return status, json.dumps({"error": {"code": code, "message": code}}).encode()


# The start of the page that Cloudflare sends in place of a 502 from the origin.
CLOUDFLARE_502 = b"""<!DOCTYPE html>
<html class="no-js" lang="en-US"><head>
<title>router.inference.ai.exchange | 502: Bad gateway</title>
<meta charset="UTF-8" /><style>body{margin:0}</style>
</head><body><div id="cf-wrapper"><h1>Bad gateway</h1></div></body></html>"""


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
    # "b64", "url", "error" or "page".
    mode: str = "b64"
    status: int = 200
    error: dict[str, Any] = field(default_factory=dict)
    delay: float = 0.0
    image: bytes = field(default_factory=png)
    image_entries: list[dict[str, Any]] = field(default_factory=lambda: list(IMAGE_CARDS))
    video_entries: list[dict[str, Any]] = field(default_factory=lambda: list(VIDEO_ENTRIES))
    # Each read of the job takes the next answer. The last answer repeats.
    statuses: list[tuple[int, bytes]] = field(
        default_factory=lambda: [(200, json.dumps(video_status()).encode())]
    )
    contents: list[tuple[int, bytes]] = field(default_factory=lambda: [(200, CLIP)])
    requests: list[Recorded] = field(default_factory=list)

    def calls(self, path: str) -> list[Recorded]:
        return [r for r in self.requests if r.path == path]

    def answer_statuses(self, *answers: dict[str, Any] | tuple[int, bytes]) -> None:
        self.statuses = [
            a if isinstance(a, tuple) else (200, json.dumps(a).encode()) for a in answers
        ]


def _next(answers: list[tuple[int, bytes]]) -> tuple[int, bytes]:
    return answers.pop(0) if len(answers) > 1 else answers[0]


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
            elif self.path == "/v1/exchange/models?kind=image":
                self._send(200, json.dumps(stub.image_entries).encode(), "application/json", {})
            elif self.path == "/files/image.png":
                self._send(200, stub.image, "image/png", {})
            elif self.path == "/v1/exchange/models?kind=video":
                body = json.dumps(stub.video_entries).encode()
                self._send(200, body, "application/json", {})
            elif self.path == f"/v1/videos/{VIDEO_JOB}":
                status, body = _next(stub.statuses)
                self._send(status, body, "application/json", {})
            elif self.path == f"/v1/videos/{VIDEO_JOB}/content":
                status, body = _next(stub.contents)
                kind = "video/mp4" if status == 200 else "application/json"
                self._send(status, body, kind, {})
            else:
                self._send(404, b"{}", "application/json", {})

        def do_POST(self) -> None:
            request = self._record()
            if stub.delay:
                time.sleep(stub.delay)
            if self.path == "/v1/videos":
                self._create_video()
                return
            if self.path != "/v1/images/generations":
                self._send(404, b"{}", "application/json", {})
                return
            job = {"x-liquid-job-id": "job-stub-1"}
            if stub.mode == "error":
                self._send(stub.status, json.dumps(stub.error).encode(), "application/json", job)
                return
            if stub.mode == "page":
                # A proxy in front of the router replaced its answer.
                self._send(stub.status, CLOUDFLARE_502, "text/html; charset=UTF-8", {})
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

        def _create_video(self) -> None:
            job = {"x-liquid-job-id": VIDEO_JOB}
            if stub.mode == "error":
                self._send(stub.status, json.dumps(stub.error).encode(), "application/json", job)
                return
            body = json.dumps(video_status("in_progress")).encode()
            self._send(200, body, "application/json", job)

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

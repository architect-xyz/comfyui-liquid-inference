"""HTTP calls to the Liquid Inference router. Standard library only."""

from __future__ import annotations

import base64
import binascii
import html
import json
import re
import socket
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http.client import HTTPException, HTTPResponse
from typing import Any
from urllib.parse import urlsplit

from . import __version__

USER_AGENT = f"liquid-comfyui/{__version__}"
# An image job can wait in a queue at the seller before it starts.
GENERATION_TIMEOUT = 300.0
MAX_IMAGE_BYTES = 64 * 1024 * 1024
JOB_HEADER = "x-liquid-job-id"
# Codes that say that no seller can serve the selected model now. The model
# list of the node is from when the page loaded, so it can be old.
MODEL_LIST_CODES = frozenset({"model_not_found", "model_unavailable"})
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


class LiquidError(RuntimeError):
    """A request to Liquid Inference failed. The message is for the user."""


@dataclass(frozen=True)
class Generation:
    images: list[bytes]
    job_id: str
    exchange: dict[str, Any]


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    # urllib keeps the Authorization header on a redirect to another host.
    # The router does not redirect, so a redirect is refused before the key
    # goes anywhere else.
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl}", headers, fp)


_router = urllib.request.build_opener(_RefuseRedirect)
# A separate opener without credentials for image downloads.
_downloads = urllib.request.build_opener()


class Client:
    def __init__(
        self, base_url: str, api_key: str | None, *, timeout: float = GENERATION_TIMEOUT
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def list_models(self, timeout: float) -> list[dict[str, Any]]:
        """Returns the model cards from `GET /models`.

        The key is sent when there is one, because the list then also holds
        the presets of the key's workspace. The route never refuses a key.
        """
        request = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        with _router.open(request, timeout=timeout) as response:
            body = json.load(response)
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise LiquidError("the model list has no `data` array")
        return [card for card in data if isinstance(card, dict)]

    def generate(
        self, *, model: str, prompt: str, n: int, size: str | None, cap_usd: float
    ) -> Generation:
        """Buys images with one request. It never sends the request again.

        A replay with the same Idempotency-Key returns a reference to the job
        and not the images, so a retry cannot recover them. A retry with a new
        key can buy the images a second time.
        """
        if not self.api_key:
            raise LiquidError(
                "No Liquid Inference API key. Set it in Settings > Liquid Inference, "
                "or set the LIQUID_API_KEY environment variable before you start ComfyUI."
            )
        idempotency_key = str(uuid.uuid4())
        # No `response_format`: a seller whose upstream does not take that
        # field refuses a request that sets it. The node reads `b64_json` and
        # `url`, so it accepts the form that the seller returns.
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "n": n}
        if size is not None:
            payload["size"] = size
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["x-liquid-cap-usd"] = format_usd(cap_usd)
        headers["Idempotency-Key"] = idempotency_key
        request = urllib.request.Request(
            f"{self.base_url}/images/generations",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with _router.open(request, timeout=self.timeout) as response:
                status, job_header, raw = _read(response)
        except urllib.error.HTTPError as error:
            body = error.read()
            raise LiquidError(
                refusal_message(error.code, body, error.headers.get(JOB_HEADER), idempotency_key)
            ) from None
        except urllib.error.URLError as error:
            if isinstance(error.reason, (ConnectionRefusedError, socket.gaierror)):
                raise LiquidError(
                    f"Could not connect to {self.base_url}: {error.reason}. "
                    "The request was not sent, so there is no charge."
                ) from None
            raise _unknown(idempotency_key, error.reason) from None
        except (TimeoutError, OSError, HTTPException) as error:
            raise _unknown(idempotency_key, error) from None

        try:
            body = json.loads(raw)
        except ValueError:
            raise LiquidError(
                f"The router answered HTTP {status} with a body that is not JSON. "
                f"Job ID: {job_header or 'unknown'}. Idempotency-Key: {idempotency_key}."
            ) from None
        exchange = body.get("exchange") if isinstance(body, dict) else None
        exchange = exchange if isinstance(exchange, dict) else {}
        job_id = str(exchange.get("job_id") or job_header or "")
        items = body.get("data") if isinstance(body, dict) else None
        if not isinstance(items, list) or not items:
            raise LiquidError(
                f"The router returned no images for job {job_id or 'unknown'}. "
                "Look at the job's receipt before you run the node again."
            )
        images = [_image_bytes(item, job_id) for item in items]
        return Generation(images=images, job_id=job_id, exchange=exchange)

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _read(response: HTTPResponse) -> tuple[int, str | None, bytes]:
    return response.status, response.headers.get(JOB_HEADER), response.read()


def _unknown(idempotency_key: str, reason: object) -> LiquidError:
    return LiquidError(
        f"The request to Liquid Inference stopped before an answer came back ({reason}). "
        "The outcome is unknown: the job can have run, and you can have been charged. "
        "The node does not send the request again. "
        f"To find the job, give support this Idempotency-Key: {idempotency_key}."
    )


def _image_bytes(item: object, job_id: str) -> bytes:
    if not isinstance(item, dict):
        raise LiquidError(f"Job {job_id}: an image in the answer is not an object.")
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        return _decode_base64(encoded, job_id)
    url = item.get("url")
    if isinstance(url, str) and url:
        return download(url, job_id)
    raise LiquidError(f"Job {job_id}: an image in the answer has no `b64_json` and no `url`.")


def _decode_base64(encoded: str, job_id: str) -> bytes:
    try:
        return base64.b64decode(encoded)
    except (binascii.Error, ValueError):
        raise LiquidError(f"Job {job_id}: an image is not valid base64.") from None


def download(url: str, job_id: str, *, timeout: float = GENERATION_TIMEOUT) -> bytes:
    """Gets an image from the seller's link.

    The link points to the seller or to its storage, not to the router. Thus
    the request has no Authorization header.
    """
    if url.startswith("data:"):
        _, _, encoded = url.partition(",")
        return _decode_base64(encoded, job_id)
    host = urlsplit(url).hostname or url
    if urlsplit(url).scheme not in ("http", "https"):
        raise LiquidError(f"Job {job_id}: the image link {url!r} is not an http or https URL.")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _downloads.open(request, timeout=timeout) as response:
            data: bytes = response.read(MAX_IMAGE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as error:
        raise LiquidError(
            f"Job {job_id} was charged, but the image download from {host} failed: {error}. "
            "Look at the job's receipt for the image link."
        ) from None
    if len(data) > MAX_IMAGE_BYTES:
        raise LiquidError(f"Job {job_id}: the image from {host} is larger than 64 MiB.")
    return data


def format_usd(amount: float) -> str:
    """Writes a USD amount without exponent notation, to the nano-dollar."""
    return f"{amount:.9f}".rstrip("0").rstrip(".")


def refusal_message(
    status: int, body: bytes, job_id: str | None, idempotency_key: str | None = None
) -> str:
    """Makes one readable line from an error answer of the router."""
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        error = None
    if not isinstance(error, dict):
        # A proxy in front of the router, such as Cloudflare, can replace the
        # router's answer with its own page. The page does not tell whether
        # the job ran.
        message = (
            f"Liquid Inference answered HTTP {status} with {_describe(body)}, not with an "
            "error from the exchange. A proxy between ComfyUI and the exchange can send such "
            "a page. The job can have run. Look at your receipts before you run the node again"
        )
        if idempotency_key:
            message += f". Idempotency-Key: {idempotency_key}"
    else:
        code = error.get("code") or "error"
        message = f"Liquid Inference refused the request (HTTP {status}, {code}): "
        message += str(error.get("message") or "no message")
        if error.get("param"):
            message += f" [parameter: {error['param']}]"
        reasons = _exclusions(error.get("eligibility"))
        if reasons:
            message += f". Offers excluded: {reasons}"
        if code in MODEL_LIST_CODES:
            message += ". Reload the page to get the current model list"
    if job_id:
        message += f". Job ID: {job_id}"
    return message


def _describe(body: bytes) -> str:
    """Names a body that is not JSON in a few words, not as a page of HTML."""
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return "no body"
    title = _TITLE.search(text)
    if title:
        name = " ".join(html.unescape(title.group(1)).split())[:120]
        return f'a web page titled "{name}"' if name else "a web page"
    if text.startswith("<"):
        return "a web page"
    return f'"{" ".join(text.split())[:200]}"'


def _exclusions(eligibility: object, limit: int = 8) -> str:
    rows = eligibility.get("excluded") if isinstance(eligibility, dict) else None
    if not isinstance(rows, list):
        return ""
    parts = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        part = f"{row.get('provider', '?')}: {row.get('reason', '?')}"
        if row.get("detail"):
            part += f" ({row['detail']})"
        if isinstance(row.get("count"), int) and row["count"] > 1:
            part += f" x{row['count']}"
        parts.append(part)
    if len(rows) > limit:
        parts.append(f"{len(rows) - limit} more")
    return "; ".join(parts)


def receipt(generation: Generation, model: str, cap_usd: float) -> str:
    """A short JSON summary of the job, for a text output."""
    exchange = generation.exchange
    addressing = exchange.get("addressing")
    served = addressing.get("served_model") if isinstance(addressing, dict) else None
    return json.dumps(
        {
            "job_id": generation.job_id,
            "model": served or model,
            "provider": exchange.get("provider_slug") or exchange.get("provider_id"),
            "charge_usd": exchange.get("buyer_charge_usd"),
            "cap_usd": float(format_usd(cap_usd)),
            "images": len(generation.images),
            "settlement_status": exchange.get("settlement_status"),
            "node_version": __version__,
        },
        separators=(",", ":"),
    )

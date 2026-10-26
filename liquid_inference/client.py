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
from urllib.parse import quote, urlsplit

from . import __version__

USER_AGENT = f"liquid-comfyui/{__version__}"
# An image job can wait in a queue at the seller before it starts.
GENERATION_TIMEOUT = 300.0
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_VIDEO_BYTES = 512 * 1024 * 1024
# One status read or one download, not the whole wait for a video job.
READ_TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 300.0
JOB_HEADER = "x-liquid-job-id"
# Codes that say that no seller can serve the selected model now. The model
# list of the node is from when the page loaded, so it can be old.
MODEL_LIST_CODES = frozenset({"model_not_found", "model_unavailable"})
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


class LiquidError(RuntimeError):
    """A request to Liquid Inference failed. The message is for the user."""


class RetryableError(LiquidError):
    """A read failed in a way that the next try can fix.

    Only a read is tried again. A purchase is never sent twice.
    """


# The codes of a read that can succeed later: the file is not at the router
# yet, or the seller or the store did not answer.
RETRYABLE_CODES = frozenset(
    {"video_not_ready", "provider_unavailable", "provider_failed", "storage_unavailable"}
)


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
        self,
        base_url: str,
        api_key: str | None,
        *,
        timeout: float = GENERATION_TIMEOUT,
        key_expired: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.key_expired = key_expired

    def list_image_models(self, timeout: float) -> list[dict[str, Any]]:
        return self._directory("image", timeout)

    def list_video_models(self, timeout: float) -> list[dict[str, Any]]:
        return self._directory("video", timeout)

    def _directory(self, kind: str, timeout: float) -> list[dict[str, Any]]:
        """Returns the entries of `GET /exchange/models?kind=<kind>`.

        The route needs no key, and it refuses a key that it does not know.
        Thus the request has no key, and a bad key cannot hide the list.
        """
        request = urllib.request.Request(
            f"{self.base_url}/exchange/models?kind={kind}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        with _router.open(request, timeout=timeout) as response:
            body = json.load(response)
        if not isinstance(body, list):
            raise LiquidError(f"the {kind} model list is not a JSON array")
        return [entry for entry in body if isinstance(entry, dict)]

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
        # No `response_format`: a seller whose upstream does not take that
        # field refuses a request that sets it. The node reads `b64_json` and
        # `url`, so it accepts the form that the seller returns.
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "n": n}
        if size is not None:
            payload["size"] = size
        body, job_header = self._buy("/images/generations", payload, cap_usd)
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

    def create_video(
        self,
        *,
        model: str,
        prompt: str,
        size: str | None,
        seconds: int | None,
        start_image: str | None,
        cap_usd: float,
        retention_policy: str = "none",
    ) -> str:
        """Buys one video job and returns its ID. It never sends the request again.

        `start_image` is a `data:` URL. The job runs at the seller after this
        returns, and the exchange charges for it when it ends.
        """
        if retention_policy not in {"none", "allowed"}:
            raise LiquidError("retention_policy must be none or allowed.")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "exchange": {"retention_policy": retention_policy},
        }
        if size is not None:
            payload["size"] = size
        if seconds is not None:
            payload["seconds"] = str(seconds)
        if start_image is not None:
            payload["input_reference"] = {"image_url": start_image}
        body, job_header = self._buy("/videos", payload, cap_usd)
        job_id = body.get("id") if isinstance(body, dict) else None
        job_id = str(job_id or job_header or "")
        if not job_id:
            raise LiquidError(
                "The router accepted the video job but returned no job ID. "
                "Look at your receipts before you run the node again."
            )
        return job_id

    def video_status(self, job_id: str) -> dict[str, Any]:
        """Reads `GET /videos/{id}`. A failure that can pass raises `RetryableError`."""
        raw = self._get(f"/videos/{quote(job_id, safe='')}", job_id, 1024 * 1024, READ_TIMEOUT)
        try:
            body = json.loads(raw)
        except ValueError:
            raise RetryableError(f"the status of job {job_id} is not JSON") from None
        if not isinstance(body, dict):
            raise RetryableError(f"the status of job {job_id} is not a JSON object")
        return body

    def video_content(self, job_id: str) -> bytes:
        """Gets the file of a finished job through the router, with the key."""
        path = f"/videos/{quote(job_id, safe='')}/content"
        data = self._get(path, job_id, MAX_VIDEO_BYTES, DOWNLOAD_TIMEOUT)
        if len(data) > MAX_VIDEO_BYTES:
            raise LiquidError(f"Job {job_id}: the video is larger than 512 MiB.")
        return data

    def _require_key(self) -> None:
        if self.key_expired:
            raise LiquidError(
                "Your Liquid sign-in expired. Select Sign in again in Settings > Liquid Inference."
            )
        if not self.api_key:
            raise LiquidError(
                "No Liquid Inference API key. Select Sign in in Settings > Liquid Inference. "
                "For a remote server, create a key at https://inference.ai.exchange/keys. "
                "Paste it in Settings > Liquid Inference. "
                "You can also set LIQUID_API_KEY before you start ComfyUI."
            )

    def _buy(self, path: str, payload: dict[str, Any], cap_usd: float) -> tuple[Any, str | None]:
        """Sends one paid request and returns its JSON body and its job ID header."""
        self._require_key()
        idempotency_key = str(uuid.uuid4())
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["x-liquid-cap-usd"] = format_usd(cap_usd)
        headers["Idempotency-Key"] = idempotency_key
        request = urllib.request.Request(
            f"{self.base_url}{path}",
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
            return json.loads(raw), job_header
        except ValueError:
            raise LiquidError(
                f"The router answered HTTP {status} with a body that is not JSON. "
                f"Job ID: {job_header or 'unknown'}. Idempotency-Key: {idempotency_key}."
            ) from None

    def _get(self, path: str, job_id: str, limit: int, timeout: float) -> bytes:
        """A read with the key. It reads at most `limit` + 1 bytes."""
        self._require_key()
        request = urllib.request.Request(f"{self.base_url}{path}", headers=self._headers())
        try:
            with _router.open(request, timeout=timeout) as response:
                data: bytes = response.read(limit + 1)
                return data
        except urllib.error.HTTPError as error:
            body = error.read()
            message = refusal_message(error.code, body, job_id)
            if error.code >= 500 or error.code == 429 or _code(body) in RETRYABLE_CODES:
                raise RetryableError(message) from None
            raise LiquidError(message) from None
        except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            raise RetryableError(f"the request to {self.base_url} failed: {reason}") from None

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
        if status == 401:
            message += (
                ". Select Sign in again in Settings > Liquid Inference. "
                "Or replace the saved API key at https://inference.ai.exchange/keys"
            )
        if code == "insufficient_balance":
            message += (
                ". Add funds at https://inference.ai.exchange/credits, then run the node again"
            )
        if code in MODEL_LIST_CODES:
            message += ". Reload the page to get the current model list"
    if job_id:
        message += f". Job ID: {job_id}"
    return message


def _code(body: bytes) -> str | None:
    try:
        code = json.loads(body)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None
    return code if isinstance(code, str) else None


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


def video_receipt(status: dict[str, Any]) -> str:
    """A short JSON summary of a finished video job, for a text output."""
    exchange = status.get("exchange")
    exchange = exchange if isinstance(exchange, dict) else {}
    addressing = exchange.get("addressing")
    served = addressing.get("served_model") if isinstance(addressing, dict) else None
    charge = exchange.get("charge")
    charge = charge if isinstance(charge, dict) else {}
    return json.dumps(
        {
            "job_id": status.get("id"),
            "model": served or status.get("model"),
            "provider": exchange.get("provider_slug"),
            "charge_usd": charge.get("total_usd"),
            "cap_usd": charge.get("cap_usd"),
            "seconds": status.get("seconds"),
            "size": status.get("size"),
            "settlement_status": exchange.get("status"),
            "expires_at": status.get("expires_at"),
            "node_version": __version__,
        },
        separators=(",", ":"),
    )


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

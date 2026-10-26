"""The wait for a video job, and the start image. No torch and no ComfyUI.

A video job is bought with one request, and the seller makes the file after
that request returns. Thus each step after the purchase is a read, and a
read can be tried again. The node must not lose a file that the account
paid for because one read failed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from .client import Client, LiquidError, RetryableError

POLL_INTERVAL = 5.0
# The longest that the node waits for a job to end. Most jobs end in a few
# minutes, and a seller queue can add more.
WAIT_LIMIT = 30 * 60.0
MAX_FAILED_READS = 5
DOWNLOAD_TRIES = 3
# The router takes a body of up to 20 MiB, and base64 adds a third.
MAX_START_IMAGE_BYTES = 14 * 1024 * 1024

Clock = Callable[[], float]


def still_charged(job_id: str) -> str:
    return (
        f"The job keeps running at the seller, and the exchange charges for it when it ends. "
        f"To get the video later with no new charge, put job ID {job_id} in a "
        "Liquid Video Job node."
    )


async def wait(
    client: Client,
    job_id: str,
    *,
    interrupted: Callable[[], bool],
    report: Callable[[str], None],
    interval: float = POLL_INTERVAL,
    limit: float = WAIT_LIMIT,
    clock: Clock = time.monotonic,
) -> dict[str, Any]:
    """Reads the job's status until it ends. Returns the status of a completed job."""
    start = clock()
    failures = 0
    while True:
        try:
            status = await asyncio.to_thread(client.video_status, job_id)
        except RetryableError as error:
            failures += 1
            if failures > MAX_FAILED_READS:
                raise LiquidError(
                    f"The node could not read the status of job {job_id}: {error}. "
                    + still_charged(job_id)
                ) from None
            report(f"Job {job_id}: status read failed, trying again ({failures})")
        else:
            failures = 0
            state = status.get("status")
            if state == "completed":
                return status
            if state == "failed":
                raise LiquidError(_failure(job_id, status))
            report(f"Job {job_id}: {state}, {int(clock() - start)} s")
        if clock() - start >= limit:
            raise LiquidError(
                f"Job {job_id} did not end in {int(limit // 60)} minutes. " + still_charged(job_id)
            )
        await _sleep(interval, interrupted, job_id, clock)


async def download(
    client: Client,
    job_id: str,
    *,
    interrupted: Callable[[], bool],
    pause: float = POLL_INTERVAL,
    tries: int = DOWNLOAD_TRIES,
    clock: Clock = time.monotonic,
) -> bytes:
    """Gets the file of a completed job. A failed try is tried again after `pause`."""
    attempt = 1
    while True:
        try:
            return await asyncio.to_thread(client.video_content, job_id)
        except RetryableError as error:
            if attempt >= tries:
                raise LiquidError(
                    f"Job {job_id} is complete and charged, but the download failed: {error}. "
                    f"To try again with no new charge, put job ID {job_id} in a "
                    "Liquid Video Job node."
                ) from None
        attempt += 1
        await _sleep(pause, interrupted, job_id, clock)


async def _sleep(
    seconds: float, interrupted: Callable[[], bool], job_id: str, clock: Clock
) -> None:
    # Short steps, so that a stop in ComfyUI takes effect in about a second.
    end = clock() + seconds
    while True:
        if interrupted():
            raise LiquidError(
                f"The node stopped waiting for job {job_id}. " + still_charged(job_id)
            )
        left = end - clock()
        if left <= 0:
            return
        await asyncio.sleep(min(1.0, left))


def _failure(job_id: str, status: dict[str, Any]) -> str:
    error = status.get("error")
    error = error if isinstance(error, dict) else {}
    code = error.get("code") or "failed"
    message = error.get("message") or "no message"
    return (
        f"Job {job_id} failed ({code}): {message}. A failed job costs nothing. "
        "Run the node again, or select a different model."
    )


def start_image_url(batch: npt.NDArray[np.floating[Any]]) -> str:
    """Encodes a ComfyUI IMAGE batch of one image as a PNG `data:` URL."""
    if batch.ndim != 4 or batch.shape[-1] not in (3, 4):
        raise LiquidError(f"start_image has the shape {tuple(batch.shape)}, not one image.")
    if batch.shape[0] != 1:
        raise LiquidError(
            f"start_image holds {batch.shape[0]} images. Connect one image, "
            "for example with an Image From Batch node."
        )
    pixels = (np.clip(batch[0], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) > MAX_START_IMAGE_BYTES:
        raise LiquidError(
            f"start_image is {len(data) // (1024 * 1024)} MiB as a PNG, and the limit is 14 MiB. "
            "Make it smaller, for example with an Upscale Image node."
        )
    return "data:image/png;base64," + base64.b64encode(data).decode()

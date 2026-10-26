"""The ComfyUI nodes. This is the only module that uses torch or ComfyUI."""

from __future__ import annotations

import asyncio
import functools
import io
import logging
import re
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np

from . import videos
from .client import Client, LiquidError, format_usd, receipt, video_receipt
from .config import comfy_config
from .images import decode_batch
from .models import IMAGES, VIDEOS, Catalog, ImageModel, VideoModel, seconds_choices, size_choices

log = logging.getLogger(__name__)

NO_MODELS = "(no image models: check the connection, then reload)"
NO_VIDEO_MODELS = "(no video models: check the connection, then reload)"
SIZE = re.compile(r"\d+x\d+")
# The exchange takes 1 to 20 seconds.
SECONDS = re.compile(r"[1-9]|1\d|20")
JOB_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def _client() -> Client:
    config = comfy_config()
    key = config.api_key()
    return Client(config.base_url().value, key.value if key else None)


@cache
def _catalog() -> Catalog[ImageModel]:
    return Catalog(_client, comfy_config().directory / "image_models.json", IMAGES)


@cache
def _video_catalog() -> Catalog[VideoModel]:
    return Catalog(_client, comfy_config().directory / "video_models.json", VIDEOS)


def _interrupted() -> bool:
    import comfy.model_management

    return bool(comfy.model_management.processing_interrupted())


def _reporter(node_id: str | None) -> Callable[[str], None]:
    """Shows a line of text on the node while it waits."""

    def report(text: str) -> None:
        log.info("liquid inference: %s", text)
        if node_id is None:
            return
        try:
            from server import PromptServer

            PromptServer.instance.send_progress_text(text, node_id)
        except Exception as error:  # an old ComfyUI has no progress text
            log.debug("liquid inference: no progress text: %s", error)

    return report


def _video(data: bytes) -> Any:
    from comfy_api.input_impl import VideoFromFile

    # Nothing is decoded until a later node reads the frames.
    return VideoFromFile(io.BytesIO(data))


def _array(image: Any) -> Any:
    return image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)


async def _finish(client: Client, job_id: str, node_id: str | None) -> tuple[Any, str]:
    status = await videos.wait(client, job_id, interrupted=_interrupted, report=_reporter(node_id))
    data = await videos.download(client, job_id, interrupted=_interrupted)
    return _video(data), video_receipt(status)


class LiquidImage:
    CATEGORY = "Liquid Inference"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "job_id", "receipt")
    OUTPUT_TOOLTIPS = (
        "The images, in the order that the seller returned them.",
        "The exchange's ID for this job. Give it to support with each question about a charge.",
        "JSON with the model, the seller, the charge and the cap for this job.",
    )
    DESCRIPTION = (
        "External paid API. Each run sends the prompt to Liquid Inference, an auction "
        "exchange. The seller that wins the auction makes the images, and your account pays "
        "for them, up to the cap. Set the API key in Settings > Liquid Inference."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _catalog().models()
        names = [model.id for model in models] or [NO_MODELS]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (
                    names,
                    {
                        "tooltip": "An image model that a seller quotes now. The list comes from "
                        "the exchange and refreshes each minute."
                    },
                ),
                "size": (
                    size_choices(models),
                    {
                        "default": "auto",
                        "tooltip": "auto lets each seller use its default "
                        "size. An explicit size matches only the sellers that declared it.",
                    },
                ),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Images to buy."}),
                "cap_usd": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 100.0,
                        "step": 0.001,
                        "round": 0.001,
                        "tooltip": "The most, in USD, that this node can spend each time it "
                        "runs. The cap applies to each request, not to the whole workflow.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model: str, size: str) -> bool | str:
        # The router decides which models and sizes it accepts. The local list
        # can be older than the market, so it does not refuse a value.
        if not model or model == NO_MODELS:
            return "no image model is selected; check the connection to Liquid Inference"
        if size != "auto" and not SIZE.fullmatch(size):
            return f"size {size!r} is not auto or WIDTHxHEIGHT"
        return True

    def generate(
        self,
        prompt: str,
        model: str,
        size: str,
        n: int,
        cap_usd: float,
        unique_id: str | None = None,
    ) -> tuple[Any, str, str]:
        import torch

        if not prompt.strip():
            raise LiquidError("The prompt is empty.")
        generation = _client().generate(
            model=model,
            prompt=prompt,
            n=n,
            size=None if size == "auto" else size,
            cap_usd=cap_usd,
        )
        try:
            batch = decode_batch(generation.images)
        except ValueError as error:
            raise LiquidError(f"Job {generation.job_id} was charged, but {error}.") from None
        exchange = generation.exchange
        seller = exchange.get("provider_slug") or exchange.get("provider_id") or "unknown seller"
        charge = exchange.get("buyer_charge_usd")
        charged = f"${charge}" if charge is not None else "unknown charge"
        _reporter(unique_id)(
            f"{seller} · charged {charged} · cap ${format_usd(cap_usd)} · job {generation.job_id}"
        )
        return (
            torch.from_numpy(batch),
            generation.job_id,
            receipt(generation, model, cap_usd),
        )


class LiquidVideo:
    CATEGORY = "Liquid Inference"
    FUNCTION = "generate"
    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "job_id", "receipt")
    OUTPUT_TOOLTIPS = (
        "The video file. Connect it to Save Video, or to Get Video Components for the frames.",
        "The exchange's ID for this job. A Liquid Video Job node can get the video again "
        "with this ID, with no new charge. Give it to support with each question about a charge.",
        "JSON with the model, the seller, the charge, the cap and the file expiry for this job.",
    )
    DESCRIPTION = (
        "External paid API. Each run buys one video from Liquid Inference, an auction exchange. "
        "The seller that wins the auction makes the video, and your account pays for it, up to "
        "the cap. A stop in ComfyUI does not stop the job or its charge. "
        "Set the API key in Settings > Liquid Inference."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        models = _video_catalog().models()
        names = [model.id for model in models] or [NO_VIDEO_MODELS]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (
                    names,
                    {
                        "tooltip": "A video model that a seller quotes now. The list comes from "
                        "the exchange and refreshes each minute."
                    },
                ),
                "size": (
                    size_choices(models),
                    {
                        "default": "auto",
                        "tooltip": "auto lets each seller use its default size. An explicit size "
                        "matches only the sellers that declared it.",
                    },
                ),
                "seconds": (
                    seconds_choices(models),
                    {
                        "default": "auto",
                        "tooltip": "The length of the video. auto uses the shortest length that "
                        "a seller declares. The charge is for each second.",
                    },
                ),
                "cap_usd": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": "The most, in USD, that this node can spend each time it "
                        "runs. Video is priced for each second, so the charge is the price of "
                        "the length that you ask for.",
                    },
                ),
            },
            "optional": {
                "start_image": (
                    "IMAGE",
                    {
                        "tooltip": "The first frame of the video. Only the models that read an "
                        "image input can take it. Use an image with the shape of the size."
                    },
                ),
                "retention_policy": (
                    ["none", "allowed"],
                    {
                        "default": "none",
                        "tooltip": "none excludes providers that can keep your content. "
                        "allowed permits those providers. Select allowed only if you accept "
                        "that they can keep your prompt, start image and video.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, model: str, size: str, seconds: str) -> bool | str:
        # The router decides which models, sizes and lengths it accepts. The
        # local list can be older than the market, so it does not refuse a value.
        if not model or model == NO_VIDEO_MODELS:
            return "no video model is selected; check the connection to Liquid Inference"
        if size != "auto" and not SIZE.fullmatch(size):
            return f"size {size!r} is not auto or WIDTHxHEIGHT"
        if seconds != "auto" and not SECONDS.fullmatch(seconds):
            return f"seconds {seconds!r} is not auto or a whole number from 1 to 20"
        return True

    async def generate(
        self,
        prompt: str,
        model: str,
        size: str,
        seconds: str,
        cap_usd: float,
        start_image: Any = None,
        unique_id: str | None = None,
        retention_policy: str = "none",
    ) -> tuple[Any, str, str]:
        if not prompt.strip():
            raise LiquidError("The prompt is empty.")
        image = None if start_image is None else videos.start_image_url(_array(start_image))
        client = _client()
        job_id = await asyncio.to_thread(
            functools.partial(
                client.create_video,
                model=model,
                prompt=prompt,
                size=None if size == "auto" else size,
                seconds=None if seconds == "auto" else int(seconds),
                start_image=image,
                cap_usd=cap_usd,
                retention_policy=retention_policy,
            )
        )
        video, summary = await _finish(client, job_id, unique_id)
        return video, job_id, summary


class LiquidVideoJob:
    CATEGORY = "Liquid Inference"
    FUNCTION = "load"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "receipt")
    OUTPUT_TOOLTIPS = (
        "The video file of the job.",
        "JSON with the model, the seller, the charge, the cap and the file expiry for this job.",
    )
    DESCRIPTION = (
        "Gets the video of a job that a Liquid Video node bought. This node buys nothing. "
        "Use it when a run stopped after the purchase. The seller keeps the file until the "
        "expiry in the receipt."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "job_id": (
                    "STRING",
                    {"default": "", "tooltip": "The job ID from a Liquid Video node or an error."},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, job_id: str) -> bool | str:
        if not JOB_ID.fullmatch(job_id.strip()):
            return "job_id is not a job ID, such as job_0193..."
        return True

    async def load(self, job_id: str, unique_id: str | None = None) -> tuple[Any, str]:
        return await _finish(_client(), job_id.strip(), unique_id)


NODE_CLASS_MAPPINGS = {
    "LiquidImage": LiquidImage,
    "LiquidVideo": LiquidVideo,
    "LiquidVideoJob": LiquidVideoJob,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LiquidImage": "Liquid Image (paid API)",
    "LiquidVideo": "Liquid Video (paid API)",
    "LiquidVideoJob": "Liquid Video Job",
}

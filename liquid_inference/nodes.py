"""The ComfyUI node. This is the only module that uses torch."""

from __future__ import annotations

import re
from functools import cache
from typing import Any

from .client import Client, LiquidError, receipt
from .config import comfy_config
from .images import decode_batch
from .models import Catalog, size_choices

NO_MODELS = "(no image models: check the connection, then reload)"
SIZE = re.compile(r"\d+x\d+")


def _client() -> Client:
    config = comfy_config()
    key = config.api_key()
    return Client(config.base_url().value, key.value if key else None)


@cache
def _catalog() -> Catalog:
    return Catalog(_client, comfy_config().directory / "image_models.json")


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
            }
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
        self, prompt: str, model: str, size: str, n: int, cap_usd: float
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
        return (
            torch.from_numpy(batch),
            generation.job_id,
            receipt(generation, model, cap_usd),
        )


NODE_CLASS_MAPPINGS = {"LiquidImage": LiquidImage}
NODE_DISPLAY_NAME_MAPPINGS = {"LiquidImage": "Liquid Image (paid API)"}

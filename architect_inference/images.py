"""Image bytes to the array layout of a ComfyUI IMAGE, without torch."""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps, UnidentifiedImageError


def decode_batch(images: list[bytes]) -> npt.NDArray[np.float32]:
    """Decodes PNG, JPEG or WEBP bytes to float32 [batch, height, width, 3] in 0..1.

    One batch must have one size. An image with a different size is resized
    to the size of the first image.
    """
    if not images:
        raise ValueError("no images to decode")
    pictures = [_open(data, index) for index, data in enumerate(images)]
    size = pictures[0].size
    arrays = [
        np.asarray(
            picture if picture.size == size else picture.resize(size, Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        / 255.0
        for picture in pictures
    ]
    return np.stack(arrays).astype(np.float32)


def _open(data: bytes, index: int) -> Image.Image:
    try:
        picture: Image.Image = Image.open(io.BytesIO(data))
        picture.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"image {index + 1} is not a PNG, JPEG or WEBP file: {error}") from None
    picture = ImageOps.exif_transpose(picture) or picture
    return picture.convert("RGB")

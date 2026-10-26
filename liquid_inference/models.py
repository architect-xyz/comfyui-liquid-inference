"""The lists of image and video models for the `model`, `size` and `seconds` inputs.

ComfyUI reads the inputs of a node each time it serves `/object_info` and
each time it checks a queued workflow. A network call there blocks ComfyUI.
Thus the catalog returns the last list at once and gets a new list in a
background thread. It waits for the network only when it has no list at all.

Each list belongs to the router that sent it. The catalog never returns the
list of a different router, from memory or from the cache file.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

log = logging.getLogger(__name__)

IMAGE_KIND = "image_generation"
VIDEO_KIND = "video_generation"
FETCH_TIMEOUT = 5.0
MAX_AGE = 60.0


@dataclass(frozen=True)
class ImageModel:
    id: str
    sizes: tuple[str, ...]
    rate_per_image_usd: float | None = None


@dataclass(frozen=True)
class VideoModel:
    id: str
    sizes: tuple[str, ...]
    seconds: tuple[int, ...]
    # True when an offer for the model reads an image input.
    start_image: bool
    rate_per_video_second_usd: float | None = None


M = TypeVar("M", ImageModel, VideoModel)


@dataclass(frozen=True)
class _Listing(Generic[M]):
    base_url: str
    models: list[M]


class Router(Protocol):
    @property
    def base_url(self) -> str: ...

    def list_image_models(self, timeout: float) -> list[dict[str, Any]]: ...

    def list_video_models(self, timeout: float) -> list[dict[str, Any]]: ...


class _Sized(Protocol):
    @property
    def sizes(self) -> tuple[str, ...]: ...


def image_models(cards: list[dict[str, Any]]) -> list[ImageModel]:
    models = []
    for card in cards:
        model_id = card.get("model", card.get("id"))
        if card.get("kind") != IMAGE_KIND or not isinstance(model_id, str):
            continue
        output = _modality(card.get("output_modalities"), "image") or {}
        sizes = card.get("image_sizes", output.get("sizes"))
        sizes = sizes if isinstance(sizes, list) else []
        models.append(
            ImageModel(
                model_id,
                tuple(s for s in sizes if isinstance(s, str)),
                _rate(card, "rate_per_image_usd"),
            )
        )
    return models


def video_models(entries: list[dict[str, Any]]) -> list[VideoModel]:
    """Reads the entries of `GET /exchange/models?kind=video`.

    Each list of an entry is the union across the offers for the model.
    """
    models = []
    for entry in entries:
        model_id = entry.get("model")
        if entry.get("kind") != VIDEO_KIND or not isinstance(model_id, str):
            continue
        output = _modality(entry.get("output_modalities"), "video") or {}
        sizes = output.get("sizes")
        seconds = output.get("seconds")
        models.append(
            VideoModel(
                model_id,
                tuple(s for s in sizes if isinstance(s, str)) if isinstance(sizes, list) else (),
                # bool is a subclass of int, and JSON `true` is not a length.
                tuple(s for s in seconds if type(s) is int) if isinstance(seconds, list) else (),
                _modality(entry.get("input_modalities"), "image") is not None,
                _rate(entry, "rate_per_video_second_usd"),
            )
        )
    return models


def _rate(entry: dict[str, Any], field: str) -> float | None:
    rate = entry.get(field)
    value = rate.get("min") if isinstance(rate, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _modality(modalities: object, kind: str) -> dict[str, Any] | None:
    if isinstance(modalities, list):
        for modality in modalities:
            if isinstance(modality, dict) and modality.get("type") == kind:
                return modality
    return None


def seconds_choices(models: Sequence[VideoModel]) -> list[str]:
    """Returns "auto" and then each length that a model lists, shortest first.

    "auto" omits `seconds`, and the exchange then uses the shortest length
    that an offer declares.
    """
    seconds = {length for model in models for length in model.seconds}
    return ["auto", *(str(length) for length in sorted(seconds))]


def size_choices(models: Sequence[_Sized]) -> list[str]:
    """Returns "auto" and then each size that a model lists, smallest first.

    "auto" omits `size` from the request, so each seller uses its default.
    A seller can list "auto" as a size, but omitting the field matches more
    offers than sending it.
    """
    sizes = {size for model in models for size in model.sizes if size != "auto"}
    return ["auto", *sorted(sizes, key=_area)]


def _area(size: str) -> tuple[int, int, str]:
    width, _, height = size.partition("x")
    if width.isdigit() and height.isdigit():
        return (0, int(width) * int(height), size)
    return (1, 0, size)


@dataclass(frozen=True)
class Kind(Generic[M]):
    """Where a catalog gets its list, and the type of each entry."""

    model: type[M]
    fetch: Callable[[Router, float], list[M]]


IMAGES = Kind(ImageModel, lambda router, timeout: image_models(router.list_image_models(timeout)))
VIDEOS = Kind(VideoModel, lambda router, timeout: video_models(router.list_video_models(timeout)))


def _tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


class Catalog(Generic[M]):
    def __init__(
        self,
        router: Callable[[], Router],
        cache_file: Path | None,
        kind: Kind[M],
        *,
        max_age: float = MAX_AGE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._router = router
        self._kind: Kind[M] = kind
        self._cache_file = cache_file
        self._max_age = max_age
        self._clock = clock
        self._lock = threading.Lock()
        self._listing: _Listing[M] | None = None
        self._tried: tuple[str, float] | None = None
        self._refresh: threading.Thread | None = None

    def models(self) -> list[M]:
        """The last list from the current router, or an empty list when there is none."""
        router = self._router()
        with self._lock:
            listing = self._listing
            if listing is None or listing.base_url != router.base_url:
                listing = self._listing = self._read_cache(router.base_url)
            if listing is None:
                if self._stale(router.base_url):
                    listing = self._listing = self._try_fetch(router)
                return list(listing.models) if listing else []
            if self._stale(router.base_url) and (
                self._refresh is None or not self._refresh.is_alive()
            ):
                self._tried = (router.base_url, self._clock())
                self._refresh = threading.Thread(
                    target=self._refresh_in_background,
                    args=(router,),
                    name="liquid-models",
                    daemon=True,
                )
                self._refresh.start()
            return list(listing.models)

    def wait(self) -> None:
        """Waits for a background refresh to stop. For tests."""
        refresh = self._refresh
        if refresh is not None:
            refresh.join()

    def _stale(self, base_url: str) -> bool:
        if self._tried is None or self._tried[0] != base_url:
            return True
        return self._clock() - self._tried[1] >= self._max_age

    def _refresh_in_background(self, router: Router) -> None:
        listing = self._try_fetch(router)
        if listing is not None:
            with self._lock:
                self._listing = listing

    def _try_fetch(self, router: Router) -> _Listing[M] | None:
        self._tried = (router.base_url, self._clock())
        try:
            listing = _Listing(router.base_url, self._kind.fetch(router, FETCH_TIMEOUT))
        except Exception as error:
            log.warning("liquid inference: failed to get the model list: %s", error)
            return None
        self._write_cache(listing)
        return listing

    def _read_cache(self, base_url: str) -> _Listing[M] | None:
        if self._cache_file is None:
            return None
        try:
            cached = json.loads(self._cache_file.read_text(encoding="utf-8"))
            if cached.get("base_url") != base_url:
                return None
            # JSON has no tuples, and the model types hold tuples.
            models = [
                self._kind.model(**{k: _tuple(v) for k, v in m.items()}) for m in cached["models"]
            ]
            return _Listing(base_url, models)
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            log.warning("liquid inference: ignored an unreadable model cache: %s", error)
            return None

    def _write_cache(self, listing: _Listing[M]) -> None:
        if self._cache_file is None:
            return
        try:
            self._cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = {
                "base_url": listing.base_url,
                "models": [asdict(m) for m in listing.models],
            }
            temporary = self._cache_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self._cache_file)
        except OSError as error:
            log.warning("liquid inference: failed to write the model cache: %s", error)

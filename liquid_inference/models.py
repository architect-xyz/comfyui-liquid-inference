"""The list of image models for the `model` and `size` inputs.

ComfyUI reads the inputs of a node each time it serves `/object_info` and
each time it checks a queued workflow. A network call there blocks ComfyUI.
Thus the catalog returns the last list at once and gets a new list in a
background thread. It waits for the network only when it has no list at all.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

IMAGE_KIND = "image_generation"
FETCH_TIMEOUT = 5.0
MAX_AGE = 60.0


@dataclass(frozen=True)
class ImageModel:
    id: str
    sizes: tuple[str, ...]


def image_models(cards: list[dict[str, Any]]) -> list[ImageModel]:
    models = []
    for card in cards:
        model_id = card.get("id")
        if card.get("kind") != IMAGE_KIND or not isinstance(model_id, str):
            continue
        sizes = card.get("image_sizes")
        sizes = sizes if isinstance(sizes, list) else []
        models.append(ImageModel(model_id, tuple(s for s in sizes if isinstance(s, str))))
    return models


def size_choices(models: list[ImageModel]) -> list[str]:
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


class Catalog:
    def __init__(
        self,
        fetch: Callable[[float], list[dict[str, Any]]],
        cache_file: Path | None,
        *,
        max_age: float = MAX_AGE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._cache_file = cache_file
        self._max_age = max_age
        self._clock = clock
        self._lock = threading.Lock()
        self._models: list[ImageModel] | None = None
        self._tried_at: float | None = None
        self._refresh: threading.Thread | None = None

    def models(self) -> list[ImageModel]:
        """The last list, or an empty list when there is none."""
        with self._lock:
            if self._models is None:
                self._models = self._read_cache()
            if self._models is None:
                if self._tried_at is None or self._stale():
                    self._models = self._try_fetch()
                return list(self._models or [])
            if self._stale() and (self._refresh is None or not self._refresh.is_alive()):
                self._tried_at = self._clock()
                self._refresh = threading.Thread(
                    target=self._refresh_in_background, name="liquid-models", daemon=True
                )
                self._refresh.start()
            return list(self._models)

    def wait(self) -> None:
        """Waits for a background refresh to stop. For tests."""
        refresh = self._refresh
        if refresh is not None:
            refresh.join()

    def _stale(self) -> bool:
        return self._tried_at is None or self._clock() - self._tried_at >= self._max_age

    def _refresh_in_background(self) -> None:
        models = self._try_fetch()
        if models is not None:
            with self._lock:
                self._models = models

    def _try_fetch(self) -> list[ImageModel] | None:
        self._tried_at = self._clock()
        try:
            models = image_models(self._fetch(FETCH_TIMEOUT))
        except Exception as error:
            log.warning("liquid inference: failed to get the model list: %s", error)
            return None
        self._write_cache(models)
        return models

    def _read_cache(self) -> list[ImageModel] | None:
        if self._cache_file is None:
            return None
        try:
            cached = json.loads(self._cache_file.read_text(encoding="utf-8"))
            return [ImageModel(m["id"], tuple(m["sizes"])) for m in cached["models"]]
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError) as error:
            log.warning("liquid inference: ignored an unreadable model cache: %s", error)
            return None

    def _write_cache(self, models: list[ImageModel]) -> None:
        if self._cache_file is None:
            return
        try:
            self._cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = {"models": [{"id": m.id, "sizes": list(m.sizes)} for m in models]}
            temporary = self._cache_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self._cache_file)
        except OSError as error:
            log.warning("liquid inference: failed to write the model cache: %s", error)

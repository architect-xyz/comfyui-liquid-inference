import socket
from pathlib import Path
from typing import Any

from stub_router import Stub

from liquid_inference.client import Client
from liquid_inference.models import Catalog, ImageModel, image_models, size_choices


def dead_base() -> str:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{reserved.getsockname()[1]}/v1"


def catalog(base: str, cache: Path, clock: list[float] | None = None) -> Catalog:
    now = clock if clock is not None else [0.0]
    return Catalog(Client(base, None).list_models, cache, clock=lambda: now[0])


def test_only_image_models_are_listed(router: Stub) -> None:
    models = image_models(Client(f"{router.base}/v1", None).list_models(5.0))
    assert [m.id for m in models] == [
        "blackforestlabs/flux-schnell",
        "bytedance/seedream-4.5",
        "openai/gpt-image",
    ]
    assert models[1].sizes == ()


def test_sizes_are_auto_then_the_union_smallest_first() -> None:
    models = [
        ImageModel("a", ("1024x1024", "auto", "512x512")),
        ImageModel("b", ("1536x1024", "1024x1024")),
    ]
    assert size_choices(models) == ["auto", "512x512", "1024x1024", "1536x1024"]
    assert size_choices([]) == ["auto"]


def test_the_cache_answers_when_the_router_is_down(router: Stub, tmp_path: Path) -> None:
    cache = tmp_path / "image_models.json"
    first = catalog(f"{router.base}/v1", cache).models()
    assert len(first) == 3

    down = catalog(dead_base(), cache)
    assert down.models() == first
    down.wait()
    assert down.models() == first


def test_with_no_list_a_failed_fetch_waits_before_it_tries_again(tmp_path: Path) -> None:
    calls: list[float] = []
    now = [0.0]

    def fail(timeout: float) -> list[dict[str, Any]]:
        calls.append(timeout)
        raise OSError("down")

    models = Catalog(fail, tmp_path / "image_models.json", clock=lambda: now[0])
    assert models.models() == []
    assert models.models() == []
    assert len(calls) == 1
    now[0] = 61.0
    assert models.models() == []
    assert len(calls) == 2


def test_a_stale_list_refreshes_in_the_background(router: Stub, tmp_path: Path) -> None:
    now = [0.0]
    models = catalog(f"{router.base}/v1", tmp_path / "m.json", now)
    assert len(models.models()) == 3
    router.cards = router.cards[:1]
    assert len(models.models()) == 3
    now[0] = 61.0
    assert len(models.models()) == 3
    models.wait()
    assert [m.id for m in models.models()] == ["blackforestlabs/flux-schnell"]
    assert len(router.calls("/v1/models")) == 2

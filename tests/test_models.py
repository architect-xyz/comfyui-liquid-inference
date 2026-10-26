from pathlib import Path
from typing import Any

from stub_router import Stub, dead_base, serve

from liquid_inference.client import Client
from liquid_inference.models import (
    IMAGES,
    VIDEOS,
    Catalog,
    ImageModel,
    VideoModel,
    image_models,
    seconds_choices,
    size_choices,
    video_models,
)


class Down:
    """A router that refuses each request."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.calls = 0

    def list_models(self, timeout: float) -> list[dict[str, Any]]:
        self.calls += 1
        raise OSError("down")

    list_video_models = list_models
    list_image_models = list_models


def catalog(base: str, cache: Path, clock: list[float] | None = None) -> Catalog[ImageModel]:
    now = clock if clock is not None else [0.0]
    client = Client(base, None)
    return Catalog(lambda: client, cache, IMAGES, clock=lambda: now[0])


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

    down = Down(f"{router.base}/v1")
    models = Catalog(lambda: down, cache, IMAGES)
    assert models.models() == first
    models.wait()
    assert models.models() == first
    assert down.calls == 1


def test_the_cache_of_another_router_is_not_shown(router: Stub, tmp_path: Path) -> None:
    cache = tmp_path / "image_models.json"
    assert len(catalog(f"{router.base}/v1", cache).models()) == 3

    assert catalog(dead_base(), cache).models() == []
    with serve(Stub(cards=router.cards[:1], image_entries=router.cards[:1])) as other:
        models = catalog(f"{other.base}/v1", cache).models()
    assert [m.id for m in models] == ["blackforestlabs/flux-schnell"]


def test_a_new_base_url_gets_its_list_at_once(router: Stub, tmp_path: Path) -> None:
    with serve(Stub(cards=router.cards[:1], image_entries=router.cards[:1])) as other:
        clients = [Client(f"{router.base}/v1", None)]
        models = Catalog(lambda: clients[0], tmp_path / "m.json", IMAGES, clock=lambda: 0.0)
        assert len(models.models()) == 3
        clients[0] = Client(f"{other.base}/v1", None)
        assert [m.id for m in models.models()] == ["blackforestlabs/flux-schnell"]


def test_with_no_list_a_failed_fetch_waits_before_it_tries_again(tmp_path: Path) -> None:
    now = [0.0]
    down = Down(dead_base())
    models = Catalog(lambda: down, tmp_path / "image_models.json", IMAGES, clock=lambda: now[0])
    assert models.models() == []
    assert models.models() == []
    assert down.calls == 1
    now[0] = 61.0
    assert models.models() == []
    assert down.calls == 2


def test_a_stale_list_refreshes_in_the_background(router: Stub, tmp_path: Path) -> None:
    now = [0.0]
    models = catalog(f"{router.base}/v1", tmp_path / "m.json", now)
    assert len(models.models()) == 3
    router.cards = router.cards[:1]
    router.image_entries = router.image_entries[:1]
    assert len(models.models()) == 3
    now[0] = 61.0
    assert len(models.models()) == 3
    models.wait()
    assert [m.id for m in models.models()] == ["blackforestlabs/flux-schnell"]
    assert len(router.calls("/v1/exchange/models?kind=image")) == 2


def test_video_models_list_their_sizes_lengths_and_start_image(router: Stub) -> None:
    models = video_models(Client(f"{router.base}/v1", None).list_video_models(5.0))
    assert models == [
        VideoModel("google/veo-3.1", ("1920x1080", "1280x720"), (8, 4), True),
        VideoModel("kling/kling-2.5", (), (5, 10), False),
    ]
    assert size_choices(models) == ["auto", "1280x720", "1920x1080"]
    assert seconds_choices(models) == ["auto", "4", "5", "8", "10"]


def test_the_video_list_is_fetched_without_the_key(router: Stub) -> None:
    Client(f"{router.base}/v1", "lq_some_key").list_video_models(5.0)
    (request,) = router.calls("/v1/exchange/models?kind=video")
    assert "authorization" not in request.headers


def test_the_video_cache_answers_when_the_router_is_down(router: Stub, tmp_path: Path) -> None:
    cache = tmp_path / "video_models.json"
    client = Client(f"{router.base}/v1", None)
    first = Catalog(lambda: client, cache, VIDEOS).models()
    assert len(first) == 2

    down = Down(f"{router.base}/v1")
    assert Catalog(lambda: down, cache, VIDEOS).models() == first


def test_directory_prices_share_the_catalog_cache(router: Stub, tmp_path: Path) -> None:
    router.image_entries = [
        {
            "model": "priced/image",
            "kind": "image_generation",
            "rate_per_image_usd": {"min": 0.04},
            "output_modalities": [{"type": "image", "sizes": ["1536x1536"]}],
        }
    ]
    models = catalog(f"{router.base}/v1", tmp_path / "prices.json")
    first = models.models()
    assert first[0].id == "priced/image"
    assert first[0].rate_per_image_usd == 0.04
    assert first[0].sizes == ("1536x1536",)
    assert models.models() == first
    assert len(router.calls("/v1/exchange/models?kind=image")) == 1


def test_invalid_prices_are_unknown() -> None:
    for rate in [None, {}, {"min": True}, {"min": -1}, {"min": float("inf")}]:
        model = image_models(
            [{"id": "image", "kind": "image_generation", "rate_per_image_usd": rate}]
        )[0]
        assert model.rate_per_image_usd is None

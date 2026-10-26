import asyncio
import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from stub_router import CLIP, VIDEO_JOB, Stub, video_error, video_status

from liquid_inference import __version__, nodes, videos
from liquid_inference.client import Client, LiquidError
from liquid_inference.models import VIDEOS, Catalog

KEY = "lq_video_key_9"
CREATE = "/v1/videos"
STATUS = f"/v1/videos/{VIDEO_JOB}"
CONTENT = f"{STATUS}/content"


@pytest.fixture
def client(router: Stub) -> Client:
    return Client(f"{router.base}/v1", KEY, timeout=2.0)


@pytest.fixture
def node(
    router: Stub, client: Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> nodes.LiquidVideo:
    catalog = Catalog(lambda: client, tmp_path / "video_models.json", VIDEOS)
    monkeypatch.setattr(nodes, "_client", lambda: client)
    monkeypatch.setattr(nodes, "_video_catalog", lambda: catalog)
    monkeypatch.setattr(nodes, "_interrupted", lambda: False)
    # VideoFromFile is ComfyUI's. The tests keep the bytes.
    monkeypatch.setattr(nodes, "_video", lambda data: data)
    return nodes.LiquidVideo()


def wait(client: Client, **options: Any) -> dict[str, Any]:
    options.setdefault("interrupted", lambda: False)
    return asyncio.run(
        videos.wait(client, VIDEO_JOB, report=lambda _: None, interval=0.0, **options)
    )


def download(client: Client) -> bytes:
    return asyncio.run(videos.download(client, VIDEO_JOB, interrupted=lambda: False, pause=0.0))


def buy(node: nodes.LiquidVideo, **overrides: Any) -> tuple[Any, str, str]:
    inputs: dict[str, Any] = {
        "prompt": "a red square",
        "model": "google/veo-3.1",
        "size": "auto",
        "seconds": "auto",
        "cap_usd": 0.8,
    }
    inputs.update(overrides)
    return asyncio.run(node.generate(**inputs))


def test_one_purchase_then_reads_until_the_file_comes(
    node: nodes.LiquidVideo, router: Stub
) -> None:
    video, job_id, receipt = buy(node, size="1280x720", seconds="4")
    (create,) = router.calls(CREATE)
    assert create.json() == {"model": "google/veo-3.1", "prompt": "a red square",
                             "size": "1280x720", "seconds": "4",
                             "exchange": {"retention_policy": "none"}}  # fmt: skip
    assert create.headers["x-liquid-cap-usd"] == "0.8"
    assert create.headers["authorization"] == f"Bearer {KEY}"
    assert video == CLIP
    assert job_id == VIDEO_JOB
    assert json.loads(receipt) == {
        "job_id": VIDEO_JOB,
        "model": "google/veo-3.1-fast",
        "provider": "stub-seller",
        "charge_usd": 0.8,
        "cap_usd": 0.8,
        "seconds": "4",
        "size": "1280x720",
        "settlement_status": "settled",
        "expires_at": 1_900_000_000,
        "node_version": __version__,
    }


def test_auto_omits_the_size_and_the_length(node: nodes.LiquidVideo, router: Stub) -> None:
    buy(node)
    (create,) = router.calls(CREATE)
    assert "size" not in create.json()
    assert "seconds" not in create.json()


@pytest.mark.parametrize("policy", ["none", "allowed"])
def test_the_video_purchase_uses_the_selected_retention_policy(
    node: nodes.LiquidVideo, router: Stub, policy: str
) -> None:
    video, _, _ = buy(node, retention_policy=policy)
    (create,) = router.calls(CREATE)
    assert create.json()["exchange"] == {"retention_policy": policy}
    assert video == CLIP


def test_an_invalid_retention_policy_stops_before_any_request(
    node: nodes.LiquidVideo, router: Stub
) -> None:
    with pytest.raises(LiquidError, match="retention_policy"):
        buy(node, retention_policy="all")
    assert router.requests == []


def test_a_start_image_goes_as_one_png_data_url(node: nodes.LiquidVideo, router: Stub) -> None:
    frame = np.zeros((1, 6, 8, 3), dtype=np.float32)
    frame[0, :, :, 0] = 1.0
    buy(node, start_image=frame)
    url = router.calls(CREATE)[0].json()["input_reference"]["image_url"]
    assert url.startswith("data:image/png;base64,")
    picture = Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2])))
    assert picture.size == (8, 6)
    assert picture.getpixel((0, 0)) == (255, 0, 0)


def test_a_batch_of_two_start_images_is_refused_before_any_request(
    node: nodes.LiquidVideo, router: Stub
) -> None:
    with pytest.raises(LiquidError, match="holds 2 images"):
        buy(node, start_image=np.zeros((2, 6, 8, 3), dtype=np.float32))
    assert router.requests == []


def test_a_create_that_times_out_is_not_sent_again(router: Stub) -> None:
    router.delay = 1.0
    client = Client(f"{router.base}/v1", KEY, timeout=0.2)
    with pytest.raises(LiquidError, match="outcome is unknown") as error:
        client.create_video(
            model="m", prompt="p", size=None, seconds=None, start_image=None, cap_usd=1.0
        )
    (create,) = router.calls(CREATE)
    assert create.headers["idempotency-key"] in str(error.value)


def test_failed_reads_are_tried_again_and_the_clip_still_comes(
    router: Stub, client: Client
) -> None:
    router.answer_statuses(
        video_error("storage_unavailable"),
        (502, b"<html><title>502: Bad gateway</title></html>"),
        video_status("in_progress"),
        video_status(),
    )
    router.contents = [video_error("video_not_ready", 404), (200, CLIP)]
    assert wait(client)["status"] == "completed"
    assert download(client) == CLIP
    assert len(router.calls(STATUS)) == 4
    assert len(router.calls(CONTENT)) == 2


def test_too_many_failed_reads_stop_with_the_job_id(router: Stub, client: Client) -> None:
    router.answer_statuses(video_error("storage_unavailable"))
    with pytest.raises(LiquidError, match=f"put job ID {VIDEO_JOB} in a Liquid Video Job node"):
        wait(client)
    assert len(router.calls(STATUS)) == videos.MAX_FAILED_READS + 1


def test_a_failed_job_says_that_it_costs_nothing(router: Stub, client: Client) -> None:
    router.answer_statuses(
        video_status("failed", error={"code": "content_policy", "message": "refused"})
    )
    with pytest.raises(LiquidError, match=r"\(content_policy\): refused\. A failed job costs"):
        wait(client)


def test_a_read_that_the_exchange_refuses_is_not_tried_again(router: Stub, client: Client) -> None:
    router.contents = [video_error("video_expired", 410)]
    with pytest.raises(LiquidError, match="video_expired"):
        download(client)
    assert len(router.calls(CONTENT)) == 1


def test_a_stop_ends_the_wait_and_names_the_job(router: Stub, client: Client) -> None:
    router.answer_statuses(video_status("in_progress"))
    with pytest.raises(LiquidError, match=f"stopped waiting for job {VIDEO_JOB}"):
        asyncio.run(videos.wait(client, VIDEO_JOB, interrupted=lambda: True, report=lambda _: None))
    assert len(router.calls(STATUS)) == 1


def test_a_job_that_does_not_end_in_time_names_the_job(router: Stub, client: Client) -> None:
    router.answer_statuses(video_status("in_progress"))
    with pytest.raises(LiquidError, match=f"Job {VIDEO_JOB} did not end"):
        wait(client, limit=0.0)


def test_the_job_node_gets_the_video_without_a_purchase(
    node: nodes.LiquidVideo, router: Stub
) -> None:
    video, receipt = asyncio.run(nodes.LiquidVideoJob().load(f" {VIDEO_JOB} "))
    assert video == CLIP
    assert json.loads(receipt)["job_id"] == VIDEO_JOB
    assert router.calls(CREATE) == []
    assert nodes.LiquidVideoJob.VALIDATE_INPUTS("") is not True
    assert nodes.LiquidVideoJob.VALIDATE_INPUTS("../../account") is not True


def test_the_inputs_list_the_video_models_and_no_input_can_hold_a_key(
    node: nodes.LiquidVideo,
) -> None:
    inputs = nodes.LiquidVideo.INPUT_TYPES()
    widgets = {name: spec for group in inputs.values() for name, spec in group.items()}
    assert widgets["model"][0] == ["google/veo-3.1", "kling/kling-2.5"]
    assert widgets["seconds"][0] == ["auto", "4", "5", "8", "10"]
    for name in widgets:
        assert not re.search("key|token|secret|auth|password", name, re.IGNORECASE)
    assert nodes.LiquidVideo.VALIDATE_INPUTS("m", "auto", "21") is not True
    assert nodes.LiquidVideo.VALIDATE_INPUTS("m", "1080x1920", "12") is True

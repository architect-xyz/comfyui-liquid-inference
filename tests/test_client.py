import json
import socket
import time
import uuid

import numpy as np
import pytest
from stub_router import Stub, png

from liquid_inference import __version__
from liquid_inference.client import Client, LiquidError, refusal_message
from liquid_inference.images import decode_batch

KEY = "lq_test_key_123"
GENERATE = "/v1/images/generations"


def client(router: Stub, timeout: float = 5.0) -> Client:
    return Client(f"{router.base}/v1", KEY, timeout=timeout)


def buy(router: Stub, **overrides: object) -> object:
    request: dict[str, object] = {
        "model": "blackforestlabs/flux-schnell",
        "prompt": "a red square",
        "n": 1,
        "size": None,
        "cap_usd": 0.05,
    }
    request.update(overrides)
    return client(router).generate(**request)  # type: ignore[arg-type]


def test_a_base64_image_decodes_to_a_batch(router: Stub) -> None:
    generation = client(router).generate(model="m", prompt="p", n=2, size=None, cap_usd=0.05)
    batch = decode_batch(generation.images)
    assert batch.shape == (2, 6, 8, 3)
    assert batch.dtype == np.float32
    assert np.allclose(batch[0, 0, 0], [1.0, 0.0, 0.0])
    assert generation.job_id == "job-stub-1"


def test_an_rgba_image_becomes_rgb() -> None:
    batch = decode_batch([png(4, 4, (0, 255, 0, 128))])
    assert batch.shape == (1, 4, 4, 3)


def test_images_of_two_sizes_become_one_batch() -> None:
    assert decode_batch([png(8, 6), png(4, 4)]).shape == (2, 6, 8, 3)


def test_a_url_image_is_downloaded_without_the_key(router: Stub) -> None:
    router.mode = "url"
    generation = client(router).generate(model="m", prompt="p", n=1, size=None, cap_usd=0.05)
    assert decode_batch(generation.images).shape == (1, 6, 8, 3)
    (download,) = router.calls("/files/image.png")
    assert "authorization" not in download.headers
    assert KEY not in json.dumps(download.headers)
    (post,) = router.calls(GENERATE)
    assert post.headers["authorization"] == f"Bearer {KEY}"


def test_no_size_is_sent_unless_one_is_given(router: Stub) -> None:
    buy(router)
    buy(router, size="512x512")
    first, second = router.calls(GENERATE)
    assert "size" not in first.json()
    assert second.json()["size"] == "512x512"
    assert first.json() == {
        "model": "blackforestlabs/flux-schnell",
        "prompt": "a red square",
        "n": 1,
    }


def test_each_request_has_the_cap_a_new_idempotency_key_and_the_user_agent(router: Stub) -> None:
    buy(router, cap_usd=0.05)
    buy(router, cap_usd=1e-3)
    first, second = router.calls(GENERATE)
    assert first.headers["x-liquid-cap-usd"] == "0.05"
    assert second.headers["x-liquid-cap-usd"] == "0.001"
    keys = {first.headers["idempotency-key"], second.headers["idempotency-key"]}
    assert len(keys) == 2
    for key in keys:
        uuid.UUID(key)
    assert first.headers["user-agent"] == f"liquid-comfyui/{__version__}"
    assert first.headers["content-type"] == "application/json"


def test_a_refusal_is_not_sent_again(router: Stub) -> None:
    router.mode, router.status = "error", 503
    router.error = {
        "error": {"message": "no offer met the requirements", "code": "no_eligible_offer"}
    }
    with pytest.raises(LiquidError, match="no_eligible_offer"):
        buy(router)
    time.sleep(0.2)
    assert len(router.calls(GENERATE)) == 1


def test_a_timeout_is_not_sent_again_and_says_the_outcome_is_unknown(router: Stub) -> None:
    router.delay = 1.0
    with pytest.raises(LiquidError) as caught:
        client(router, timeout=0.2).generate(model="m", prompt="p", n=1, size=None, cap_usd=0.05)
    time.sleep(1.2)
    (post,) = router.calls(GENERATE)
    message = str(caught.value)
    assert "unknown" in message
    assert "charged" in message
    assert post.headers["idempotency-key"] in message
    assert KEY not in message


def test_a_router_that_cannot_be_reached_is_not_a_charge() -> None:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    with pytest.raises(LiquidError, match="no charge"):
        Client(f"http://127.0.0.1:{port}/v1", KEY).generate(
            model="m", prompt="p", n=1, size=None, cap_usd=0.05
        )


def test_a_missing_key_stops_before_any_request(router: Stub) -> None:
    with pytest.raises(LiquidError, match="LIQUID_API_KEY"):
        Client(f"{router.base}/v1", None).generate(
            model="m", prompt="p", n=1, size=None, cap_usd=0.05
        )
    assert router.requests == []


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            400,
            {
                "error": {
                    "message": "`size` is not supported by any offer for m",
                    "code": "unsupported_parameter",
                    "param": "size",
                    "eligibility": {
                        "excluded": [
                            {
                                "provider": "acme",
                                "reason": "media_unsupported",
                                "detail": "size",
                                "count": 3,
                            }
                        ]
                    },
                }
            },
            [
                "HTTP 400",
                "unsupported_parameter",
                "parameter: size",
                "acme: media_unsupported (size) x3",
            ],
        ),
        (
            402,
            {
                "error": {
                    "message": "your balance will not cover this job",
                    "code": "insufficient_balance",
                }
            },
            ["HTTP 402", "insufficient_balance", "balance will not cover"],
        ),
        (
            503,
            {
                "error": {
                    "message": "no offer met the requirements",
                    "code": "no_eligible_offer",
                    "eligibility": {
                        "excluded": [
                            {"provider": "acme", "reason": "cap_exceeds_buyer_cap", "count": 1},
                            {"provider": "beta", "reason": "offer_expired", "count": 2},
                        ]
                    },
                }
            },
            ["no_eligible_offer", "acme: cap_exceeds_buyer_cap", "beta: offer_expired x2"],
        ),
    ],
)
def test_an_error_names_the_code_and_the_exclusions(
    router: Stub, status: int, body: dict[str, object], expected: list[str]
) -> None:
    router.mode, router.status, router.error = "error", status, body
    with pytest.raises(LiquidError) as caught:
        buy(router)
    message = str(caught.value)
    for text in [*expected, "Job ID: job-stub-1"]:
        assert text in message


def test_an_error_body_that_is_not_json_is_still_readable() -> None:
    message = refusal_message(502, b"<html>Bad gateway</html>", None)
    assert message == "Liquid Inference answered HTTP 502: <html>Bad gateway</html>"

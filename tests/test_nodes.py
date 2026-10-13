import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from stub_router import Stub

from liquid_inference import __version__, nodes
from liquid_inference.client import Client
from liquid_inference.models import Catalog

KEY = "lq_node_key_7"


@pytest.fixture
def node(router: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> nodes.LiquidImage:
    # torch is large, and the node only calls `torch.from_numpy`.
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=lambda a: a))
    client = Client(f"{router.base}/v1", KEY)
    catalog = Catalog(client.list_models, tmp_path / "models.json")
    monkeypatch.setattr(nodes, "_client", lambda: client)
    monkeypatch.setattr(nodes, "_catalog", lambda: catalog)
    return nodes.LiquidImage()


def widgets() -> dict[str, Any]:
    inputs: dict[str, Any] = nodes.LiquidImage.INPUT_TYPES()
    return {name: spec for group in inputs.values() for name, spec in group.items()}


def test_no_input_can_hold_a_key(node: nodes.LiquidImage) -> None:
    names = set(widgets())
    assert names == {"prompt", "model", "size", "n", "cap_usd"}
    for name in names:
        assert not re.search("key|token|secret|auth|password", name, re.IGNORECASE)


def test_the_inputs_list_the_image_models_and_their_sizes(node: nodes.LiquidImage) -> None:
    inputs = widgets()
    assert inputs["model"][0] == [
        "blackforestlabs/flux-schnell",
        "bytedance/seedream-4.5",
        "openai/gpt-image",
    ]
    assert inputs["size"][0] == ["auto", "512x512", "1024x1024", "1536x1024"]
    assert inputs["size"][1]["default"] == "auto"


def test_with_no_list_the_model_input_shows_a_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(timeout: float) -> list[dict[str, Any]]:
        raise OSError("down")

    monkeypatch.setattr(nodes, "_catalog", lambda: Catalog(fail, tmp_path / "m.json"))
    assert widgets()["model"][0] == [nodes.NO_MODELS]
    assert nodes.LiquidImage.VALIDATE_INPUTS(nodes.NO_MODELS, "auto") is not True


def test_validation_accepts_a_model_the_local_list_does_not_have() -> None:
    assert nodes.LiquidImage.VALIDATE_INPUTS("new/model", "2048x2048") is True
    assert nodes.LiquidImage.VALIDATE_INPUTS("new/model", "huge") is not True


def test_auto_omits_the_size_and_the_outputs_hold_the_job(
    node: nodes.LiquidImage, router: Stub
) -> None:
    images, job_id, receipt = node.generate("a cat", "openai/gpt-image", "auto", 2, 0.05)
    node.generate("a cat", "openai/gpt-image", "1024x1024", 1, 0.05)
    first, second = router.calls("/v1/images/generations")
    assert "size" not in first.json()
    assert second.json()["size"] == "1024x1024"
    assert images.shape == (2, 6, 8, 3)
    assert job_id == "job-stub-1"
    assert json.loads(receipt) == {
        "job_id": "job-stub-1",
        "model": "stub/served-model",
        "provider": "stub-seller",
        "charge_usd": 0.04,
        "cap_usd": 0.05,
        "images": 2,
        "settlement_status": "settled",
        "node_version": __version__,
    }


def test_the_version_matches_the_project() -> None:
    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', project, re.MULTILINE)
    assert match is not None
    assert match.group(1) == __version__

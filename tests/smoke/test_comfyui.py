"""Runs the example workflows in a real ComfyUI against the stub router.

Run it with `uv run pytest -m smoke`. It installs a pinned ComfyUI release
and CPU torch, so the first run takes some minutes. Set LIQUID_SMOKE_DIR to
keep the installation for the next run.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from comfyui_install import install
from PIL import Image
from stub_router import Stub, serve

PACKAGE = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((PACKAGE / "example_workflows").glob("*.json"))
IMAGE_WORKFLOWS = [p for p in WORKFLOWS if '"LiquidImage"' in p.read_text()]
VIDEO_WORKFLOWS = [p for p in WORKFLOWS if '"LiquidVideo"' in p.read_text()]
# A real MP4 of 8 frames, so that Save Video can read it.
CLIP = (PACKAGE / "tests" / "fixtures" / "clip.mp4").read_bytes()
SENTINEL = "lq_smoke_sentinel_" + uuid.uuid4().hex
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
# A combo whose value adds inputs, such as `format` of Save Video. The API
# names an added input `<combo>.<input>`.
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
LIQUID_NODES = {"LiquidImage", "LiquidVideo"}

pytestmark = pytest.mark.smoke


def free_port() -> int:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port: int = reserved.getsockname()[1]
        return port


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def post(url: str, payload: object) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise AssertionError(f"POST {url}: {error.code} {error.read().decode()}") from None


def to_api(workflow: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    """Converts a workflow in the UI format to the API format of `/prompt`.

    Widget values are in the order of the node's inputs, without the inputs
    that a link sets.
    """
    links = {link[0]: link for link in workflow["links"]}
    prompt = {}
    for node in workflow["nodes"]:
        info = object_info[node["type"]]
        order = info["input_order"]
        spec = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        linked = {i["name"]: i["link"] for i in node.get("inputs", []) if i.get("link") is not None}
        values = list(node.get("widgets_values", []))
        inputs: dict[str, Any] = {}
        for name in order.get("required", []) + order.get("optional", []):
            kind = spec[name][0]
            options = spec[name][1] if len(spec[name]) > 1 else {}
            if name in linked:
                link = links[linked[name]]
                inputs[name] = [str(link[1]), link[2]]
            elif kind == DYNAMIC_COMBO and not options.get("hidden"):
                inputs.update(_dynamic(name, options, values))
            elif isinstance(kind, list) or kind in WIDGET_TYPES:
                inputs[name] = values.pop(0)
                if options.get("control_after_generate"):
                    values.pop(0)
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prompt


def _dynamic(name: str, options: dict[str, Any], values: list[Any]) -> dict[str, Any]:
    key = values.pop(0)
    inputs = {name: key}
    (option,) = [o for o in options["options"] if o["key"] == key]
    for sub, spec in option["inputs"].get("required", {}).items():
        if spec[0] == DYNAMIC_COMBO:
            inputs.update(_dynamic(f"{name}.{sub}", spec[1], values))
        else:
            inputs[f"{name}.{sub}"] = values.pop(0)
    return inputs


@pytest.fixture(scope="module")
def comfyui() -> Iterator[tuple[str, Path, Path, Stub]]:
    keep = os.environ.get("LIQUID_SMOKE_DIR")
    root = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="liquid-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    python = install(root)
    nodes = root / "ComfyUI" / "custom_nodes" / "liquid-inference"
    if nodes.is_symlink() or nodes.exists():
        nodes.unlink()
    nodes.symlink_to(PACKAGE, target_is_directory=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    output, user = run_dir / "output", run_dir / "user"
    output.mkdir()
    user.mkdir()
    port = free_port()
    with serve() as stub:
        env = {
            **os.environ,
            "LIQUID_BASE_URL": f"{stub.base}/v1",
            "LIQUID_API_KEY": SENTINEL,
        }
        env.pop("PYTHONPATH", None)
        log = (run_dir / "comfyui.log").open("w")
        process = subprocess.Popen(
            [
                str(python),
                "main.py",
                "--cpu",
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
                "--output-directory",
                str(output),
                "--user-directory",
                str(user),
                "--disable-auto-launch",
            ],
            cwd=root / "ComfyUI",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 300
            while True:
                assert process.poll() is None, (run_dir / "comfyui.log").read_text()
                try:
                    get(f"{base}/system_stats")
                    break
                except (urllib.error.URLError, OSError):
                    assert time.monotonic() < deadline, "ComfyUI did not start in 300 s"
                    time.sleep(1)
            yield base, output, run_dir, stub
        finally:
            process.terminate()
            process.wait(timeout=30)
            log.close()
    if keep:
        print(f"ComfyUI log: {run_dir / 'comfyui.log'}", file=sys.stderr)
    else:
        shutil.rmtree(root, ignore_errors=True)


def test_the_node_loads_without_import_errors(comfyui: tuple[str, Path, Path, Stub]) -> None:
    base, _, run_dir, _ = comfyui
    info = get(f"{base}/object_info/LiquidImage")["LiquidImage"]
    assert info["display_name"] == "Liquid Image (paid API)"
    assert "blackforestlabs/flux-schnell" in info["input"]["required"]["model"][0]
    info = get(f"{base}/object_info/LiquidVideo")["LiquidVideo"]
    assert info["output"] == ["VIDEO", "STRING", "STRING"]
    assert "google/veo-3.1" in info["input"]["required"]["model"][0]
    assert "LiquidVideoJob" in get(f"{base}/object_info")
    log = (run_dir / "comfyui.log").read_text()
    assert "IMPORT FAILED" not in log, log
    with urllib.request.urlopen(f"{base}/extensions/liquid-inference/liquid.js") as script:
        assert b"registerExtension" in script.read()


def queue(base: str, prompt: dict[str, Any], workflow: dict[str, Any]) -> dict[str, Any]:
    """Queues a prompt as the ComfyUI page does, and waits until it stops."""
    queued = post(
        f"{base}/prompt",
        {
            "prompt": prompt,
            "client_id": "smoke",
            "extra_data": {"extra_pnginfo": {"workflow": workflow}},
        },
    )
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + 300
    while True:
        history = get(f"{base}/history/{prompt_id}").get(prompt_id)
        if history and history.get("status", {}).get("completed") is not None:
            status: dict[str, Any] = history["status"]
            return status
        assert time.monotonic() < deadline, "the prompt did not finish in 300 s"
        time.sleep(0.5)


def example(base: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The workflow and its API prompt.

    ComfyUI keeps the output of a node whose inputs did not change, also from
    an earlier prompt. The prompt text gets the workflow name, so that each
    workflow buys its own images.
    """
    text = path.read_text()
    assert SENTINEL not in text
    workflow = json.loads(text)
    prompt = to_api(workflow, get(f"{base}/object_info"))
    for node in prompt.values():
        if node["class_type"] in LIQUID_NODES:
            node["inputs"]["prompt"] += f" ({path.stem})"
    return workflow, prompt


def assert_no_key(path: Path) -> None:
    assert SENTINEL not in path.read_bytes().decode("latin-1"), path


@pytest.mark.parametrize("path", IMAGE_WORKFLOWS, ids=[p.stem for p in IMAGE_WORKFLOWS])
def test_an_example_workflow_saves_images_without_the_key(
    comfyui: tuple[str, Path, Path, Stub], path: Path
) -> None:
    base, output, _, stub = comfyui
    workflow, prompt = example(base, path)
    before = set(output.rglob("*.png"))
    posts_before = len(stub.calls("/v1/images/generations"))
    status = queue(base, prompt, workflow)
    assert status["status_str"] == "success", json.dumps(status)

    liquid_nodes = sum(1 for node in workflow["nodes"] if node["type"] == "LiquidImage")
    posts = stub.calls("/v1/images/generations")[posts_before:]
    assert len(posts) == liquid_nodes
    for request in posts:
        assert request.headers["authorization"] == f"Bearer {SENTINEL}"
        assert request.headers["x-liquid-cap-usd"] == "0.05"
        assert "size" not in request.json()

    written = set(output.rglob("*.png")) - before
    saves = sum(1 for node in workflow["nodes"] if node["type"] == "SaveImage")
    assert len(written) == saves
    for image_path in written:
        with Image.open(image_path) as image:
            chunks: dict[str, str] = dict(getattr(image, "text", {}))
        assert "prompt" in chunks
        assert "workflow" in chunks
        for value in chunks.values():
            assert SENTINEL not in value
        assert_no_key(image_path)


@pytest.mark.parametrize("path", VIDEO_WORKFLOWS, ids=[p.stem for p in VIDEO_WORKFLOWS])
@pytest.mark.parametrize("retention_policy", [None, "none", "allowed"])
def test_an_example_workflow_saves_a_video_without_the_key(
    comfyui: tuple[str, Path, Path, Stub], path: Path, retention_policy: str | None
) -> None:
    base, output, _, stub = comfyui
    stub.contents = [(200, CLIP)]
    workflow, prompt = example(base, path)
    for node in prompt.values():
        if node["class_type"] == "LiquidVideo":
            if retention_policy is None:
                node["inputs"].pop("retention_policy")
            else:
                node["inputs"]["retention_policy"] = retention_policy
    before = set(output.rglob("*.mp4"))
    posts_before = len(stub.calls("/v1/videos"))
    status = queue(base, prompt, workflow)
    assert status["status_str"] == "success", json.dumps(status)

    (create,) = stub.calls("/v1/videos")[posts_before:]
    assert create.headers["authorization"] == f"Bearer {SENTINEL}"
    assert create.headers["x-liquid-cap-usd"] == "1"
    assert create.json()["exchange"]["retention_policy"] == (retention_policy or "none")
    starts = any(node["type"] == "LoadImage" for node in workflow["nodes"])
    assert ("input_reference" in create.json()) == starts

    (written,) = set(output.rglob("*.mp4")) - before
    # Save Video writes the workflow into the file's metadata.
    assert b"workflow" in written.read_bytes()
    assert_no_key(written)


def test_a_second_queue_with_the_same_inputs_does_not_buy_again(
    comfyui: tuple[str, Path, Path, Stub],
) -> None:
    base, _, _, stub = comfyui
    path = PACKAGE / "example_workflows" / "liquid_text_to_image.json"
    workflow, prompt = example(base, path)
    for node in prompt.values():
        if node["class_type"] == "LiquidImage":
            node["inputs"]["prompt"] += " again"
    assert queue(base, prompt, workflow)["status_str"] == "success"
    posts = len(stub.calls("/v1/images/generations"))
    assert queue(base, prompt, workflow)["status_str"] == "success"
    assert len(stub.calls("/v1/images/generations")) == posts


def test_the_settings_route_stores_the_key_in_a_private_file(
    comfyui: tuple[str, Path, Path, Stub],
) -> None:
    base, _, run_dir, _ = comfyui
    stored = SENTINEL + "_stored"
    answer = post(f"{base}/api/liquid-inference/key", {"api_key": stored})
    assert stored not in json.dumps(answer)
    key_file = run_dir / "user" / "__liquid_inference" / "api_key"
    assert key_file.read_text() == stored
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    form = urllib.request.Request(
        f"{base}/liquid-inference/key",
        data=b"api_key=x",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(form, timeout=10)
    assert refused.value.code == 415
    post(f"{base}/liquid-inference/key", {"api_key": ""})
    assert not key_file.exists()


def test_no_server_answer_holds_the_key(comfyui: tuple[str, Path, Path, Stub]) -> None:
    base, _, run_dir, _ = comfyui
    status = get(f"{base}/liquid-inference/status")
    assert status["key_set"] is True
    assert status["key_source"] == "environment"
    for route in ["/liquid-inference/status", "/object_info", "/history"]:
        with urllib.request.urlopen(f"{base}{route}", timeout=10) as response:
            assert SENTINEL not in response.read().decode()
    for file in (run_dir / "user").rglob("*"):
        if file.is_file():
            assert SENTINEL not in file.read_bytes().decode("latin-1"), file

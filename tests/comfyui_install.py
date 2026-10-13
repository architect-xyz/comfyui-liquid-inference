"""Installs a pinned ComfyUI release and CPU torch in a directory.

The smoke test and a local development ComfyUI use the same installation.
Run `python tests/comfyui_install.py <directory>` to install and to print the
path of its Python.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

COMFYUI_TAG = "v0.39.0"
COMFYUI_REPOSITORY = "https://github.com/comfyanonymous/ComfyUI.git"


def run(*command: str | Path, cwd: Path | None = None) -> None:
    # stdout carries the path that `main` prints, so the tools write to stderr.
    subprocess.run([str(part) for part in command], cwd=cwd, check=True, stdout=sys.stderr)


def install(root: Path) -> Path:
    """Installs ComfyUI and its packages in `root` once. Returns the Python."""
    comfy = root / "ComfyUI"
    python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = root / f"installed-{COMFYUI_TAG}"
    if marker.exists():
        return python
    shutil.rmtree(comfy, ignore_errors=True)
    run(
        "git",
        "clone",
        "--quiet",
        "--depth",
        "1",
        "--branch",
        COMFYUI_TAG,
        COMFYUI_REPOSITORY,
        comfy,
    )
    run("uv", "venv", "--quiet", "--python", "3.12", root / "venv")
    torch = ["torch", "torchvision", "torchaudio"]
    if platform.system() == "Darwin":
        run("uv", "pip", "install", "--quiet", "--python", python, *torch)
    else:
        # The default wheels on Linux include CUDA, which is gigabytes.
        index = "https://download.pytorch.org/whl/cpu"
        run("uv", "pip", "install", "--quiet", "--python", python, "--index-url", index, *torch)
    run("uv", "pip", "install", "--quiet", "--python", python, "-r", comfy / "requirements.txt")
    marker.touch()
    return python


if __name__ == "__main__":
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    print(install(root))

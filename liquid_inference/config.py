"""The API key and the router address that the node uses.

The key is never a node input. ComfyUI writes each input value into the
workflow and into the metadata of each saved image, so a key in an input
would go to each person who receives the file.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://router.inference.ai.exchange/v1"
KEY_ENV = "LIQUID_API_KEY"
BASE_URL_ENV = "LIQUID_BASE_URL"
MAX_KEY_LENGTH = 1024


@dataclass(frozen=True)
class Setting:
    value: str
    # "environment", "settings" or "default".
    source: str


class Config:
    """Reads and stores the key and the base URL.

    The environment comes first, then the files that the Settings panel
    wrote in `directory`.
    """

    def __init__(self, directory: Path, environ: Mapping[str, str] | None = None) -> None:
        self.directory = directory
        self.environ = os.environ if environ is None else environ

    @property
    def key_file(self) -> Path:
        return self.directory / "api_key"

    @property
    def base_url_file(self) -> Path:
        return self.directory / "base_url"

    def api_key(self) -> Setting | None:
        key = self.environ.get(KEY_ENV, "").strip()
        if key:
            return Setting(key, "environment")
        key = _read(self.key_file)
        return Setting(key, "settings") if key else None

    def set_api_key(self, key: str) -> None:
        """Stores the key, or deletes the stored key when `key` is empty."""
        key = key.strip()
        if not key:
            self.key_file.unlink(missing_ok=True)
            return
        if len(key) > MAX_KEY_LENGTH or any(c.isspace() or not c.isprintable() for c in key):
            raise ValueError("the API key has characters that a key cannot have")
        self._write(self.key_file, key)

    def base_url(self) -> Setting:
        url = self.environ.get(BASE_URL_ENV, "").strip()
        if url:
            return Setting(normalize_base_url(url), "environment")
        url = _read(self.base_url_file)
        if url:
            return Setting(normalize_base_url(url), "settings")
        return Setting(DEFAULT_BASE_URL, "default")

    def set_base_url(self, url: str) -> None:
        """Stores the base URL, or deletes it when `url` is empty."""
        url = url.strip()
        if not url:
            self.base_url_file.unlink(missing_ok=True)
            return
        self._write(self.base_url_file, normalize_base_url(url))

    def status(self) -> dict[str, object]:
        """Tells where the key and the base URL come from. It never holds the key."""
        key = self.api_key()
        base_url = self.base_url()
        return {
            "key_set": key is not None,
            "key_source": key.source if key else None,
            "base_url": base_url.value,
            "base_url_source": base_url.source,
        }

    def _write(self, path: Path, text: str) -> None:
        # The file is created with mode 0600 and then moved into place. Thus no
        # other user can read the key, not even for a moment.
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".tmp-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(text)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def normalize_base_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"the base URL must start with https:// or http://, not {url!r}")
    return url.strip().rstrip("/")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


@cache
def comfy_config() -> Config:
    """The configuration in the ComfyUI user directory.

    ComfyUI does not serve a folder whose name starts with `__` over HTTP.
    Thus the stored key cannot be read through the user data API.
    """
    import folder_paths

    system_directory = getattr(folder_paths, "get_system_user_directory", None)
    if system_directory is not None:
        return Config(Path(system_directory("liquid_inference")))
    return Config(Path(folder_paths.get_user_directory()) / "__liquid_inference")

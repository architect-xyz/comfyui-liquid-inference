"""The API key and the router address that the node uses.

The key is never a node input. ComfyUI writes each input value into the
workflow and into the metadata of each saved image, so a key in an input
would go to each person who receives the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://router.inference.ai.exchange/v1"
KEY_ENV = "ARCHITECT_API_KEY"
BASE_URL_ENV = "ARCHITECT_BASE_URL"
# The names before the rename. They still work.
LEGACY_KEY_ENV = "LIQUID_API_KEY"
LEGACY_BASE_URL_ENV = "LIQUID_BASE_URL"
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

    def __init__(
        self,
        directory: Path,
        environ: Mapping[str, str] | None = None,
        *,
        legacy_directory: Path | None = None,
    ) -> None:
        self.directory = directory
        self.environ = os.environ if environ is None else environ
        self.legacy_directory = legacy_directory

    @property
    def key_file(self) -> Path:
        return self.directory / "api_key"

    @property
    def base_url_file(self) -> Path:
        return self.directory / "base_url"

    def _legacy_file(self, name: str) -> Path | None:
        return self.legacy_directory / name if self.legacy_directory is not None else None

    def _env_value(self, primary: str, legacy: str) -> tuple[str, str] | None:
        for name in (primary, legacy):
            value = (self.environ.get(name) or "").strip()
            if value:
                return value, name
        return None

    def _stored_key(self) -> tuple[str, Path] | None:
        key = _read(self.key_file)
        if key:
            return key, self.directory
        legacy = self._legacy_file("api_key")
        if legacy is not None:
            key = _read(legacy)
            if key and self.legacy_directory is not None:
                return key, self.legacy_directory
        return None

    def _unlink_legacy(self, *names: str) -> None:
        if self.legacy_directory is None:
            return
        for name in names:
            (self.legacy_directory / name).unlink(missing_ok=True)

    def api_key(self) -> Setting | None:
        env = self._env_value(KEY_ENV, LEGACY_KEY_ENV)
        if env is not None:
            return Setting(env[0], "environment")
        stored = self._stored_key()
        return Setting(stored[0], "settings") if stored is not None else None

    def set_api_key(self, key: str, *, expires_at: float | None = None) -> None:
        """Stores the key, or deletes the stored key when `key` is empty."""
        key = key.strip()
        if not key:
            self.key_file.unlink(missing_ok=True)
            (self.directory / "key_expiry.json").unlink(missing_ok=True)
            self._unlink_legacy("api_key", "key_expiry.json")
            return
        if len(key) > MAX_KEY_LENGTH or any(c.isspace() or not c.isprintable() for c in key):
            raise ValueError("the API key has characters that a key cannot have")
        self._write(self.key_file, key)
        expiry_file = self.directory / "key_expiry.json"
        if expires_at is None:
            expiry_file.unlink(missing_ok=True)
        else:
            self._write(
                expiry_file,
                json.dumps(
                    {"key_hash": hashlib.sha256(key.encode()).hexdigest(), "expires_at": expires_at}
                ),
            )
        self._unlink_legacy("api_key", "key_expiry.json")

    def key_expired(self, *, now: float | None = None) -> bool:
        if self._env_value(KEY_ENV, LEGACY_KEY_ENV) is not None:
            return False
        stored = self._stored_key()
        if stored is None:
            return False
        key_value, owner = stored
        try:
            expiry = json.loads((owner / "key_expiry.json").read_text())
            return bool(
                expiry["key_hash"] == hashlib.sha256(key_value.encode()).hexdigest()
                and (time.time() if now is None else now) >= expiry["expires_at"]
            )
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            return False

    def base_url(self) -> Setting:
        env = self._env_value(BASE_URL_ENV, LEGACY_BASE_URL_ENV)
        if env is not None:
            return Setting(normalize_base_url(env[0]), "environment")
        url = _read(self.base_url_file)
        if url:
            return Setting(normalize_base_url(url), "settings")
        legacy = self._legacy_file("base_url")
        if legacy is not None:
            url = _read(legacy)
            if url:
                return Setting(normalize_base_url(url), "settings")
        return Setting(DEFAULT_BASE_URL, "default")

    def set_base_url(self, url: str) -> None:
        """Stores the base URL, or deletes it when `url` is empty."""
        url = url.strip()
        if not url:
            self.base_url_file.unlink(missing_ok=True)
            self._unlink_legacy("base_url")
            return
        self._write(self.base_url_file, normalize_base_url(url))
        self._unlink_legacy("base_url")

    def status(self) -> dict[str, object]:
        """Tells where the key and the base URL come from. It never holds the key."""
        key_env = self._env_value(KEY_ENV, LEGACY_KEY_ENV)
        base_url_env = self._env_value(BASE_URL_ENV, LEGACY_BASE_URL_ENV)
        key = self.api_key()
        base_url = self.base_url()
        return {
            "key_set": key is not None,
            "key_expired": self.key_expired(),
            "key_source": key.source if key else None,
            "key_env": key_env[1] if key_env is not None else None,
            "base_url": base_url.value,
            "base_url_source": base_url.source,
            "base_url_env": base_url_env[1] if base_url_env is not None else None,
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
        return Config(
            Path(system_directory("architect_inference")),
            legacy_directory=Path(system_directory("liquid_inference")),
        )
    user_directory = Path(folder_paths.get_user_directory())
    return Config(
        user_directory / "__architect_inference",
        legacy_directory=user_directory / "__liquid_inference",
    )

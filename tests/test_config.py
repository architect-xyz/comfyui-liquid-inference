import json
import stat
from pathlib import Path

import pytest

from architect_inference.config import DEFAULT_BASE_URL, Config, Setting
from architect_inference.routes import status, store_base_url, store_key

KEY = "sk-architect-secret-value-42"


def test_the_environment_comes_before_the_stored_key(tmp_path: Path) -> None:
    config = Config(tmp_path, {"OPENAI_API_KEY": "sk-not-ours"})
    assert config.api_key() is None
    config.set_api_key(KEY)
    assert config.api_key() == Setting(KEY, "settings")
    config.environ = {"LIQUID_API_KEY": "from-legacy-env"}
    assert config.api_key() == Setting("from-legacy-env", "environment")
    assert config.status()["key_env"] == "LIQUID_API_KEY"
    config.environ = {"ARCHITECT_API_KEY": "from-env", "LIQUID_API_KEY": "from-legacy-env"}
    assert config.api_key() == Setting("from-env", "environment")
    assert config.status()["key_env"] == "ARCHITECT_API_KEY"


def test_the_key_file_is_private(tmp_path: Path) -> None:
    config = Config(tmp_path / "architect", {})
    config.set_api_key(KEY)
    assert stat.S_IMODE(config.key_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.directory.stat().st_mode) == 0o700
    config.set_api_key("")
    assert not config.key_file.exists()
    assert config.api_key() is None


def test_a_key_with_whitespace_inside_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Config(tmp_path, {}).set_api_key("two words")


def test_the_base_url_comes_from_the_environment_then_the_file_then_the_default(
    tmp_path: Path,
) -> None:
    config = Config(tmp_path, {})
    assert config.base_url() == Setting(DEFAULT_BASE_URL, "default")
    config.set_base_url("http://localhost:8080/v1/")
    assert config.base_url() == Setting("http://localhost:8080/v1", "settings")
    config.environ = {"LIQUID_BASE_URL": "https://legacy.test/v1"}
    assert config.base_url() == Setting("https://legacy.test/v1", "environment")
    assert config.status()["base_url_env"] == "LIQUID_BASE_URL"
    config.environ = {
        "ARCHITECT_BASE_URL": "https://example.test/v1",
        "LIQUID_BASE_URL": "https://legacy.test/v1",
    }
    assert config.base_url() == Setting("https://example.test/v1", "environment")
    assert config.status()["base_url_env"] == "ARCHITECT_BASE_URL"
    with pytest.raises(ValueError):
        config.set_base_url("file:///etc/passwd")


def test_the_status_never_holds_the_key(tmp_path: Path) -> None:
    config = Config(tmp_path, {})
    config.set_api_key(KEY)
    code, payload = status(config)
    assert code == 200
    assert payload["key_set"] is True
    assert payload["key_source"] == "settings"
    assert payload["key_env"] is None
    assert KEY not in json.dumps(payload)


def test_the_key_route_takes_json_only(tmp_path: Path) -> None:
    config = Config(tmp_path, {})
    body = json.dumps({"api_key": KEY}).encode()
    assert store_key(config, "application/x-www-form-urlencoded", body)[0] == 415
    assert config.api_key() is None
    assert store_key(config, "application/json", b"[]")[0] == 400

    code, payload = store_key(config, "application/json", body)
    assert code == 200
    assert payload["key_set"] is True
    assert KEY not in json.dumps(payload)
    assert config.api_key() == Setting(KEY, "settings")

    store_key(config, "application/json", json.dumps({"api_key": ""}).encode())
    assert config.api_key() is None


def test_the_base_url_route_refuses_a_bad_url(tmp_path: Path) -> None:
    config = Config(tmp_path, {})
    code, payload = store_base_url(config, "application/json", b'{"base_url": "ftp://x"}')
    assert code == 400
    assert "https://" in str(payload["error"])


def test_an_oauth_expiry_belongs_only_to_its_saved_key(tmp_path: Path) -> None:
    config = Config(tmp_path, {})
    config.set_api_key(KEY, expires_at=100)
    assert config.key_expired(now=101)
    assert not config.key_expired(now=99)
    config.environ = {"ARCHITECT_API_KEY": KEY}
    assert not config.key_expired(now=101)
    config.environ = {}
    config.set_api_key("manual-key")
    assert not config.key_expired(now=101)


def test_legacy_directory_settings_are_read_and_cleared(tmp_path: Path) -> None:
    legacy = tmp_path / "__liquid_inference"
    current = tmp_path / "__architect_inference"
    old_config = Config(legacy, {})
    old_config.set_api_key(KEY, expires_at=100)
    old_config.set_base_url("https://legacy.test/v1")

    config = Config(current, {}, legacy_directory=legacy)
    assert config.api_key() == Setting(KEY, "settings")
    assert config.key_expired(now=101)
    assert not config.key_expired(now=99)
    assert config.base_url() == Setting("https://legacy.test/v1", "settings")

    config.set_api_key("sk-architect-new")
    assert config.api_key() == Setting("sk-architect-new", "settings")
    assert not config.key_expired(now=101)
    assert not (legacy / "api_key").exists()
    assert not (legacy / "key_expiry.json").exists()

    config.set_base_url("")
    assert config.base_url() == Setting(DEFAULT_BASE_URL, "default")
    assert not (legacy / "base_url").exists()

    old_config.set_api_key("sk-liquid-old")
    config.set_api_key("")
    assert config.api_key() is None
    assert not (legacy / "api_key").exists()

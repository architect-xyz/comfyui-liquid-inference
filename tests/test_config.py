import json
import stat
from pathlib import Path

import pytest

from liquid_inference.config import DEFAULT_BASE_URL, Config, Setting
from liquid_inference.routes import status, store_base_url, store_key

KEY = "lq_secret_value_42"


def test_the_environment_comes_before_the_stored_key(tmp_path: Path) -> None:
    config = Config(tmp_path, {"OPENAI_API_KEY": "sk-not-ours"})
    assert config.api_key() is None
    config.set_api_key(KEY)
    assert config.api_key() == Setting(KEY, "settings")
    config.environ = {"LIQUID_API_KEY": "from-env"}
    assert config.api_key() == Setting("from-env", "environment")


def test_the_key_file_is_private(tmp_path: Path) -> None:
    config = Config(tmp_path / "liquid", {})
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
    config.environ = {"LIQUID_BASE_URL": "https://example.test/v1"}
    assert config.base_url() == Setting("https://example.test/v1", "environment")
    with pytest.raises(ValueError):
        config.set_base_url("file:///etc/passwd")


def test_the_status_never_holds_the_key(tmp_path: Path) -> None:
    config = Config(tmp_path, {})
    config.set_api_key(KEY)
    code, payload = status(config)
    assert code == 200
    assert payload["key_set"] is True
    assert payload["key_source"] == "settings"
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

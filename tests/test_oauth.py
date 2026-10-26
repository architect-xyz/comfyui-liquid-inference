import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from liquid_inference.config import Config
from liquid_inference.oauth import AUTHORITY, CLIENT_ID, SignIn


def flow(tmp_path: Path) -> tuple[SignIn, Config, list[dict[str, str]], list[float]]:
    config = Config(tmp_path, {})
    calls: list[dict[str, str]] = []
    now = [100.0]

    def exchange(fields: dict[str, str]) -> dict[str, object]:
        calls.append(fields)
        return {
            "access_token": "sk-liquid-test",
            "token_type": "Bearer",
            "expires_in": 604800,
            "scope": "mcp:read mcp:spend",
        }

    return SignIn(config, exchange=exchange, clock=lambda: now[0]), config, calls, now


def start(sign_in: SignIn, origin: str = "http://127.0.0.1:8188") -> dict[str, str]:
    return {
        k: v[0] for k, v in parse_qs(urlsplit(sign_in.start(origin, "127.0.0.1")).query).items()
    }


def test_sign_in_uses_pkce_and_saves_the_key_only_on_the_server(tmp_path: Path) -> None:
    sign_in, config, calls, _ = flow(tmp_path)
    fields = start(sign_in)
    assert fields["client_id"] == CLIENT_ID
    assert fields["scope"] == "mcp:read mcp:spend"
    assert fields["resource"] == "https://router.inference.ai.exchange"
    assert fields["code_challenge_method"] == "S256"
    assert not config.key_file.exists()
    sign_in.finish(
        {"state": fields["state"], "iss": AUTHORITY, "code": "code"}, "http://127.0.0.1:8188"
    )
    key = config.api_key()
    assert key is not None
    assert key.value == "sk-liquid-test"
    assert calls[0]["redirect_uri"] == "http://127.0.0.1:8188/liquid-inference/callback"
    import base64

    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(calls[0]["code_verifier"].encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert fields["code_challenge"] == challenge
    assert "sk-liquid-test" not in json.dumps(sign_in.status())
    with pytest.raises(ValueError):
        sign_in.finish(
            {"state": fields["state"], "iss": AUTHORITY, "code": "code"}, "http://127.0.0.1:8188"
        )


@pytest.mark.parametrize(
    "origin,peer",
    [
        ("http://remote.example:8188", "127.0.0.1"),
        ("http://127.0.0.1:8188", "192.168.1.2"),
        ("http://127.0.0.1.evil.test", "127.0.0.1"),
        ("http://user@localhost:8188", "127.0.0.1"),
    ],
)
def test_remote_sign_in_is_refused(tmp_path: Path, origin: str, peer: str) -> None:
    sign_in, _, _, _ = flow(tmp_path)
    with pytest.raises(ValueError):
        sign_in.start(origin, peer)


@pytest.mark.parametrize("change", [{"state": "wrong"}, {"iss": "https://evil.test"}])
def test_callback_must_match_the_state_and_issuer(tmp_path: Path, change: dict[str, str]) -> None:
    sign_in, config, calls, _ = flow(tmp_path)
    fields = start(sign_in)
    with pytest.raises(ValueError):
        sign_in.finish(
            {"state": fields["state"], "iss": AUTHORITY, "code": "code", **change},
            "http://127.0.0.1:8188",
        )
    assert not calls
    assert not config.key_file.exists()


def test_expired_flow_does_not_exchange_a_code(tmp_path: Path) -> None:
    sign_in, _, calls, now = flow(tmp_path)
    fields = start(sign_in)
    now[0] += 601
    with pytest.raises(ValueError):
        sign_in.finish(
            {"state": fields["state"], "iss": AUTHORITY, "code": "code"}, "http://127.0.0.1:8188"
        )
    assert not calls


def test_denial_and_wrong_callback_origin_do_not_save_a_key(tmp_path: Path) -> None:
    sign_in, config, calls, _ = flow(tmp_path)
    fields = start(sign_in)
    with pytest.raises(ValueError):
        sign_in.finish(
            {"state": fields["state"], "iss": AUTHORITY, "code": "code"}, "http://localhost:8188"
        )
    with pytest.raises(ValueError, match="denied"):
        sign_in.finish(
            {"state": fields["state"], "iss": AUTHORITY, "error": "access_denied"},
            "http://127.0.0.1:8188",
        )
    assert not calls
    assert not config.key_file.exists()


def test_environment_key_and_custom_router_require_manual_configuration(tmp_path: Path) -> None:
    for env in [
        {"LIQUID_API_KEY": "environment-key"},
        {"LIQUID_BASE_URL": "http://127.0.0.1:8080/v1"},
    ]:
        with pytest.raises(ValueError):
            SignIn(Config(tmp_path, env)).start("http://localhost:8188", "127.0.0.1")


def test_sign_in_start_refuses_cross_site_and_form_requests(tmp_path: Path) -> None:
    from liquid_inference.routes import start_sign_in

    sign_in, _, _, _ = flow(tmp_path)
    origin = "http://127.0.0.1:8188"
    assert start_sign_in(sign_in, "text/plain", origin, origin, "127.0.0.1")[0] == 415
    assert (
        start_sign_in(sign_in, "application/json", "https://evil.test", origin, "127.0.0.1")[0]
        == 403
    )
    code, answer = start_sign_in(sign_in, "application/json", origin, origin, "127.0.0.1")
    assert code == 200
    assert "access_token" not in answer


@pytest.mark.parametrize(
    "token",
    [
        {"token_type": 3},
        {"expires_in": True},
        {"scope": "mcp:read"},
        {"access_token": "bad key"},
    ],
)
def test_an_invalid_token_answer_does_not_save_a_key(
    tmp_path: Path, token: dict[str, object]
) -> None:
    sign_in, config, _, _ = flow(tmp_path)
    sign_in._exchange = lambda fields: {
        "access_token": "key",
        "token_type": "Bearer",
        "expires_in": 600,
        "scope": "mcp:spend",
        **token,
    }
    fields = start(sign_in)
    with pytest.raises(ValueError):
        sign_in.finish(
            {"state": fields["state"], "iss": AUTHORITY, "code": "code"}, "http://127.0.0.1:8188"
        )
    assert not config.key_file.exists()


def test_a_truncated_token_answer_allows_a_new_sign_in(tmp_path: Path) -> None:
    from http.client import IncompleteRead

    sign_in, config, _, now = flow(tmp_path)

    def truncated(fields: dict[str, str]) -> dict[str, object]:
        raise IncompleteRead(b"partial", 100)

    sign_in = SignIn(config, exchange=truncated, clock=lambda: now[0])
    fields = start(sign_in)
    with pytest.raises(ValueError):
        sign_in.finish(
            {"state": fields["state"], "iss": AUTHORITY, "code": "code"}, "http://127.0.0.1:8188"
        )
    assert sign_in.status()["sign_in"] == "failed"
    assert start(sign_in)["state"] != fields["state"]

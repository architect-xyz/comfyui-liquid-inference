"""Local sign-in with one pending PKCE request, valid for ten minutes."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any
from urllib.parse import urlencode, urlsplit

from .client import USER_AGENT, _router
from .config import DEFAULT_BASE_URL, Config

AUTHORITY = "https://gateway.inference.ai.exchange"
CLIENT_ID = "https://inference.ai.exchange/oauth/comfyui.json"
RESOURCE = "https://router.inference.ai.exchange"
CALLBACK = "/liquid-inference/callback"
TTL = 600


def exchange_code(fields: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{AUTHORITY}/oauth/token",
        data=urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with _router.open(request, timeout=20) as response:
            result = json.loads(response.read(65536))
    except (OSError, ValueError, HTTPException) as error:
        raise ValueError("Sign-in failed. Select Sign in again.") from error
    if not isinstance(result, dict):
        raise ValueError("Sign-in returned an invalid answer. Select Sign in again.")
    return result


@dataclass(frozen=True)
class Pending:
    state: str
    verifier: str
    origin: str
    expires_at: float


class SignIn:
    def __init__(
        self,
        config: Config,
        *,
        exchange: Callable[[dict[str, str]], dict[str, Any]] = exchange_code,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._exchange = exchange
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: Pending | None = None
        self._result = "idle"
        self._error = ""

    def start(self, origin: str, peer: str) -> str:
        parts = urlsplit(origin)
        try:
            local_peer = ipaddress.ip_address(peer).is_loopback
            _ = parts.port
        except ValueError:
            local_peer = False
        if (
            not local_peer
            or parts.scheme != "http"
            or parts.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parts.username is not None
            or parts.password is not None
            or parts.path
            or parts.query
            or parts.fragment
        ):
            raise ValueError(
                "Sign in needs ComfyUI on this machine. For a remote server, paste an API key."
            )
        key = self.config.api_key()
        if key and key.source == "environment":
            raise ValueError("LIQUID_API_KEY has priority. Change that variable to change the key.")
        if self.config.base_url().value != DEFAULT_BASE_URL:
            raise ValueError(
                "Sign in uses the default Liquid router. For a custom router, paste an API key."
            )
        pending = Pending(
            secrets.token_urlsafe(32), secrets.token_urlsafe(64), origin, self._clock() + TTL
        )
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(pending.verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        with self._lock:
            if self._pending is None and self._result == "pending":
                raise ValueError("A sign-in code is in progress. Wait for its result.")
            self._pending = pending
            self._result = "pending"
            self._error = ""
        fields = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": origin + CALLBACK,
            "scope": "mcp:read mcp:spend",
            "resource": RESOURCE,
            "state": pending.state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return f"{AUTHORITY}/oauth/authorize?{urlencode(fields)}"

    def finish(self, query: Mapping[str, str], origin: str) -> None:
        with self._lock:
            pending = self._pending
            if (
                pending is None
                or self._clock() >= pending.expires_at
                or not secrets.compare_digest(
                    query.get("state", "").encode(), pending.state.encode()
                )
                or query.get("iss") != AUTHORITY
                or origin != pending.origin
            ):
                raise ValueError("The sign-in callback does not match. Select Sign in again.")
            self._pending = None
            self._result = "pending"
        try:
            if query.get("error"):
                raise ValueError("Sign-in was denied. Select Sign in to try again.")
            code = query.get("code")
            if not code:
                raise ValueError("The sign-in code is missing. Select Sign in again.")
            token = self._exchange(
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": CLIENT_ID,
                    "redirect_uri": pending.origin + CALLBACK,
                    "code_verifier": pending.verifier,
                    "resource": RESOURCE,
                }
            )
            key = token.get("access_token")
            expiry = token.get("expires_in")
            scope = token.get("scope", "")
            token_type = token.get("token_type")
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(token_type, str)
                or token_type.lower() != "bearer"
                or type(expiry) is not int
                or not 0 < expiry <= 7 * 24 * 3600
                or not isinstance(scope, str)
                or "mcp:spend" not in scope.split()
            ):
                raise ValueError(
                    "The sign-in key cannot buy images or videos. Sign in again with spend access."
                )
            current_key = self.config.api_key()
            if self.config.base_url().value != DEFAULT_BASE_URL or (
                current_key and current_key.source == "environment"
            ):
                raise ValueError("The account settings changed. Select Sign in again.")
            self.config.set_api_key(key, expires_at=self._clock() + expiry)
        except Exception as error:
            with self._lock:
                self._result = "failed"
                self._error = "Sign-in failed. Select Sign in to try again."
            if isinstance(error, ValueError):
                raise
            raise ValueError("Sign-in failed. Select Sign in again.") from error
        with self._lock:
            self._result = "complete"

    def status(self) -> dict[str, object]:
        with self._lock:
            if self._pending is not None and self._clock() >= self._pending.expires_at:
                self._pending = None
                self._result = "failed"
                self._error = "Sign-in expired. Select Sign in again."
            return {"sign_in": self._result, "sign_in_error": self._error}

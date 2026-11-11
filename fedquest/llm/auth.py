# ALCF access tokens through Argonne's auth script; API keys for openai endpoints
from __future__ import annotations

import importlib.util
import logging
import os
import platform
import threading
import time
from pathlib import Path
from types import ModuleType

import requests

from fedquest.errors import AuthExpired
from fedquest.paths import REPO

log = logging.getLogger(__name__)

AUTH_SCRIPT_URL = (
    "https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py"
)
AUTH_SCRIPT_ENV = "FEDQUEST_ALCF_AUTH_SCRIPT"
PROVIDERS = ("alcf", "openai", "fake", "replay")
GLOBUS_LOGOUT = "https://app.globus.org/logout"

_DEFAULT_SCRIPT = REPO / "third_party" / "inference_auth_token.py"


def auth_script_path() -> Path:
    env = os.environ.get(AUTH_SCRIPT_ENV)
    return Path(env).expanduser() if env else _DEFAULT_SCRIPT


def relogin(provider: str = "alcf", key_env: str | None = None) -> str:
    rerun = "then rerun the same command (runs resume and redo only the unfinished or failed questions)"
    here = f"On this machine ({platform.node() or 'this host'})"
    if provider == "alcf":
        script = auth_script_path()
        shown = script.relative_to(REPO) if script.is_relative_to(REPO) else script
        return (f"{here}, in the clone with the venv active: log out at {GLOBUS_LOGOUT}, run `python {shown} "
                f"authenticate --force`, {rerun}.")
    return f"{here}: set a valid key in {key_env or 'the server configuration'}, {rerun}."


NO_KEY = "none"


class TokenProvider:
    def __init__(self, provider: str, refresh_s: int = 900, key_env: str | None = "OPENAI_API_KEY") -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown LLM provider {provider!r}; expected one of {PROVIDERS}")
        self.provider = provider
        self.refresh_s = refresh_s
        self.key_env = key_env
        self._lock = threading.Lock()
        self._token = ""
        self._fetched = 0.0
        self._script: ModuleType | None = None

    def get(self) -> str:
        if self.provider in ("fake", "replay"):
            return self.provider
        if self.provider == "openai":
            return self._openai_key()
        return os.environ.get("ALCF_ACCESS_TOKEN") or self._alcf_token()

    def invalidate(self) -> None:
        self._token = ""

    def _openai_key(self) -> str:
        if self.key_env is None:
            return NO_KEY
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(f"{self.key_env} is not set")
        return key

    def _alcf_token(self) -> str:
        with self._lock:
            if self._token and time.monotonic() - self._fetched < self.refresh_s:
                return self._token
            self._token = self._fetch()
            self._fetched = time.monotonic()
            return self._token

    def _fetch(self) -> str:
        script = self._load_script()
        tokens_file = getattr(script, "TOKENS_PATH", None)
        if tokens_file and not os.path.isfile(tokens_file):
            raise AuthExpired(f"ALCF login required: this machine has none ({tokens_file} is missing). {relogin()}")
        try:
            return str(script.get_access_token())
        except Exception as e:
            raise AuthExpired(f"ALCF login required: the auth script could not refresh this machine's token "
                              f"({type(e).__name__}: {e}). {relogin()}") from e

    def _load_script(self) -> ModuleType:
        if self._script is None:
            path = auth_script_path()
            if not path.is_file():
                raise RuntimeError(f"ALCF auth script not found at {path}: run `fedquest llm auth`")
            spec = importlib.util.spec_from_file_location("fedquest_inference_auth_token", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load the ALCF auth script at {path}")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                raise RuntimeError(f"cannot load the ALCF auth script at {path} ({type(e).__name__}: {e})") from e
            self._script = module
        return self._script


def download_auth_script(dest: Path | None = None) -> Path:
    dest = Path(dest) if dest else auth_script_path()
    response = requests.get(AUTH_SCRIPT_URL, timeout=30)
    response.raise_for_status()
    if b"get_access_token" not in response.content:
        raise RuntimeError(f"{AUTH_SCRIPT_URL} did not return the ALCF auth script")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(response.content)
    part.replace(dest)
    log.info("downloaded the ALCF auth script to %s", dest)
    return dest

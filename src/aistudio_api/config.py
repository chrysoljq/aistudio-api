"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

DEFAULT_TEXT_MODEL = os.getenv("AISTUDIO_DEFAULT_TEXT_MODEL", "gemma-4-31b-it")
DEFAULT_IMAGE_MODEL = os.getenv("AISTUDIO_DEFAULT_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
DEFAULT_BROWSER_PORT = 9222


def _load_browser_engine() -> str:
    """Load browser engine name.

    Supported values:
    - chromium: stealth Chromium via cloakbrowser (default)
    - camoufox: Camoufox Firefox-based backend
    """
    value = (os.getenv("AISTUDIO_BROWSER", "chromium") or "chromium").strip().lower()
    if value not in {"camoufox", "chromium"}:
        return "chromium"
    return value


def _load_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return None


def _load_bool_env(*names: str, default: bool) -> bool:
    value = _load_env(*names)
    if value is None:
        return default
    return value not in ("0", "false", "False")


def _load_int_env(*names: str, default: int) -> int:
    value = _load_env(*names)
    if value is None:
        return default
    return int(value)


def _parse_api_keys(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()

    keys: list[str] = []
    for line in raw.splitlines():
        for part in line.split(","):
            key = part.strip()
            if key and key not in keys:
                keys.append(key)
    return tuple(keys)


def _load_api_keys() -> frozenset[str]:
    values: list[str] = []
    for name in ("AISTUDIO_API_KEY", "AISTUDIO_API_KEYS"):
        for key in _parse_api_keys(os.getenv(name)):
            if key not in values:
                values.append(key)
    return frozenset(values)


def _default_chromium_sandbox() -> bool:
    # On many recent Linux distros (including Ubuntu with AppArmor userns
    # restrictions), Chromium sandboxing is unavailable unless the host has been
    # explicitly configured for it. Default to disabled there so the browser can
    # actually start; keep enabled elsewhere.
    return os.name != "posix" or os.uname().sysname != "Linux"

_AUTH_SEARCH_ROOTS = [
    Path(__file__).resolve().parents[2] / "data",  # 项目内 data/ 目录
]


def discover_auth_file() -> str | None:
    override = os.getenv("AISTUDIO_AUTH_FILE")
    if override:
        return override

    for root in _AUTH_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        # 优先从 registry.json 读取活跃账号
        registry_path = root / "accounts" / "registry.json"
        if registry_path.exists():
            try:
                import json
                registry = json.loads(registry_path.read_text())
                active_id = registry.get("active_account_id")
                if active_id:
                    auth_path = root / "accounts" / active_id / "auth.json"
                    if auth_path.exists():
                        return str(auth_path)
            except (json.JSONDecodeError, KeyError):
                pass
        # 回退：扫描 data/ 目录下的 .json 文件
        for file in root.iterdir():
            if file.suffix == ".json":
                return str(file)
    return None


def discover_proxy_url() -> str | None:
    return (
        os.getenv("AISTUDIO_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )


def build_browser_proxy(proxy_url: str | None) -> dict[str, str] | None:
    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        return None

    proxy: dict[str, str] = {
        "server": f"{parsed.scheme}://{parsed.hostname}",
    }
    if parsed.port:
        proxy["server"] += f":{parsed.port}"
    if parsed.username:
        proxy["username"] = parsed.username
    if parsed.password:
        proxy["password"] = parsed.password
    return proxy


def build_camoufox_proxy(proxy_url: str | None) -> dict[str, str] | None:
    """Backward-compatible alias for old imports."""
    return build_browser_proxy(proxy_url)


@dataclass(slots=True)
class Settings:
    port: int = int(os.getenv("AISTUDIO_PORT", "8080"))
    browser_engine: str = _load_browser_engine()
    browser_port: int = _load_int_env("AISTUDIO_BROWSER_PORT", "AISTUDIO_CAMOUFOX_PORT", default=DEFAULT_BROWSER_PORT)
    browser_headless: bool = _load_bool_env("AISTUDIO_BROWSER_HEADLESS", "AISTUDIO_CAMOUFOX_HEADLESS", default=True)
    browser_channel: str | None = os.getenv("AISTUDIO_BROWSER_CHANNEL")
    browser_executable_path: str | None = os.getenv("AISTUDIO_BROWSER_EXECUTABLE")
    browser_chromium_sandbox: bool = _load_bool_env(
        "AISTUDIO_CHROMIUM_SANDBOX",
        default=_default_chromium_sandbox(),
    )
    browser_python: str | None = _load_env("AISTUDIO_BROWSER_PYTHON", "AISTUDIO_CAMOUFOX_PYTHON")
    login_browser_port: int = _load_int_env("AISTUDIO_LOGIN_BROWSER_PORT", "AISTUDIO_LOGIN_CAMOUFOX_PORT", default=9223)
    auth_file: str | None = discover_auth_file()
    tmp_dir: str = os.getenv("AISTUDIO_TMP_DIR", "/tmp")
    proxy_url: str | None = discover_proxy_url()
    api_keys: frozenset[str] = _load_api_keys()
    timeout_replay: int = int(os.getenv("AISTUDIO_TIMEOUT_REPLAY", "120"))
    timeout_stream: int = int(os.getenv("AISTUDIO_TIMEOUT_STREAM", "120"))
    timeout_capture: int = int(os.getenv("AISTUDIO_TIMEOUT_CAPTURE", "30"))
    dump_raw_response: bool = os.getenv("AISTUDIO_DUMP_RAW_RESPONSE", "0") in ("1", "true", "True")
    dump_raw_response_dir: str = os.getenv("AISTUDIO_DUMP_RAW_RESPONSE_DIR", "/tmp")
    accounts_dir: str = os.getenv("AISTUDIO_ACCOUNTS_DIR", "")
    # 账号轮询配置
    account_rotation_mode: str = os.getenv("AISTUDIO_ACCOUNT_ROTATION_MODE", "round_robin")  # round_robin, lru, least_rl
    account_cooldown_seconds: int = int(os.getenv("AISTUDIO_ACCOUNT_COOLDOWN_SECONDS", "60"))
    account_max_retries: int = int(os.getenv("AISTUDIO_ACCOUNT_MAX_RETRIES", "3"))
    max_concurrency: int = int(os.getenv("AISTUDIO_MAX_CONCURRENCY", "3"))

    @property
    def camoufox_port(self) -> int:
        return self.browser_port

    @property
    def camoufox_headless(self) -> bool:
        return self.browser_headless

    @property
    def camoufox_python(self) -> str | None:
        return self.browser_python

    @property
    def login_camoufox_port(self) -> int:
        return self.login_browser_port

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


settings = Settings()

"""Shared browser session management for gateway operations."""

from __future__ import annotations

import asyncio

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any

from aistudio_api.config import settings
from aistudio_api.infrastructure.account.account_store import AccountStore
from aistudio_api.infrastructure.browser.browser_engine import (
    build_browser_context_options,
    describe_browser_backend,
    is_camoufox_engine,
    sync_launch_browser,
    sync_launch_persistent_context,
    sync_maximize_page_window,
)
from aistudio_api.infrastructure.gateway.wire_types import AistudioContent

log = logging.getLogger("aistudio.session")

AI_STUDIO_URL = "https://aistudio.google.com/prompts/new_chat?model=gemma-4-31b-it"
AI_STUDIO_URL_FALLBACK = "https://aistudio.google.com/app/prompts/new_chat"
GOOGLE_LOGIN_BOOTSTRAP_URL = (
    "https://accounts.google.com/ServiceLogin?continue=https://aistudio.google.com"
)
INSTALL_HOOKS_JS = r"""
mw:((() => {
    // Verify hooks are actually present on XHR prototype, not just a stale flag
    const xhrHookAlive = XMLHttpRequest.prototype.open.__api_hooked === true;
    const fetchHookAlive = window.fetch.__api_hooked === true;
    if (window.__bg_hooked && xhrHookAlive && fetchHookAlive) return 'already_hooked';
    // Reset stale flag if hooks are missing
    if (window.__bg_hooked && (!xhrHookAlive || !fetchHookAlive)) window.__bg_hooked = false;

    const dms = window.default_MakerSuite;
    if (!dms) return 'no_default_MakerSuite';

    // Auto-detect snapshot function via feature matching
    let snapKey = null;
    for (const k of Object.keys(dms)) {
        try {
            if (typeof dms[k] !== 'function') continue;
            const src = dms[k].toString();
            if (src.includes('.snapshot({') && src.includes('content') && src.includes('yield')) {
                snapKey = k;
                break;
            }
        } catch(e) {}
    }
    if (!snapKey) return 'no_snapshot_fn';

    // Hook snapshot function to capture service (only if not already hooked)
    if (!dms[snapKey].__api_hooked) {
        const origSnap = dms[snapKey];
        dms[snapKey] = function(...args) {
            window.__bg_service = args[0];
            const result = origSnap.apply(this, args);
            if (result instanceof Promise) return result.then(s => { window.__bg_snapshot = s; return s; });
            window.__bg_snapshot = result;
            return result;
        };
        dms[snapKey].__api_hooked = true;
    }

    // XHR hook for body replacement (always re-install if missing)
    const origOpen = XMLHttpRequest.prototype.open;
    const origSend = XMLHttpRequest.prototype.send;
    const hookedOpen = function(method, url, ...args) {
        this.__url = url;
        this.__is_gen = url.includes('GenerateContent') && !url.includes('CountTokens');
        window.__last_hook_url = url;
        return origOpen.call(this, method, url, ...args);
    };
    hookedOpen.__api_hooked = true;
    XMLHttpRequest.prototype.open = hookedOpen;
    XMLHttpRequest.prototype.send = function(body) {
        if (this.__is_gen && window.__pending_body) {
            const captured = window.__pending_body;
            window.__pending_body = null;
            window.__hooked = true;
            window.__last_hook_url = this.__url || '';
            return origSend.call(this, captured);
        }
        return origSend.call(this, body);
    };

    // fetch hook for body replacement (streaming uses fetch)
    const origFetch = window.fetch;
    const hookedFetch = function(input, init) {
        let url = typeof input === 'string' ? input : (input instanceof Request ? input.url : String(input));
        if (url.includes('GenerateContent') && !url.includes('CountTokens') && window.__pending_body) {
            const captured = window.__pending_body;
            window.__pending_body = null;
            window.__hooked = true;
            window.__last_hook_url = url;
            if (init) {
                init.body = captured;
            } else {
                init = { body: captured };
            }
            return origFetch.call(this, input, init);
        }
        return origFetch.call(this, input, init);
    };
    hookedFetch.__api_hooked = true;
    window.fetch = hookedFetch;

    window.__bg_hooked = true;
    window.__snap_key = snapKey;
    return 'hooked:' + snapKey;
})())
"""

DIALOG_CLEANUP_JS = """(() => {
    document.querySelectorAll('button').forEach((button) => {
        const text = (button.textContent || '').trim().toLowerCase();
        if (['dismiss', 'close', 'accept', 'ok', 'agree', 'got it'].includes(text)) {
            button.click();
        }
    });
    document.querySelectorAll('.cdk-overlay-backdrop').forEach((node) => node.remove());
    document.querySelectorAll('.cdk-overlay-container').forEach((node) => node.remove());
})()"""

BOTGUARD_BOOTSTRAP_PROMPT = "say '1'"
TEMPLATE_CAPTURE_PROMPT = "say 't'"


def _snapshot_content_hash(contents: list[AistudioContent]) -> str:
    hash_parts: list[str] = []
    for content in contents:
        for part in content.parts:
            if part.inline_data:
                hash_parts.append(part.inline_data[1])  # base64 data
            if part.text:
                hash_parts.append(str(part.text))
            if part.thought_signature:
                hash_parts.append(str(part.thought_signature))
    return sha256(" ".join(hash_parts).encode("utf-8")).hexdigest()


def _is_aistudio_url(url: str) -> bool:
    return "aistudio.google.com" in (url or "")


def _is_google_signin_url(url: str) -> bool:
    return "accounts.google.com" in (url or "") and "signin" in (url or "")


def _clear_chromium_profile_locks(profile_path: Path) -> None:
    for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        lock_path = profile_path / name
        try:
            lock_path.unlink(missing_ok=True)
        except Exception as exc:
            log.debug("Failed to remove Chromium profile lock %s: %s", lock_path, exc)


class BrowserSession:
    def __init__(self, port: int):
        self.port = port
        self._auth_file = settings.auth_file or self._discover_active_auth_file()
        self._profile_dir = self._derive_profile_dir(self._auth_file)
        self._hook_page = None
        self._ctx = None
        self._browser = None
        self._cf = None
        self._playwright = None
        self._snap_key: str | None = None
        self._templates: dict[str, dict[str, Any]] = {}
        self._bootstrap_template: dict[str, Any] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aistudio-browser")

    async def ensure_context(self):
        return await self._run_sync(self._ensure_browser_sync)

    async def switch_auth(self, auth_file: str | None) -> None:
        await self._run_sync(self._switch_auth_sync, auth_file)

    async def ensure_hook_page(self):
        await self._run_sync(self._ensure_hook_page_sync)
        return True

    async def ensure_botguard_service(self):
        await self._run_sync(self._ensure_botguard_service_sync)
        return True

    async def import_cookies(self, cookie_string: str, auth_file: str | None = None) -> int:
        """注入 cookie 字符串到浏览器，访问页面获取完整 cookie，再导出保存。"""
        def _sync():
            from aistudio_api.infrastructure.account.cookie_refresher import load_cookies_from_string

            pw_cookies = load_cookies_from_string(cookie_string)
            target_auth_file = auth_file or self._auth_file
            original_auth_file = self._auth_file
            original_profile_dir = self._profile_dir
            had_live_context = (
                self._ctx is not None
                and self._hook_page is not None
                and not self._hook_page.is_closed()
            )

            if not target_auth_file:
                return len(pw_cookies)

            # Seed target auth first so a fresh persistent profile can bootstrap
            # from the refreshed cookie jar on first launch.
            self._save_cookies_sync(auth_file=target_auth_file, cookies=pw_cookies)

            switched_target = False
            target_ctx = self._ctx
            if (
                self._ctx is None
                or not original_auth_file
                or Path(target_auth_file).resolve() != Path(original_auth_file).resolve()
            ):
                self._switch_auth_sync(target_auth_file)
                switched_target = True
                target_ctx = self._ensure_browser_sync()
            elif self._ctx is None:
                target_ctx = self._ensure_browser_sync()

            if not target_ctx:
                return len(pw_cookies)

            target_ctx.add_cookies(pw_cookies)

            # 先走 Google 登录页，让浏览器补全 host-only / session cookies
            try:
                page = target_ctx.pages[0] if target_ctx.pages else target_ctx.new_page()
                self._bootstrap_google_session_sync(page)
            except Exception as e:
                log.warning("[import_cookies] browser visit failed: %s", e)

            # 从浏览器导出全部 cookies
            try:
                browser_cookies = target_ctx.cookies()
                if browser_cookies:
                    self._save_cookies_sync(auth_file=target_auth_file, cookies=browser_cookies)
                    log.info("[import_cookies] exported %d cookies from browser", len(browser_cookies))
                else:
                    # fallback: 用 curl_cffi 的结果
                    self._save_cookies_sync(auth_file=target_auth_file, cookies=pw_cookies)
                return len(browser_cookies or pw_cookies)
            finally:
                if switched_target and original_auth_file:
                    self._switch_auth_sync(original_auth_file)
                    self._profile_dir = original_profile_dir
                    if had_live_context:
                        try:
                            self._ensure_browser_sync()
                        except Exception as restore_exc:
                            log.warning("[import_cookies] failed to restore original browser context: %s", restore_exc)

        return await self._run_sync(_sync)

    async def capture_template(self, model: str) -> dict[str, Any]:
        return await self._run_sync(self._capture_template_sync, model)

    async def upload_images(self, image_paths: list[str]) -> list[str]:
        return await self._run_sync(self._upload_images_sync, image_paths)

    async def generate_snapshot(self, contents: list[AistudioContent]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self._generate_snapshot_sync(contents))

    async def send_hooked_request(
        self,
        *,
        body: str,
        timeout_ms: int,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        return await self._run_sync(self._send_hooked_request_sync, body, timeout_ms, url, headers)

    async def send_streaming_request(
        self,
        *,
        body: str,
        timeout_ms: int,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        """Send a streaming request, yielding ("status", int) and ("chunk", bytes) events."""
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()

        def _stream_worker():
            try:
                log.debug("[stream] worker started")
                self._send_streaming_request_sync(body, timeout_ms, queue, loop, cancel_event, url, headers)
                log.debug("[stream] worker finished")
            except Exception as e:
                log.debug(f"[stream] worker exception: {e}")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", e))
                loop.call_soon_threadsafe(queue.put_nowait, None)

        executor_task = loop.run_in_executor(self._executor, _stream_worker)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                tag, data = item
                if tag == "error":
                    raise data
                yield tag, data
        finally:
            cancel_event.set()
            await executor_task

    async def _run_sync(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args))

    @staticmethod
    def _discover_active_auth_file() -> str | None:
        try:
            store = AccountStore()
            account = store.get_active_account()
            if account is None:
                return None
            path = store.get_auth_path_optional(account.id, require_exists=False)
            return str(path) if path is not None else None
        except Exception:
            return None

    @staticmethod
    def _derive_profile_dir(auth_file: str | None) -> str | None:
        if not auth_file:
            fallback_auth_file = BrowserSession._discover_active_auth_file()
            if not fallback_auth_file:
                return None
            auth_file = fallback_auth_file
        return str(Path(auth_file).resolve().parent / "profile")

    def _bootstrap_google_session_sync(self, page) -> None:
        """Visit Google surfaces so Chromium can materialize a stable profile."""
        page.goto(GOOGLE_LOGIN_BOOTSTRAP_URL, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)
        for url in (AI_STUDIO_URL, AI_STUDIO_URL_FALLBACK):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(1500)
                if "accounts.google.com" not in (page.url or ""):
                    return
            except Exception:
                continue
        raise RuntimeError(f"bootstrap stayed on login flow: url={page.url}")

    def _get_captured_info(self) -> tuple[str, dict[str, str]]:
        """Get captured URL and headers from template."""
        for tpl in self._templates.values():
            if tpl.get("url"):
                url = tpl["url"]
                headers = {k: v for k, v in tpl.get("headers", {}).items() if k.lower() not in ("host", "content-length")}
                return url, headers
        raise RuntimeError("no captured URL available for replay")

    def _resolve_captured_info(
        self,
        captured_url: str | None,
        captured_headers: dict[str, str] | None,
    ) -> tuple[str, dict[str, str]]:
        if captured_url is None or captured_headers is None:
            fallback_url, fallback_headers = self._get_captured_info()
            if captured_url is None:
                captured_url = fallback_url
            if captured_headers is None:
                captured_headers = fallback_headers
        return captured_url, captured_headers

    def _send_streaming_request_sync(
        self,
        body: str,
        timeout_ms: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        cancel_event: threading.Event,
        captured_url: str | None = None,
        captured_headers: dict[str, str] | None = None,
    ):
        """Sync method: sends XHR request and consumes page-side stream events."""
        import time as _t
        _t0 = _t.time()

        page = self._ensure_botguard_service_sync()
        captured_url, captured_headers = self._resolve_captured_info(captured_url, captured_headers)
        log.debug(f"[stream] prep done in {_t.time()-_t0:.1f}s, url={captured_url}")

        timeout_s = timeout_ms / 1000
        rid = uuid.uuid4().hex[:8]

        # Start XHR in page context. Each request gets an isolated state object
        # keyed by rid, allowing multiple concurrent XHRs on the same page.
        page.evaluate("""(args) => {
            const rid = args.rid;
            if (!window.__streams) window.__streams = {};

            const existing = window.__streams[rid];
            if (existing && existing.xhr && existing.xhr.readyState !== 4) {
                try { existing.xhr.abort(); } catch (e) {}
            }

            const state = {
                xhr: null,
                events: [],
                waiter: null,
                recvPos: 0,
                statusSent: false,
            };
            window.__streams[rid] = state;

            function push(event) {
                if (state.waiter) {
                    const waiter = state.waiter;
                    state.waiter = null;
                    waiter(event);
                    return;
                }
                state.events.push(event);
            }

            function pushStatus(xhr) {
                if (state.statusSent || xhr.readyState < 2) return;
                state.statusSent = true;
                push({type: 'status', status: xhr.status || 0});
            }

            function pushChunk(xhr) {
                if (xhr.readyState < 3) return;
                const chunk = xhr.responseText.substring(state.recvPos);
                if (!chunk) return;
                state.recvPos = xhr.responseText.length;
                push({type: 'chunk', text: chunk});
            }

            if (!window.__stream_next) window.__stream_next = {};
            window.__stream_next[rid] = function(timeoutMs) {
                if (state.events.length) return Promise.resolve(state.events.shift());
                return new Promise((resolve) => {
                    let done = false;
                    const timer = setTimeout(() => {
                        if (done) return;
                        done = true;
                        if (state.waiter === finish) state.waiter = null;
                        resolve({type: 'idle'});
                    }, timeoutMs);
                    const finish = (event) => {
                        if (done) return;
                        done = true;
                        clearTimeout(timer);
                        resolve(event);
                    };
                    state.waiter = finish;
                });
            };

            if (!window.__stream_abort) window.__stream_abort = {};
            window.__stream_abort[rid] = function() {
                if (state.xhr && state.xhr.readyState !== 4) {
                    try { state.xhr.abort(); } catch (e) {}
                }
            };

            var xhr = new XMLHttpRequest();
            xhr.open('POST', args.url);
            var h = args.headers;
            for (var k in h) {
                xhr.setRequestHeader(k, h[k]);
            }
            xhr.withCredentials = true;
            xhr.timeout = args.timeout * 1000;

            xhr.onreadystatechange = function() {
                pushStatus(xhr);
                pushChunk(xhr);
            };
            xhr.onprogress = function() {
                pushStatus(xhr);
                pushChunk(xhr);
            };
            xhr.onload = function() {
                pushStatus(xhr);
                pushChunk(xhr);
                push({type: 'done'});
            };
            xhr.onerror = function() {
                push({type: 'error', message: 'network error'});
            };
            xhr.ontimeout = function() {
                push({type: 'error', message: 'timeout'});
            };
            xhr.onabort = function() {
                push({type: 'aborted'});
            };

            state.xhr = xhr;
            xhr.send(args.body);
        }""", {
            "url": captured_url,
            "headers": captured_headers,
            "body": body,
            "timeout": timeout_s,
            "rid": rid,
        })

        deadline = _t.time() + timeout_s
        status_sent = False
        while _t.time() < deadline:
            if cancel_event.is_set():
                log.debug("[stream] cancellation requested for %s", rid)
                page.evaluate("rid => { if (window.__stream_abort && window.__stream_abort[rid]) window.__stream_abort[rid](); }", rid)
                break

            event = page.evaluate("rid => window.__stream_next[rid](250)", rid)
            event_type = event.get("type")

            if event_type == "idle":
                continue
            if event_type == "status":
                status = event.get("status", 0)
                log.debug(f"[stream] got status={status} after {_t.time()-_t0:.1f}s")
                loop.call_soon_threadsafe(queue.put_nowait, ("status", status))
                status_sent = True
                continue
            if event_type == "chunk":
                text = event.get("text") or ""
                if text:
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", text.encode("utf-8")))
                continue
            if event_type == "error":
                message = event.get("message", "unknown error")
                log.debug(f"[stream] error after {_t.time()-_t0:.1f}s: {message}")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", RuntimeError(f"streaming request failed: {message}")))
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return
            if event_type in ("done", "aborted"):
                break

        if not status_sent:
            log.debug(f"[stream] timeout after {_t.time()-_t0:.1f}s before response status")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", RuntimeError("streaming request timeout: no response status")))
            loop.call_soon_threadsafe(queue.put_nowait, None)
            return

        # Signal completion
        loop.call_soon_threadsafe(queue.put_nowait, None)

    def _prepare_streaming_sync(self):
        """Prepare page for streaming request. Returns (page, url, headers)."""
        page = self._ensure_botguard_service_sync()
        if not self._templates:
            # Template not yet captured — capture one so we have a URL
            from aistudio_api.config import DEFAULT_TEXT_MODEL
            try:
                self._capture_template_sync(DEFAULT_TEXT_MODEL)
            except Exception as e:
                log.warning("auto template capture failed: %s", e)
        url, headers = self._get_captured_info()
        return page, url, headers

    def _switch_auth_sync(self, auth_file: str | None) -> None:
        self._auth_file = auth_file
        self._profile_dir = self._derive_profile_dir(auth_file)
        self._templates.clear()
        self._bootstrap_template = None
        self._close_sync()

    def _ensure_browser_sync(self):
        if self._ctx is not None and self._hook_page is not None and not self._hook_page.is_closed():
            return self._ctx

        import time as _t
        _t0 = _t.time()

        self._close_sync()

        # Chromium backend: auth.json
        if not is_camoufox_engine():
            return self._ensure_browser_chromium_sync(_t0)

        # Legacy mode: Camoufox + auth.json
        from camoufox.sync_api import Camoufox
        from aistudio_api.config import build_camoufox_proxy

        self._cf = Camoufox(
            headless=settings.browser_headless,
            main_world_eval=True,
            proxy=build_camoufox_proxy(settings.proxy_url),
        )
        self._browser = self._cf.__enter__()
        self._ctx = self._browser.new_context(**build_browser_context_options())
        self._apply_auth_file_sync()
        self._hook_page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        sync_maximize_page_window(self._hook_page)
        log.debug(f"[timing] browser launched in {_t.time()-_t0:.1f}s")
        self._goto_aistudio_sync(self._hook_page)
        log.debug(f"[timing] page loaded in {_t.time()-_t0:.1f}s")
        self._install_hooks_sync(self._hook_page)
        log.debug(f"[timing] hooks installed in {_t.time()-_t0:.1f}s")
        return self._ctx

    def _ensure_browser_chromium_sync(self, _t0: float):
        """Chromium backend: prefer per-account persistent profile, fallback to auth.json."""
        import time as _t

        profile_dir = self._profile_dir
        should_seed_from_auth = True
        if profile_dir:
            profile_path = Path(profile_dir)
            # If a profile already existed before launch, trust it as the source of
            # truth and do not re-inject auth.json on failures. Mixing the two was
            # causing Google to flag the profile's cookie state as inconsistent.
            should_seed_from_auth = not (profile_path.exists() and any(profile_path.iterdir()))
            profile_path.mkdir(parents=True, exist_ok=True)
            _clear_chromium_profile_locks(profile_path)
            self._ctx = sync_launch_persistent_context(
                profile_dir,
                **build_browser_context_options(),
            )
            self._browser = None
            self._cf = None
            self._playwright = None
        else:
            self._browser, self._cf, self._playwright = sync_launch_browser()
            self._ctx = self._browser.new_context(**build_browser_context_options())

        self._hook_page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        sync_maximize_page_window(self._hook_page)

        # First, see whether the persistent profile / current context is already alive.
        try:
            self._hook_page.goto("https://aistudio.google.com/", wait_until="domcontentloaded", timeout=15000)
            if "accounts.google.com" not in (self._hook_page.url or ""):
                profile_label = "profile" if profile_dir else "auth.json cache"
                log.info("[chromium-auth] %s hit", profile_label)
                self._goto_aistudio_sync(self._hook_page)
                self._install_hooks_sync(self._hook_page)
                log.debug(f"[timing] page loaded (cached) in {_t.time()-_t0:.1f}s")
                return self._ctx
        except Exception as e:
            log.debug("[chromium-auth] initial profile check failed: %s", e)

        # Only seed a fresh profile from auth.json once. After a profile exists,
        # auth.json must not be re-injected or it can corrupt the browser state.
        if should_seed_from_auth and self._auth_file and Path(self._auth_file).exists():
            try:
                data = json.loads(Path(self._auth_file).read_text())
                cached = data.get("cookies") or []
                if cached:
                    self._ctx.add_cookies(cached)
                    self._bootstrap_google_session_sync(self._hook_page)
                    if "accounts.google.com" not in (self._hook_page.url or ""):
                        log.info("[chromium-auth] auth.json seeded context (%d cookies)", len(cached))
                        self._save_cookies_sync()
                        self._goto_aistudio_sync(self._hook_page)
                        self._install_hooks_sync(self._hook_page)
                        log.debug(f"[timing] page loaded (cached) in {_t.time()-_t0:.1f}s")
                        return self._ctx
                    log.info("[chromium-auth] auth.json appears expired")
            except Exception as e:
                log.debug("[chromium-auth] auth.json load failed: %s", e)
        elif profile_dir:
            log.info("[chromium-auth] existing profile present; skipped auth.json seeding to avoid cookie pollution")

        log.debug(f"[timing] browser launched in {_t.time()-_t0:.1f}s")
        self._goto_aistudio_sync(self._hook_page)
        log.debug(f"[timing] page loaded in {_t.time()-_t0:.1f}s")
        self._install_hooks_sync(self._hook_page)
        log.debug(f"[timing] hooks installed in {_t.time()-_t0:.1f}s")
        return self._ctx

    def _apply_auth_file_sync(self):
        """Legacy mode: load cookies from auth.json."""
        if self._auth_file and Path(self._auth_file).exists():
            log.info(f"Loading auth from: {self._auth_file}")
            data = json.loads(Path(self._auth_file).read_text())
            cookies = data.get("cookies") or []
            if cookies:
                self._ctx.add_cookies(cookies)
                log.info(f"Added {len(cookies)} cookies to context")
        else:
            log.warning(f"No auth_file! self._auth_file={self._auth_file}")


    def _save_cookies_sync(
        self,
        *,
        auth_file: str | None = None,
        cookies: list[dict[str, Any]] | None = None,
    ) -> None:
        """将 cookie 保存回 auth.json。"""
        target_auth_file = auth_file or self._auth_file
        if not target_auth_file:
            return
        try:
            current_cookies = cookies
            if current_cookies is None:
                if self._ctx is None:
                    return
                current_cookies = self._ctx.cookies()
            if not current_cookies:
                return
            auth_path = Path(target_auth_file)
            # 读取现有的 origins 数据（如果有）
            origins = []
            if auth_path.exists():
                try:
                    existing = json.loads(auth_path.read_text())
                    origins = existing.get("origins", [])
                except Exception:
                    pass
            auth_path.parent.mkdir(parents=True, exist_ok=True)
            auth_path.write_text(json.dumps({"cookies": current_cookies, "origins": origins}, indent=2))
            log.info(f"Saved {len(current_cookies)} cookies to {target_auth_file}")
        except Exception as e:
            log.debug(f"Failed to save cookies: {e}")

    def _ensure_hook_page_sync(self):
        self._ensure_browser_sync()
        if "aistudio.google.com" not in (self._hook_page.url or ""):
            self._goto_aistudio_sync(self._hook_page)
        self._install_hooks_sync(self._hook_page)
        return self._hook_page

    def _ensure_botguard_service_sync(self):
        import time as _t
        _t0 = _t.time()
        page = self._ensure_hook_page_sync()
        if page.evaluate("mw:!!window.__bg_service"):
            log.debug(f"[timing] botguard cached, took {_t.time()-_t0:.1f}s")
            return page

        captured: dict[str, Any] = {}

        def on_request(request):
            if "GenerateContent" not in request.url or "Count" in request.url or captured:
                return
            body = request.post_data
            if not body:
                return
            captured["url"] = request.url
            captured["headers"] = dict(request.headers)
            captured["body"] = body

        page.evaluate(DIALOG_CLEANUP_JS)
        textarea = page.query_selector("textarea")
        if textarea is None:
            # Debug: show page state
            try:
                dbg_url = page.url
                dbg_title = page.title()
                dbg_body = page.evaluate("() => document.body?.innerText?.substring(0, 300) || ''")
            except Exception:
                dbg_url = dbg_title = dbg_body = '<error>'
            raise RuntimeError(f"textarea not found while capturing BotGuardService; url={dbg_url}, title={dbg_title}, body={dbg_body[:200]}")
        original_text = self._read_textarea_value_sync(textarea)
        page.on("request", on_request)
        try:
            textarea.fill(BOTGUARD_BOOTSTRAP_PROMPT)
            page.wait_for_timeout(800)
            page.evaluate(DIALOG_CLEANUP_JS)
            if not self._click_run_button_sync(page):
                raise RuntimeError("failed to trigger send while capturing BotGuardService")

            for i in range(45):
                page.wait_for_timeout(1000)
                if page.evaluate("mw:!!window.__bg_service"):
                    self._wait_until_idle_sync(page)
                    if captured and self._bootstrap_template is None:
                        self._bootstrap_template = dict(captured)
                    self._restore_textarea_value_sync(textarea, original_text)
                    log.debug(f"[timing] botguard captured after {i+1}s, total {_t.time()-_t0:.1f}s")
                    return page

            raise RuntimeError("BotGuardService capture timeout")
        finally:
            page.remove_listener("request", on_request)
            self._restore_textarea_value_sync(textarea, original_text)

    def _capture_template_sync(self, model: str) -> dict[str, Any]:
        import time as _t
        _t0 = _t.time()
        if model in self._templates:
            log.debug(f"[timing] template cached for {model}")
            return self._templates[model]

        page = self._ensure_botguard_service_sync()
        if self._bootstrap_template:
            captured = dict(self._bootstrap_template)
            self._templates[model] = captured
            log.debug(f"[timing] reused bootstrap template for {model} in {_t.time()-_t0:.1f}s")
            return captured
        log.debug(f"[timing] botguard done in {_t.time()-_t0:.1f}s, starting template capture")
        captured: dict[str, Any] = {}
        last_generate_response: dict[str, Any] | None = None

        def on_request(request):
            if "GenerateContent" not in request.url or "Count" in request.url or captured:
                return
            body = request.post_data
            if not body or len(body) <= 100:
                return
            captured["url"] = request.url
            captured["headers"] = dict(request.headers)
            captured["body"] = body

        def on_response(response):
            nonlocal last_generate_response
            if "GenerateContent" not in response.url or "Count" in response.url:
                return
            try:
                text = response.text()
            except Exception as exc:
                text = f"<response.text() failed: {exc}>"
            last_generate_response = {
                "status": response.status,
                "url": response.url,
                "body": text[:500],
            }

        page.on("request", on_request)
        page.on("response", on_response)
        try:
            textarea = page.query_selector("textarea")
            if textarea is None:
                raise RuntimeError("textarea not found during template capture")
            original_text = self._read_textarea_value_sync(textarea)
            textarea.fill(TEMPLATE_CAPTURE_PROMPT)
            page.wait_for_timeout(500)
            if not self._click_run_button_sync(page):
                raise RuntimeError("failed to trigger send during template capture")

            for _ in range(30):
                page.wait_for_timeout(1000)
                if captured:
                    break
            if not captured:
                if last_generate_response is not None:
                    raise RuntimeError(
                        "template capture failed after request: "
                        f"status={last_generate_response['status']} "
                        f"url={last_generate_response['url']} "
                        f"body={last_generate_response['body']}"
                    )
                raise RuntimeError(f"template capture timeout for model={model}")

            self._wait_until_idle_sync(page)
            self._restore_textarea_value_sync(textarea, original_text)
            self._templates[model] = captured
            log.debug(f"[timing] template captured for {model} in {_t.time()-_t0:.1f}s")
            return captured
        finally:
            page.remove_listener("request", on_request)
            page.remove_listener("response", on_response)
            if 'textarea' in locals() and textarea is not None and 'original_text' in locals():
                self._restore_textarea_value_sync(textarea, original_text)

    def _generate_snapshot_sync(self, contents: list[AistudioContent]) -> str:
        page = self._ensure_botguard_service_sync()
        if not self._snap_key:
            raise RuntimeError("Snapshot function not detected")

        content_hash = _snapshot_content_hash(contents)

        page.evaluate(
            """
mw:((hash) => {
    const dms = window.default_MakerSuite;
    const service = window.__bg_service;
    const snapKey = window.__snap_key;
    if (!dms || !service || !snapKey || typeof dms[snapKey] !== 'function') {
        window.__sr = '';
        window.__sl = 0;
        window.__snap_error = 'service_unavailable';
        return;
    }
    window.__sr = '';
    window.__sl = 0;
    window.__snap_error = '';
    const result = dms[snapKey](service, hash);
    if (result instanceof Promise) {
        result.then((snapshot) => {
            window.__sr = snapshot || '';
            window.__sl = snapshot ? snapshot.length : 0;
        }).catch((error) => {
            window.__snap_error = String(error);
        });
        return;
    }
    window.__sr = result || '';
    window.__sl = result ? result.length : 0;
})(%s)
"""
            % json.dumps(content_hash)
        )
        for _ in range(20):
            if page.evaluate("mw:(window.__sl || 0)") > 0:
                break
            page.wait_for_timeout(500)

        snapshot = page.evaluate("mw:window.__sr")
        if snapshot:
            return snapshot
        error = page.evaluate("mw:window.__snap_error || ''")
        raise RuntimeError(f"Snapshot generation failed: {error or 'unknown'}")

    def _upload_images_sync(self, image_paths: list[str]) -> list[str]:
        if not image_paths:
            return []

        # 尝试非 UI 方式上传（更快、更可靠）
        # 需要在主线程中获取 cookies，因为 Playwright 的同步 API 有 greenlet 限制
        try:
            if self._ctx is not None:
                cookies = self._ctx.cookies()
                return self._upload_images_via_api_sync(image_paths, cookies)
        except Exception as e:
            # 如果非 UI 方式失败，回退到 UI 方式
            log.debug("Non-UI upload failed, falling back to UI: %s", e)

        # UI 方式上传（原有逻辑）
        page = self._ensure_botguard_service_sync()
        self._wait_until_idle_sync(page)
        uploaded_ids: list[str] = []

        def on_response(response):
            if "content.googleapis.com/upload/drive/v3/files" not in response.url:
                return
            try:
                payload = json.loads(response.text())
            except Exception:
                return
            file_id = payload.get("id")
            if file_id:
                uploaded_ids.append(file_id)

        page.on("response", on_response)
        try:
            for image_path in image_paths:
                target_count = len(uploaded_ids) + 1
                page.evaluate(DIALOG_CLEANUP_JS)
                upload_btn = page.locator('[aria-label="Insert images, videos, audio, or files"]').first
                if not upload_btn.is_visible(timeout=3000):
                    raise RuntimeError("upload button not visible")
                upload_btn.click()
                page.wait_for_timeout(1500)
                page.evaluate(DIALOG_CLEANUP_JS)
                upload_files_btn = page.locator("text=Upload files").first
                if not upload_files_btn.is_visible(timeout=3000):
                    upload_btn.click()
                    page.wait_for_timeout(1000)
                    upload_files_btn = page.locator("text=Upload files").first
                if not upload_files_btn.is_visible(timeout=3000):
                    raise RuntimeError("upload files button not visible")
                with page.expect_file_chooser(timeout=10000) as chooser_info:
                    upload_files_btn.click()
                chooser_info.value.set_files(image_path)

                deadline = time.time() + 30
                while time.time() < deadline:
                    if len(uploaded_ids) >= target_count:
                        break
                    page.wait_for_timeout(500)
                page.wait_for_timeout(1500)
        finally:
            page.remove_listener("response", on_response)

        if len(uploaded_ids) != len(image_paths):
            raise RuntimeError(f"image upload incomplete: expected={len(image_paths)} uploaded={len(uploaded_ids)}")
        return uploaded_ids

    def _upload_images_via_api_sync(self, image_paths: list[str], cookies: list[dict]) -> list[str]:
        """通过 Playwright 的 setInputFiles 方法上传图片（非 UI 点击方式）"""
        page = self._hook_page
        if page is None:
            raise RuntimeError("Hook page not initialized")

        uploaded_ids: list[str] = []

        def on_response(response):
            if "content.googleapis.com/upload/drive/v3/files" not in response.url:
                return
            try:
                payload = json.loads(response.text())
            except Exception:
                return
            file_id = payload.get("id")
            if file_id:
                uploaded_ids.append(file_id)

        page.on("response", on_response)
        try:
            # 找到文件输入元素（如果有的话）
            file_input = page.query_selector('input[type="file"]')

            if file_input:
                # 直接使用 setInputFiles 方法上传
                for image_path in image_paths:
                    target_count = len(uploaded_ids) + 1
                    file_input.set_input_files(image_path)

                    # 等待上传完成
                    deadline = time.time() + 30
                    while time.time() < deadline:
                        if len(uploaded_ids) >= target_count:
                            break
                        page.wait_for_timeout(500)
                    page.wait_for_timeout(1000)
            else:
                # 如果没有 file input，尝试创建一个
                page.evaluate("""
                    () => {
                        const input = document.createElement('input');
                        input.type = 'file';
                        input.id = '__api_file_input__';
                        input.style.display = 'none';
                        input.accept = 'image/*';
                        document.body.appendChild(input);

                        // 监听文件选择事件
                        input.addEventListener('change', (e) => {
                            const file = e.target.files[0];
                            if (file) {
                                // 触发上传逻辑
                                window.__api_upload_file = file;
                            }
                        });
                    }
                """)

                file_input = page.query_selector('#__api_file_input__')
                if not file_input:
                    raise RuntimeError("Failed to create file input")

                for image_path in image_paths:
                    target_count = len(uploaded_ids) + 1
                    file_input.set_input_files(image_path)
                    page.wait_for_timeout(1000)

                    # 触发上传
                    page.evaluate("""
                        () => {
                            if (window.__api_upload_file) {
                                // 模拟拖放或触发上传按钮
                                const event = new Event('change', { bubbles: true });
                                const input = document.querySelector('#__api_file_input__');
                                if (input) input.dispatchEvent(event);
                            }
                        }
                    """)

                    # 等待上传完成
                    deadline = time.time() + 30
                    while time.time() < deadline:
                        if len(uploaded_ids) >= target_count:
                            break
                        page.wait_for_timeout(500)
                    page.wait_for_timeout(1000)

        finally:
            page.remove_listener("response", on_response)

        if len(uploaded_ids) != len(image_paths):
            raise RuntimeError(f"image upload incomplete: expected={len(image_paths)} uploaded={len(uploaded_ids)}")
        return uploaded_ids

    def _send_hooked_request_sync(
        self,
        body: str,
        timeout_ms: int,
        captured_url: str | None = None,
        captured_headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        import time as _t
        _t0 = _t.time()
        page = self._ensure_botguard_service_sync()
        log.debug(f"[timing] botguard ready in {_t.time()-_t0:.1f}s")
        captured_url, captured_headers = self._resolve_captured_info(captured_url, captured_headers)

        # Replay via XHR in browser context (same approach as non-streaming replay_v2)
        timeout_s = timeout_ms / 1000
        result = page.evaluate("""(args) => {
            return new Promise((resolve) => {
                var xhr = new XMLHttpRequest();
                xhr.open('POST', args.url);
                var h = args.headers;
                for (var k in h) {
                    xhr.setRequestHeader(k, h[k]);
                }
                xhr.withCredentials = true;
                xhr.timeout = args.timeout * 1000;
                xhr.onload = function() {
                    resolve({status: xhr.status, body: xhr.responseText});
                };
                xhr.onerror = function() {
                    resolve({status: 0, body: 'network error'});
                };
                xhr.ontimeout = function() {
                    resolve({status: 0, body: 'timeout'});
                };
                xhr.send(args.body);
            });
        }""", {
            "url": captured_url,
            "headers": captured_headers,
            "body": body,
            "timeout": timeout_s,
        })

        status = result.get("status", 0)
        raw_text = result.get("body", "")
        log.debug(f"[timing] replay done in {_t.time()-_t0:.1f}s, status={status}")
        if status == 0:
            raise RuntimeError(f"replay failed: {raw_text}")
        return status, raw_text.encode("utf-8")

    def _goto_aistudio_sync(self, page) -> None:
        import time as _t
        last_exc = None
        for url in (AI_STUDIO_URL, AI_STUDIO_URL_FALLBACK):
            try:
                _t0 = _t.time()
                page.goto(url, wait_until="networkidle", timeout=30000)
                log.debug(f"[timing] goto {url} took {_t.time()-_t0:.1f}s")
                # 检查是否被重定向到登录页
                current_url = page.url or ""
                if _is_google_signin_url(current_url):
                    raise RuntimeError(
                        f"Cookie 认证失败，已被重定向到 Google 登录页。"
                        f" (url={current_url})"
                    )
                if not _is_aistudio_url(current_url):
                    raise RuntimeError(f"AI Studio redirected away before UI ready: url={current_url}")
                # Wait for SPA framework and chat UI to render
                for _ in range(60):
                    page.wait_for_timeout(1000)
                    current_url = page.url or ""
                    if _is_google_signin_url(current_url):
                        raise RuntimeError(
                            f"Cookie 认证失败，已被重定向到 Google 登录页。"
                            f" (url={current_url})"
                        )
                    if not _is_aistudio_url(current_url):
                        raise RuntimeError(f"AI Studio redirected away before UI ready: url={current_url}")
                    has_dms = page.evaluate("mw:!!window.default_MakerSuite")
                    has_textarea = page.query_selector("textarea") is not None
                    if has_dms and has_textarea:
                        log.debug(f"[timing] UI ready (dms+textarea) after {_t.time()-_t0:.1f}s")
                        self._save_cookies_sync()
                        return
                    if has_dms and _ > 20:
                        page.evaluate(DIALOG_CLEANUP_JS)
                raise RuntimeError(
                    f"AI Studio UI not ready after {_t.time()-_t0:.1f}s "
                    f"(url={page.url}, dms={has_dms}, textarea={has_textarea})"
                )
            except Exception as exc:
                log.debug(f"[timing] goto {url} failed after {_t.time()-_t0:.1f}s: {exc}")
                last_exc = exc
        if last_exc is not None:
            raise last_exc

    def _install_hooks_sync(self, page) -> None:
        result = page.evaluate(INSTALL_HOOKS_JS)
        if result == "already_hooked":
            return
        if isinstance(result, str) and result.startswith("hooked:"):
            self._snap_key = result.split(":", 1)[1]
            return
        for _ in range(3):
            page.wait_for_timeout(2000)
            result = page.evaluate(INSTALL_HOOKS_JS)
            if result == "already_hooked":
                return
            if isinstance(result, str) and result.startswith("hooked:"):
                self._snap_key = result.split(":", 1)[1]
                return
        page_url = page.url if page else "(no page)"
        page_title = ""
        try:
            page_title = page.title()
        except Exception:
            pass
        raise RuntimeError(f"Hook install failed: {result} (url={page_url}, title={page_title!r})")

    def _click_run_button_sync(self, page) -> bool:
        # Ctrl+Enter is the most reliable trigger on the new AI Studio layout.
        # The visible "Run" button is a ctrl-enter-submit control and clicking it
        # via ElementHandle can be flaky.  Ensure the textarea is focused first so
        # the shortcut reaches the right target.
        try:
            textarea = page.query_selector("textarea")
            if textarea is not None:
                textarea.focus()
            page.keyboard.press("Control+Enter")
            return True
        except Exception:
            pass

        # 2. Fallback: locate the real submit button by its CSS class, avoiding
        #    false matches from category cards that happen to contain "Run".
        try:
            button = page.query_selector("button.ctrl-enter-submits")
            if button is None:
                button = page.locator("button", has_text="Run").filter(
                    has=page.locator("keyboard_return")
                ).first
                button = button.element_handle(timeout=2000)
        except Exception:
            button = None
        if button is None:
            return False
        try:
            button.click()
            return True
        except Exception:
            return False

    def _has_run_button_sync(self, page) -> bool:
        # Idle = no Stop button visible AND submit button present.
        # During generation AI Studio shows a "Stop" button; checking for its
        # absence is more reliable than the submit button's disabled state.
        try:
            if page.query_selector("button:has-text('Stop')") is not None:
                return False
            return page.query_selector("button.ctrl-enter-submits") is not None
        except Exception:
            return False

    def _wait_until_idle_sync(self, page) -> None:
        for _ in range(60):
            if self._has_run_button_sync(page):
                return
            page.wait_for_timeout(1000)
        raise RuntimeError("page never became idle")

    def _read_textarea_value_sync(self, textarea) -> str:
        try:
            return textarea.input_value()
        except Exception:
            return ""

    def _restore_textarea_value_sync(self, textarea, value: str) -> None:
        try:
            current = textarea.input_value()
        except Exception:
            current = None
        if current == value:
            return
        try:
            textarea.fill(value)
        except Exception:
            pass

    def _close_sync(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:
                pass
        if self._browser is not None and self._cf is None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._cf is not None:
            try:
                self._cf.__exit__(None, None, None)
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._hook_page = None
        self._ctx = None
        self._browser = None
        self._cf = None
        self._playwright = None
        self._snap_key = None
        self._templates.clear()
        self._bootstrap_template = None

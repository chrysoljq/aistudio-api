"""FastAPI application entrypoint."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from aistudio_api.infrastructure.gateway.client import AIStudioClient

from .routes_anthropic import router as anthropic_router
from .dependencies import require_api_key
from .routes_accounts import router as accounts_router
from .routes_gemini import router as gemini_router
from .routes_openai import router as openai_router
from .routes_system import protected_router as system_protected_router
from .routes_system import public_router as system_public_router
from .state import runtime_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("aistudio.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from aistudio_api.config import settings
    from aistudio_api.infrastructure.account.account_store import AccountStore
    from aistudio_api.infrastructure.account.login_service import LoginService
    from aistudio_api.application.account_service import AccountService
    from aistudio_api.application.account_rotator import init_rotator, RotationMode

    client = AIStudioClient(
        port=runtime_state.browser_port,
    )
    runtime_state.client = client
    from aistudio_api.config import settings as app_settings
    runtime_state.busy_lock = asyncio.Semaphore(app_settings.max_concurrency)

    # 初始化账号管理服务
    account_store = AccountStore()
    login_service = LoginService(port=settings.login_browser_port)
    account_service = AccountService(account_store, login_service)
    runtime_state.account_service = account_service

    # 初始化账号轮询器
    rotation_mode = getattr(settings, "account_rotation_mode", "round_robin")
    cooldown = getattr(settings, "account_cooldown_seconds", 60)
    rotator = init_rotator(
        account_store,
        mode=RotationMode(rotation_mode),
        cooldown_seconds=cooldown,
    )
    runtime_state.rotator = rotator

    logger.info(
        "Client initialized (browser=%s, port=%s, rotation=%s, accounts=%d)",
        settings.browser_engine,
        runtime_state.browser_port,
        rotator.mode,
        len(account_store.list_accounts()),
    )

    # 后台预热浏览器，避免首次请求延迟
    warmup_task = None
    async def _warmup():
        try:
            await client.warmup()
        except Exception as e:
            logger.warning("浏览器预热失败: %s", e)
    warmup_task = asyncio.create_task(_warmup())

    yield
    logger.info("Shutting down")
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
    runtime_state.client = None
    runtime_state.busy_lock = None
    runtime_state.account_service = None
    runtime_state.rotator = None


app = FastAPI(title="AI Studio API", lifespan=lifespan)
app.include_router(system_public_router)
app.include_router(system_protected_router, dependencies=[Depends(require_api_key)])
app.include_router(gemini_router, dependencies=[Depends(require_api_key)])
app.include_router(openai_router, dependencies=[Depends(require_api_key)])
app.include_router(anthropic_router, dependencies=[Depends(require_api_key)])
app.include_router(accounts_router, dependencies=[Depends(require_api_key)])

# 挂载静态文件
import os
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/login")
async def login_page():
    return RedirectResponse(url="/static/login.html")


@app.get("/auth/check")
async def auth_check():
    """检查认证状态，用于前端判断是否需要登录。"""
    from aistudio_api.config import settings
    return {"auth_enabled": settings.auth_enabled}


def main():
    from aistudio_api.config import settings

    parser = argparse.ArgumentParser(description="AI Studio OpenAI-compatible API Server")
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--browser-port", type=int, default=settings.browser_port)
    parser.add_argument("--camoufox-port", type=int, dest="browser_port", help=argparse.SUPPRESS)
    args = parser.parse_args()

    runtime_state.browser_port = args.browser_port

    import uvicorn

    logger.info("Starting server on port %s", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

"""Shared helpers for API service handlers."""

from __future__ import annotations

import base64
import contextvars
import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import HTTPException

from aistudio_api.config import settings
from aistudio_api.api.response_models import (
    HealthResponse,
    ImageGenerationResponse,
    ImageResponseData,
    ModelStatsResponse,
    StatsResponse,
    StatsTotalsResponse,
)
from aistudio_api.api.state import runtime_state
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.wire_types import AistudioPart

logger = logging.getLogger("aistudio.server")
MAX_RETRIES = 3
_current_account_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aistudio_current_account_id",
    default=None,
)


def validate_image_request_options(*, size: str, n: int) -> None:
    if n != 1:
        raise HTTPException(
            400,
            detail={"message": "Only n=1 is currently supported", "type": "invalid_request_error"},
        )
    if AIStudioClient.resolve_image_size(size) is None:
        raise HTTPException(
            400,
            detail={"message": f"Unsupported image size '{size}'", "type": "invalid_request_error"},
        )


async def build_inline_image_parts(image_files: list) -> list[AistudioPart]:
    parts: list[AistudioPart] = []
    for image_file in image_files:
        mime = image_file.content_type or mimetypes.guess_type(image_file.filename or "")[0] or "image/png"
        content = await image_file.read()
        parts.append(AistudioPart(inline_data=(mime, base64.b64encode(content).decode("ascii"))))
    return parts


async def try_switch_account() -> bool:
    """尝试切换到下一个可用账号。返回是否成功切换。"""
    rotator = runtime_state.rotator
    if rotator is None:
        return False

    account_service = runtime_state.account_service
    client = runtime_state.client
    if not all([account_service, client]):
        return False

    max_attempts = 1
    if hasattr(rotator, "get_all_stats"):
        try:
            max_attempts = max(1, len(rotator.get_all_stats()))
        except Exception:
            max_attempts = 1

    for _ in range(max_attempts):
        next_account = await rotator.get_next_account()
        if next_account is None:
            return False

        active_account = account_service.get_active_account()
        if active_account is not None and active_account.id == next_account.id:
            return True

        try:
            result = await account_service.activate_account(
                next_account.id,
                client._session,
                runtime_state.snapshot_cache,
                None,  # skip lock — caller already holds it
                keep_snapshot_cache=False,
            )
        except Exception:
            logger.warning("切换账号失败，跳过账号: %s", next_account.id, exc_info=True)
            rotator.record_error(next_account.id)
            continue

        if result is not None:
            return True
        rotator.record_error(next_account.id)

    return False


def require_busy_lock():
    busy_lock = runtime_state.busy_lock
    if busy_lock is None:
        raise HTTPException(503, detail={"message": "Server not ready", "type": "service_unavailable"})
    if busy_lock.locked():
        raise HTTPException(429, detail={"message": "Server is busy", "type": "rate_limit_exceeded"})
    return busy_lock


def is_client_pool_enabled() -> bool:
    pool = runtime_state.client_pool
    return bool(pool and getattr(pool, "enabled", False))


@asynccontextmanager
async def request_client(default_client: AIStudioClient, attempt: int = 0) -> AsyncIterator[AIStudioClient]:
    pool = runtime_state.client_pool
    if pool and getattr(pool, "enabled", False):
        async with pool.acquire() as lease:
            token = _current_account_id.set(lease.account_id)
            try:
                yield lease.client
            finally:
                _current_account_id.reset(token)
        return

    busy_lock = require_busy_lock()
    async with busy_lock:
        await ensure_active_account(attempt)
        yield default_client


async def should_retry_rate_limit() -> bool:
    if is_client_pool_enabled():
        return True
    return await try_switch_account()


async def ensure_active_account(attempt: int) -> None:
    if attempt != 0:
        return
    account_svc = runtime_state.account_service
    if account_svc and runtime_state.rotator:
        active_account = account_svc.get_active_account()
        sticky_seconds = settings.account_rotation_sticky_seconds
        if active_account is not None and sticky_seconds > 0 and active_account.last_used:
            try:
                last_switch = datetime.fromisoformat(active_account.last_used)
                if last_switch.tzinfo is None:
                    last_switch = last_switch.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - last_switch.astimezone(timezone.utc)).total_seconds()
                if age < sticky_seconds:
                    return
            except ValueError:
                pass
        await try_switch_account()


def record_rotator_event(event: str) -> None:
    rotator = runtime_state.rotator
    account_service = runtime_state.account_service
    account_id = _current_account_id.get()
    if account_id is None and account_service:
        account = account_service.get_active_account()
        account_id = account.id if account else None
    if not rotator or account_id is None:
        return
    if event == "success":
        rotator.record_success(account_id)
    elif event == "rate_limited":
        rotator.record_rate_limited(account_id)
    elif event == "error":
        rotator.record_error(account_id)


def image_response(output: Any) -> ImageGenerationResponse:
    data: list[ImageResponseData] = []
    for img in output.images:
        b64 = base64.b64encode(img.data).decode("ascii")
        data.append(ImageResponseData(b64_json=b64, revised_prompt=output.text or ""))
    return ImageGenerationResponse(created=int(time.time()), data=data)


def health_response() -> HealthResponse:
    busy_lock = runtime_state.busy_lock
    pool = runtime_state.client_pool
    if pool and getattr(pool, "enabled", False):
        status = pool.status()
        busy = bool(status["accounts"]) and all(account["busy"] for account in status["accounts"])
        return HealthResponse(status="ok", busy=busy)
    return HealthResponse(status="ok", busy=busy_lock.locked() if busy_lock else False)


def stats_response() -> StatsResponse:
    stats = dict(runtime_state.model_stats)
    totals = StatsTotalsResponse(
        requests=sum(s["requests"] for s in stats.values()),
        success=sum(s["success"] for s in stats.values()),
        rate_limited=sum(s["rate_limited"] for s in stats.values()),
        errors=sum(s["errors"] for s in stats.values()),
        prompt_tokens=sum(s["prompt_tokens"] for s in stats.values()),
        completion_tokens=sum(s["completion_tokens"] for s in stats.values()),
        total_tokens=sum(s["total_tokens"] for s in stats.values()),
    )
    models = {name: ModelStatsResponse(**values) for name, values in stats.items()}
    return StatsResponse(models=models, totals=totals)

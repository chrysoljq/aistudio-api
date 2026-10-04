"""Shared helpers for API service handlers."""

from __future__ import annotations

import base64
import logging
import mimetypes
import time
from typing import Any

from fastapi import HTTPException

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

    next_account = await rotator.get_next_account()
    if next_account is None:
        return False

    account_service = runtime_state.account_service
    client = runtime_state.client
    if not all([account_service, client]):
        return False

    result = await account_service.activate_account(
        next_account.id,
        client._session,
        runtime_state.snapshot_cache,
        None,  # skip lock — caller already holds it
        keep_snapshot_cache=False,
    )
    return result is not None


def require_busy_lock():
    busy_lock = runtime_state.busy_lock
    if busy_lock is None:
        raise HTTPException(503, detail={"message": "Server not ready", "type": "service_unavailable"})
    if busy_lock.locked():
        raise HTTPException(429, detail={"message": "Server is busy", "type": "rate_limit_exceeded"})
    return busy_lock


async def ensure_active_account(attempt: int) -> None:
    if attempt != 0:
        return
    account_svc = runtime_state.account_service
    if account_svc and not account_svc.get_active_account():
        await try_switch_account()


def record_rotator_event(event: str) -> None:
    rotator = runtime_state.rotator
    account_service = runtime_state.account_service
    account = account_service.get_active_account() if account_service else None
    if not rotator or account is None:
        return
    if event == "success":
        rotator.record_success(account.id)
    elif event == "rate_limited":
        rotator.record_rate_limited(account.id)
    elif event == "error":
        rotator.record_error(account.id)


def image_response(output: Any) -> ImageGenerationResponse:
    data: list[ImageResponseData] = []
    for img in output.images:
        b64 = base64.b64encode(img.data).decode("ascii")
        data.append(ImageResponseData(b64_json=b64, revised_prompt=output.text or ""))
    return ImageGenerationResponse(created=int(time.time()), data=data)


def health_response() -> HealthResponse:
    busy_lock = runtime_state.busy_lock
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


# ============================================================================
# ARM overlay (2026-09-24, guo-issue-202609242350)：上游内容过滤可见化共享件。
# 背景：AI Studio wire 对被过滤请求返回空 content + 非 0 finish 码（实测 11=带
# 政策文案 / 15=无文案）。上游实现把 finish 码丢弃、空 content 静默 200，消费端
# 只能看到空结果或系统异常，内容决策被伪装成系统故障（反模式见 GUO
# memos/空内容语义丢失伪装成系统故障反模式）。本组 helper 只做「如实翻译」，
# 不改变也不绕过上游过滤行为。
# ============================================================================

def arm_block_notice(output) -> str:
    """非流式：把「空内容 + 非 0 finish 码」翻译成可读文本。

    判定域与 2026-09-24 首版 overlay 逐位一致：text/images/function_calls 任一
    存在即视为有内容（thinking 不参与判定，会经 to_gemini_parts 原样透传）。
    仅当上游确实没产出任何内容且带非零 finish 码时返回非空字符串；否则返回 ""，
    不影响正常路径。绝无异常外抛。
    """
    try:
        if output.text or output.images or output.function_calls:
            return ""
        candidates = output.candidates or []
        if not candidates:
            return ""
        finish_code = candidates[0].finish_reason
        if not finish_code:
            return ""
        finish_message = (candidates[0].finish_message or "").strip()
        return _arm_notice_text(finish_code, finish_message)
    except Exception:
        return ""


def arm_block_notice_from_finish(finish: dict | None) -> str:
    """流式：把 gateway 透出的 finish 事件翻译成可读文本。语义同 arm_block_notice。"""
    try:
        if not finish:
            return ""
        finish_code = finish.get("code")
        if not finish_code:
            return ""
        return _arm_notice_text(finish_code, (finish.get("message") or "").strip())
    except Exception:
        return ""


def _arm_notice_text(finish_code, finish_message: str) -> str:
    return (
        f"上游未返回内容（finish_code={finish_code}）"
        + (f"：{finish_message}" if finish_message
           else "：生成结果被内容安全策略拦截，请调整提示词后重试")
    )

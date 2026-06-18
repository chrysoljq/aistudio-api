"""Gemini-compatible application service handlers."""

from __future__ import annotations

import json

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from aistudio_api.api.response_models import GeminiCandidateResponse, GeminiContentResponse, GeminiGenerateContentResponse
from aistudio_api.api.responses import to_gemini_parts, to_gemini_usage_metadata
from aistudio_api.api.schemas import GeminiGenerateContentRequest
from aistudio_api.api.state import runtime_state
from aistudio_api.application.api_service_common import (
    MAX_RETRIES,
    ensure_active_account,
    logger,
    record_rotator_event,
    require_busy_lock,
    try_switch_account,
)
from aistudio_api.application.chat_service import cleanup_files, normalize_gemini_request
from aistudio_api.domain.errors import AistudioError, AuthError, RequestError, UsageLimitExceeded
from aistudio_api.infrastructure.gateway.client import AIStudioClient


async def handle_gemini_generate_content(
    model_path: str,
    req: GeminiGenerateContentRequest,
    client: AIStudioClient,
    *,
    stream: bool,
):
    busy_lock = require_busy_lock()
    last_error = None

    for attempt in range(MAX_RETRIES):
        async with busy_lock:
            await ensure_active_account(attempt)
            normalized = None
            try:
                normalized = normalize_gemini_request(req, model_path)
                logger.info(
                    "Gemini: model=%s, contents=%s, stream=%s, attempt=%d",
                    normalized["model"],
                    len(req.contents),
                    stream,
                    attempt + 1,
                )

                if stream:
                    return _build_gemini_streaming_response(client=client, normalized=normalized)

                output = await client.generate_content(
                    model=normalized["model"],
                    capture_prompt=normalized["capture_prompt"],
                    capture_images=normalized["capture_images"],
                    contents=normalized["contents"],
                    system_instruction_content=normalized["system_instruction"],
                    tools=normalized["tools"],
                    safety_settings=normalized["safety_settings"],
                    temperature=normalized["temperature"],
                    top_p=normalized["top_p"],
                    top_k=normalized["top_k"],
                    max_tokens=normalized["max_tokens"],
                    generation_config_overrides=normalized["generation_config_overrides"],
                    sanitize_plain_text=False,
                )

                record_rotator_event("success")
                runtime_state.record(normalized["model"], "success", output.usage)
                return GeminiGenerateContentResponse(
                    candidates=[
                        GeminiCandidateResponse(
                            content=GeminiContentResponse(
                                parts=to_gemini_parts(
                                    output.text,
                                    function_calls=output.function_calls,
                                    function_responses=output.function_responses,
                                    thinking=output.thinking,
                                    images=output.images,
                                    reasoning_images=output.reasoning_images,
                                ),
                            ),
                            finishReason="STOP" if not output.function_calls else "FUNCTION_CALL",
                        )
                    ],
                    usageMetadata=to_gemini_usage_metadata(output.usage),
                )
            except ValueError as exc:
                raise HTTPException(400, detail={"message": str(exc), "type": "bad_request"}) from exc
            except UsageLimitExceeded as exc:
                runtime_state.record(model_path, "rate_limited")
                last_error = exc

                record_rotator_event("rate_limited")
                if await try_switch_account():
                    logger.info("Gemini 429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                    continue
                logger.warning("Gemini 429 限流，无法切换账号")
                raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(model_path, "errors")
                record_rotator_event("error")
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(model_path, "errors")
                record_rotator_event("error")
                logger.error("Gemini error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            finally:
                if normalized is not None and not stream:
                    cleanup_files(normalized["cleanup_paths"])

    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


def _build_gemini_streaming_response(*, client: AIStudioClient, normalized: dict) -> StreamingResponse:
    async def stream_response():
        busy_lock = runtime_state.busy_lock
        if busy_lock is None:
            yield "data: " + json.dumps({"error": {"message": "Server not ready"}}, ensure_ascii=False) + "\n\n"
            cleanup_files(normalized["cleanup_paths"])
            return

        async with busy_lock:
            try:
                final_usage = None
                for stream_attempt in range(MAX_RETRIES):
                    try:
                        has_yielded_data = False
                        async for event_type, text in client.stream_generate_content(
                            model=normalized["model"],
                            capture_prompt=normalized["capture_prompt"],
                            capture_images=normalized["capture_images"],
                            contents=normalized["contents"],
                            system_instruction_content=normalized["system_instruction"],
                            tools=normalized["tools"],
                            safety_settings=normalized["safety_settings"],
                            temperature=normalized["temperature"],
                            top_p=normalized["top_p"],
                            top_k=normalized["top_k"],
                            max_tokens=normalized["max_tokens"],
                            generation_config_overrides=normalized["generation_config_overrides"],
                            sanitize_plain_text=False,
                            force_refresh_capture=stream_attempt > 0,
                        ):
                            has_yielded_data = True
                            if event_type == "body" and text:
                                yield "data: " + json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "content": {"role": "model", "parts": [{"text": text}]},
                                                "finishReason": None,
                                            }
                                        ]
                                    },
                                    ensure_ascii=False,
                                ) + "\n\n"
                            elif event_type == "images" and text:
                                yield "data: " + json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "content": {
                                                    "role": "model",
                                                    "parts": [
                                                        part.model_dump(mode="json", exclude_none=True)
                                                        for part in to_gemini_parts("", images=text)
                                                    ],
                                                },
                                                "finishReason": None,
                                            }
                                        ]
                                    },
                                    ensure_ascii=False,
                                ) + "\n\n"
                            elif event_type == "reasoning_images" and text:
                                yield "data: " + json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "content": {
                                                    "role": "model",
                                                    "parts": [
                                                        part.model_dump(mode="json", exclude_none=True)
                                                        for part in to_gemini_parts("", reasoning_images=text)
                                                    ],
                                                },
                                                "finishReason": None,
                                            }
                                        ]
                                    },
                                    ensure_ascii=False,
                                ) + "\n\n"
                            elif event_type == "thinking" and text:
                                yield "data: " + json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "content": {
                                                    "role": "model",
                                                    "parts": [{"text": text, "thought": True}],
                                                },
                                                "finishReason": None,
                                            }
                                        ]
                                    },
                                    ensure_ascii=False,
                                ) + "\n\n"
                            elif event_type == "usage":
                                final_usage = text if isinstance(text, dict) else None
                        break
                    except UsageLimitExceeded:
                        runtime_state.record(normalized["model"], "rate_limited")
                        record_rotator_event("rate_limited")
                        if not has_yielded_data and await try_switch_account():
                            logger.warning("Gemini stream 429 限流，已切换账号，重试 %d/%d", stream_attempt + 1, MAX_RETRIES)
                            continue
                        raise
                    except RequestError as exc:
                        if exc.status == 204 and stream_attempt == 0:
                            logger.warning("Gemini stream 收到 204，刷新 capture 后重试一次")
                            continue
                        raise
                    except AuthError as exc:
                        if stream_attempt == 0:
                            logger.warning("Gemini stream 鉴权异常，刷新 capture 后重试一次: %s", exc)
                            continue
                        raise

                record_rotator_event("success")
                runtime_state.record(normalized["model"], "success", final_usage)
                if final_usage:
                    yield "data: " + json.dumps(
                        {
                            "candidates": [],
                            "usageMetadata": to_gemini_usage_metadata(final_usage).model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    ) + "\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("Gemini stream error: %s", exc, exc_info=True)
                if not isinstance(exc, UsageLimitExceeded):
                    record_rotator_event("error")
                runtime_state.record(normalized["model"], "errors")
                yield "data: " + json.dumps({"error": {"message": str(exc)}}, ensure_ascii=False) + "\n\n"
            finally:
                cleanup_files(normalized["cleanup_paths"])

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

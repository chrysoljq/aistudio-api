"""OpenAI-compatible application service handlers."""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from aistudio_api.api.responses import (
    chat_completion_response,
    new_chat_id,
    sse_chunk,
    sse_error,
    sse_usage_chunk,
    to_openai_tool_calls,
)
from aistudio_api.api.schemas import ChatRequest, ImageRequest
from aistudio_api.api.state import runtime_state
from aistudio_api.application.api_service_common import (
    MAX_RETRIES,
    build_inline_image_parts,
    ensure_active_account,
    image_response,
    logger,
    record_rotator_event,
    require_busy_lock,
    try_switch_account,
    validate_image_request_options,
)
from aistudio_api.application.chat_service import cleanup_files, normalize_chat_request, normalize_openai_tools
from aistudio_api.domain.errors import AistudioError, AuthError, RequestError, UsageLimitExceeded
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.model_defaults import resolve_model_defaults
from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart


async def handle_chat(req: ChatRequest, client: AIStudioClient):
    busy_lock = require_busy_lock()
    last_error = None

    for attempt in range(MAX_RETRIES):
        async with busy_lock:
            await ensure_active_account(attempt)
            normalized = normalize_chat_request(req.messages, req.model)
            model = normalized["model"]
            tmp_files = list(normalized["cleanup_paths"])

            try:
                logger.info(
                    "Chat: model=%s, contents=%s, capture_prompt=%s..., images=%s, stream=%s, attempt=%d",
                    model,
                    len(normalized["contents"]),
                    normalized["capture_prompt"][:50],
                    len(normalized["capture_images"]),
                    req.stream,
                    attempt + 1,
                )
                tools = None if req.tools is None else (normalize_openai_tools(req.tools) or [])

                if req.tools is None:
                    from aistudio_api.infrastructure.gateway.request_rewriter import build_tools_from_names

                    model_defaults = resolve_model_defaults(model)
                    if model_defaults.default_tools:
                        tools = build_tools_from_names(
                            model_defaults.default_tools,
                            model=model,
                            is_image_model=model_defaults.is_image_model,
                        )

                if req.stream:
                    include_usage = True
                    if req.stream_options is not None:
                        include_usage = req.stream_options.include_usage
                    return _build_streaming_response(
                        client=client,
                        capture_prompt=normalized["capture_prompt"],
                        model=model,
                        capture_images=normalized["capture_images"] if normalized["capture_images"] else None,
                        contents=normalized["contents"],
                        system_instruction=normalized["system_instruction"],
                        cleanup_paths=tmp_files,
                        include_usage=include_usage,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        max_tokens=req.max_tokens,
                        tools=tools,
                    )

                output = await client.generate_content(
                    model=model,
                    capture_prompt=normalized["capture_prompt"],
                    capture_images=normalized["capture_images"] if normalized["capture_images"] else None,
                    contents=normalized["contents"],
                    system_instruction_content=(
                        AistudioContent(role="user", parts=[AistudioPart(text=normalized["system_instruction"])])
                        if normalized["system_instruction"]
                        else None
                    ),
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    max_tokens=req.max_tokens,
                    tools=tools,
                    sanitize_plain_text=True,
                )

                record_rotator_event("success")
                runtime_state.record(model, "success", output.usage)
                return chat_completion_response(
                    model=model,
                    content=output.text,
                    thinking=output.thinking,
                    usage=output.usage,
                    function_calls=output.function_calls,
                )
            except UsageLimitExceeded as exc:
                runtime_state.record(model, "rate_limited")
                last_error = exc

                record_rotator_event("rate_limited")
                if await try_switch_account():
                    logger.info("429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                    continue
                logger.warning("429 限流，无法切换账号")
                raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                logger.error("Chat error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            finally:
                if not req.stream:
                    cleanup_files(tmp_files)

    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


async def handle_image_generation(req: ImageRequest, client: AIStudioClient):
    validate_image_request_options(size=req.size, n=req.n)

    busy_lock = require_busy_lock()
    last_error = None

    for attempt in range(MAX_RETRIES):
        async with busy_lock:
            await ensure_active_account(attempt)
            try:
                logger.info("Image: model=%s, prompt=%s..., attempt=%d", req.model, req.prompt[:50], attempt + 1)
                output = await client.generate_image(
                    prompt=req.prompt,
                    model=req.model,
                    size=req.size,
                    google_search=req.google_search,
                    image_search=req.image_search,
                    use_default_tools=not bool({"google_search", "image_search"} & req.model_fields_set),
                )
                record_rotator_event("success")
                runtime_state.record(req.model, "success", output.usage)
                return image_response(output)
            except UsageLimitExceeded as exc:
                runtime_state.record(req.model, "rate_limited")
                last_error = exc

                record_rotator_event("rate_limited")
                if await try_switch_account():
                    logger.info("Image 429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                    continue
                logger.warning("Image 429 限流，无法切换账号")
                raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(req.model, "errors")
                record_rotator_event("error")
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(req.model, "errors")
                record_rotator_event("error")
                logger.error("Image error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc

    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


async def handle_image_edit(
    prompt: str,
    image_files: list,
    mask_file,
    model: str,
    n: int,
    size: str,
    client: AIStudioClient,
):
    validate_image_request_options(size=size, n=n)

    busy_lock = require_busy_lock()
    image_parts = await build_inline_image_parts(image_files)
    request_contents = [AistudioContent(role="user", parts=[*image_parts, AistudioPart(text=prompt)])]

    last_error = None

    for attempt in range(MAX_RETRIES):
        async with busy_lock:
            await ensure_active_account(attempt)
            try:
                logger.info(
                    "Image Edit: model=%s, prompt=%s..., images=%d, attempt=%d",
                    model,
                    prompt[:50],
                    len(image_parts),
                    attempt + 1,
                )
                output = await client.generate_image(
                    prompt=prompt,
                    model=model,
                    size=size,
                    contents=request_contents,
                )
                record_rotator_event("success")
                runtime_state.record(model, "success", output.usage)
                return image_response(output)
            except UsageLimitExceeded as exc:
                runtime_state.record(model, "rate_limited")
                last_error = exc

                record_rotator_event("rate_limited")
                if await try_switch_account():
                    logger.info("Image Edit 429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                    continue
                logger.warning("Image Edit 429 限流，无法切换账号")
                raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                logger.error("Image Edit error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc

    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


def _build_streaming_response(
    *,
    client: AIStudioClient,
    capture_prompt: str,
    model: str,
    capture_images: list[str] | None,
    contents: list[AistudioContent],
    system_instruction: str | None,
    cleanup_paths: list[str],
    include_usage: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    tools: list[list] | None = None,
) -> StreamingResponse:
    async def stream_response():
        busy_lock = runtime_state.busy_lock
        if busy_lock is None:
            yield sse_error("Server not ready")
            cleanup_files(cleanup_paths)
            return

        async with busy_lock:
            try:
                chat_id = new_chat_id()
                final_usage = None
                saw_tool_calls = False
                for stream_attempt in range(MAX_RETRIES):
                    try:
                        has_yielded_data = False
                        async for event_type, text in client.stream_generate_content(
                            model=model,
                            capture_prompt=capture_prompt,
                            capture_images=capture_images,
                            contents=contents,
                            system_instruction_content=(
                                AistudioContent(role="user", parts=[AistudioPart(text=system_instruction)])
                                if system_instruction
                                else None
                            ),
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_tokens=max_tokens,
                            tools=tools,
                            force_refresh_capture=stream_attempt > 0,
                        ):
                            has_yielded_data = True
                            if event_type == "body" and text:
                                yield sse_chunk(chat_id, model, text, include_usage=include_usage)
                            elif event_type == "thinking" and text:
                                yield sse_chunk(chat_id, model, "", thinking=text, include_usage=include_usage)
                            elif event_type == "tool_calls" and text:
                                saw_tool_calls = True
                                yield sse_chunk(
                                    chat_id,
                                    model,
                                    "",
                                    tool_calls=to_openai_tool_calls(text if isinstance(text, list) else []),
                                    include_usage=include_usage,
                                )
                            elif event_type == "usage":
                                final_usage = text if isinstance(text, dict) else None
                        break
                    except UsageLimitExceeded:
                        runtime_state.record(model, "rate_limited")
                        record_rotator_event("rate_limited")
                        if not has_yielded_data and stream_attempt < MAX_RETRIES - 1 and await try_switch_account():
                            logger.warning("Stream 429 限流，已切换账号，重试 %d/%d", stream_attempt + 1, MAX_RETRIES)
                            continue
                        raise
                    except RequestError as exc:
                        if exc.status == 204 and stream_attempt == 0:
                            logger.warning("Stream 收到 204，刷新 capture 后重试一次")
                            continue
                        raise
                    except AuthError as exc:
                        if stream_attempt == 0:
                            logger.warning("Stream 鉴权异常，刷新 capture 后重试一次: %s", exc)
                            continue
                        raise

                record_rotator_event("success")
                runtime_state.record(model, "success", final_usage)
                yield sse_chunk(chat_id, model, "", finish="tool_calls" if saw_tool_calls else "stop", include_usage=include_usage)
                if include_usage:
                    yield sse_usage_chunk(chat_id, model, final_usage)
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("Stream error: %s", exc, exc_info=True)
                if not isinstance(exc, UsageLimitExceeded):
                    record_rotator_event("error")
                runtime_state.record(model, "errors")
                yield sse_error(str(exc))
            finally:
                cleanup_files(cleanup_paths)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

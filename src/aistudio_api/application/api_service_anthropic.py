"""Anthropic-compatible application service handlers."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from aistudio_api.api.responses import (
    anthropic_error_sse,
    anthropic_message_response,
    anthropic_sse,
    anthropic_usage,
    function_call_args,
    new_chat_id,
    new_message_id,
)
from aistudio_api.api.schemas import AnthropicMessageRequest
from aistudio_api.api.state import runtime_state
from aistudio_api.application.api_service_common import (
    MAX_RETRIES,
    ensure_active_account,
    logger,
    record_rotator_event,
    require_busy_lock,
    try_switch_account,
)
from aistudio_api.application.chat_service import (
    ANTHROPIC_TOOL_HISTORY_END,
    ANTHROPIC_TOOL_HISTORY_START,
    cleanup_files,
    normalize_anthropic_request,
)
from aistudio_api.domain.errors import AistudioError, AuthError, RequestError, UsageLimitExceeded
from aistudio_api.infrastructure.gateway.client import AIStudioClient


def _longest_marker_prefix_suffix(text: str, marker: str) -> int:
    max_length = min(len(text), len(marker) - 1)
    for length in range(max_length, 0, -1):
        if marker.startswith(text[-length:]):
            return length
    return 0


def _filter_internal_tool_history_delta(text: str, *, state: dict[str, object], final: bool = False) -> str:
    """Drop leaked internal tool-history transcript fragments from model text."""
    pending = str(state.get("pending") or "")
    remaining = pending + (text or "")
    state["pending"] = ""
    output_parts: list[str] = []

    while remaining:
        if state.get("in_tool_history"):
            end_index = remaining.find(ANTHROPIC_TOOL_HISTORY_END)
            if end_index < 0:
                keep = _longest_marker_prefix_suffix(remaining, ANTHROPIC_TOOL_HISTORY_END)
                state["pending"] = remaining[-keep:] if keep else ""
                return "".join(output_parts)
            remaining = remaining[end_index + len(ANTHROPIC_TOOL_HISTORY_END) :]
            state["in_tool_history"] = False
            continue

        start_index = remaining.find(ANTHROPIC_TOOL_HISTORY_START)
        if start_index < 0:
            keep = 0 if final else _longest_marker_prefix_suffix(remaining, ANTHROPIC_TOOL_HISTORY_START)
            if keep:
                output_parts.append(remaining[:-keep])
                state["pending"] = remaining[-keep:]
            else:
                output_parts.append(remaining)
            break

        output_parts.append(remaining[:start_index])
        remaining = remaining[start_index + len(ANTHROPIC_TOOL_HISTORY_START) :]
        state["in_tool_history"] = True

    if final and not state.get("in_tool_history") and state.get("pending"):
        output_parts.append(str(state["pending"]))
        state["pending"] = ""

    return "".join(output_parts)


def _filter_internal_tool_history_text(text: str) -> str:
    state = {"in_tool_history": False}
    return _filter_internal_tool_history_delta(text or "", state=state, final=True)


def _anthropic_tool_names(req: AnthropicMessageRequest) -> set[str]:
    return {tool.name for tool in req.tools or [] if tool.name}


def _balanced_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for pos in range(start, len(text)):
            current = text[pos]
            if escape:
                escape = False
                continue
            if current == "\\" and in_string:
                escape = True
                continue
            if current == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : pos + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    break
    return objects


def _normalize_tool_name(name: str, allowed_names: set[str]) -> str | None:
    if not name:
        return None
    candidates = [name, name.split(":", 1)[-1]]
    lowered = {tool_name.lower(): tool_name for tool_name in allowed_names}
    for candidate in candidates:
        if candidate in allowed_names:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return candidates[-1] if not allowed_names else None


def _fallback_function_calls_from_thinking(
    *,
    thinking: str,
    allowed_names: set[str],
    thought_signature: str = "",
) -> list[dict[str, Any]]:
    for match in re.finditer(r"Call:([A-Za-z0-9_.:-]+)\s*(?=\{)", thinking):
        raw_name = match.group(1)
        name = _normalize_tool_name(raw_name, allowed_names)
        if not name:
            continue
        objects = _balanced_json_objects(thinking[match.end() :])
        if not objects:
            continue
        call: dict[str, Any] = {"name": name, "args": objects[0]}
        call["synthetic"] = True
        if thought_signature:
            call["thought_signature"] = thought_signature
        logger.warning("Recovered malformed AI Studio Call: function call from thinking text: %s", name)
        return [call]

    for obj in _balanced_json_objects(thinking):
        raw_name = obj.get("name") or obj.get("function") or obj.get("tool")
        if not isinstance(raw_name, str):
            continue
        name = _normalize_tool_name(raw_name, allowed_names)
        if not name:
            continue
        args = obj.get("arguments", obj.get("args", obj.get("input", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"arguments": args}
        if not isinstance(args, dict):
            args = {"arguments": args}
        call: dict[str, Any] = {"name": name, "args": args}
        call["synthetic"] = True
        if thought_signature:
            call["thought_signature"] = thought_signature
        logger.warning("Recovered malformed AI Studio function call from thinking text: %s", name)
        return [call]
    return []


def _prepare_anthropic_function_calls(function_calls: list[dict] | None) -> list[dict]:
    prepared: list[dict] = []
    for function_call in function_calls or []:
        call = dict(function_call)
        call_id = call.get("call_id")
        raw = call.get("raw")
        if not call_id and isinstance(raw, list) and len(raw) > 2 and isinstance(raw[2], str):
            call_id = raw[2]
        if not call_id:
            call_id = uuid.uuid4().hex[:12]
        tool_use_id = f"toolu_{call_id}"
        call["call_id"] = call_id
        call["anthropic_tool_use_id"] = tool_use_id
        thought_signature = call.get("thought_signature")
        if thought_signature:
            runtime_state.anthropic_tool_context[tool_use_id] = {
                "call_id": call_id,
                "thought_signature": thought_signature,
                "name": call.get("name", "unknown"),
                "synthetic": bool(call.get("synthetic")),
            }
        elif call.get("synthetic"):
            runtime_state.anthropic_tool_context[tool_use_id] = {
                "call_id": call_id,
                "thought_signature": "",
                "name": call.get("name", "unknown"),
                "synthetic": True,
            }
        prepared.append(call)
    return prepared


async def handle_anthropic_messages(req: AnthropicMessageRequest, client: AIStudioClient):
    busy_lock = require_busy_lock()

    normalized = normalize_anthropic_request(req, tool_context=runtime_state.anthropic_tool_context)
    model = normalized["model"]
    cleanup_paths = list(normalized["cleanup_paths"])
    tool_names = _anthropic_tool_names(req)

    if req.stream:
        return _build_anthropic_streaming_response(
            client=client,
            normalized=normalized,
            cleanup_paths=cleanup_paths,
            tool_names=tool_names,
        )

    last_error = None
    try:
        for attempt in range(MAX_RETRIES):
            async with busy_lock:
                await ensure_active_account(attempt)
                try:
                    logger.info(
                        "Anthropic messages: model=%s, contents=%s, capture_prompt=%s..., stream=%s, attempt=%d",
                        model,
                        len(normalized["contents"]),
                        normalized["capture_prompt"][:50],
                        req.stream,
                        attempt + 1,
                    )
                    output = await client.generate_content(
                        model=model,
                        capture_prompt=normalized["capture_prompt"],
                        capture_images=normalized["capture_images"],
                        contents=normalized["contents"],
                        system_instruction_content=normalized["system_instruction"],
                        temperature=normalized["temperature"],
                        top_p=normalized["top_p"],
                        top_k=normalized["top_k"],
                        max_tokens=normalized["max_tokens"],
                        tools=normalized["tools"],
                        sanitize_plain_text=not bool(normalized["tools"]),
                    )

                    record_rotator_event("success")
                    runtime_state.record(model, "success", output.usage)
                    raw_function_calls = output.function_calls
                    if not raw_function_calls and not output.text and output.candidates:
                        candidate = output.candidates[0]
                        raw_function_calls = _fallback_function_calls_from_thinking(
                            thinking=output.thinking,
                            allowed_names=tool_names,
                            thought_signature=candidate.thought_signature,
                        )
                    function_calls = _prepare_anthropic_function_calls(raw_function_calls)
                    return anthropic_message_response(
                        model=model,
                        content=_filter_internal_tool_history_text(output.text),
                        usage=output.usage,
                        function_calls=function_calls,
                    )
                except UsageLimitExceeded as exc:
                    runtime_state.record(model, "rate_limited")
                    last_error = exc
                    record_rotator_event("rate_limited")
                    if await try_switch_account():
                        logger.info("Anthropic 429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                        continue
                    raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_error"}) from exc
                except AistudioError as exc:
                    runtime_state.record(model, "errors")
                    record_rotator_event("error")
                    raise HTTPException(500, detail={"message": str(exc), "type": "api_error"}) from exc
                except Exception as exc:
                    runtime_state.record(model, "errors")
                    record_rotator_event("error")
                    logger.error("Anthropic messages error: %s", exc, exc_info=True)
                    raise HTTPException(500, detail={"message": str(exc), "type": "api_error"}) from exc

        raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_error"}) from last_error
    finally:
        cleanup_files(cleanup_paths)


def _build_anthropic_streaming_response(
    *,
    client: AIStudioClient,
    normalized: dict,
    cleanup_paths: list[str],
    tool_names: set[str],
) -> StreamingResponse:
    async def stream_response():
        busy_lock = runtime_state.busy_lock
        model = normalized["model"]
        if busy_lock is None:
            yield anthropic_error_sse("Server not ready")
            cleanup_files(cleanup_paths)
            return

        async with busy_lock:
            try:
                message_id = new_message_id()
                content_block_index = 0
                text_block_started = False
                text_block_index = 0
                final_usage = None
                saw_tool_use = False
                thinking_fragments: list[str] = []
                latest_thought_signature = ""

                yield anthropic_sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )

                for stream_attempt in range(MAX_RETRIES):
                    try:
                        tool_history_filter_state = {"in_tool_history": False}
                        has_yielded_model_data = False
                        async for event_type, text in client.stream_generate_content(
                            model=model,
                            capture_prompt=normalized["capture_prompt"],
                            capture_images=normalized["capture_images"],
                            contents=normalized["contents"],
                            system_instruction_content=normalized["system_instruction"],
                            temperature=normalized["temperature"],
                            top_p=normalized["top_p"],
                            top_k=normalized["top_k"],
                            max_tokens=normalized["max_tokens"],
                            tools=normalized["tools"],
                            sanitize_plain_text=not bool(normalized["tools"]),
                            force_refresh_capture=stream_attempt > 0,
                        ):
                            if event_type == "body" and text:
                                text = _filter_internal_tool_history_delta(
                                    str(text),
                                    state=tool_history_filter_state,
                                )
                                if not text:
                                    continue
                                has_yielded_model_data = True
                                if not text_block_started:
                                    text_block_index = content_block_index
                                    content_block_index += 1
                                    text_block_started = True
                                    yield anthropic_sse(
                                        "content_block_start",
                                        {
                                            "type": "content_block_start",
                                            "index": text_block_index,
                                            "content_block": {"type": "text", "text": ""},
                                        },
                                    )
                                yield anthropic_sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": text_block_index,
                                        "delta": {"type": "text_delta", "text": text},
                                    },
                                )
                            elif event_type == "thinking" and text:
                                thinking_fragments.append(str(text))
                            elif event_type == "thought_signature" and text:
                                latest_thought_signature = str(text)
                            elif event_type == "tool_calls" and text:
                                function_calls = _prepare_anthropic_function_calls(text if isinstance(text, list) else [])
                                for function_call in function_calls:
                                    has_yielded_model_data = True
                                    saw_tool_use = True
                                    tool_index = content_block_index
                                    content_block_index += 1
                                    yield anthropic_sse(
                                        "content_block_start",
                                        {
                                            "type": "content_block_start",
                                            "index": tool_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": function_call.get("anthropic_tool_use_id")
                                                or f"toolu_{new_chat_id().removeprefix('chatcmpl-')}",
                                                "name": function_call.get("name", "unknown"),
                                                "input": {},
                                            },
                                        },
                                    )
                                    yield anthropic_sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": tool_index,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": json.dumps(
                                                    function_call_args(function_call),
                                                    ensure_ascii=False,
                                                ),
                                            },
                                        },
                                    )
                                    yield anthropic_sse(
                                        "content_block_stop",
                                        {"type": "content_block_stop", "index": tool_index},
                                    )
                            elif event_type == "usage":
                                final_usage = text if isinstance(text, dict) else None
                        text = _filter_internal_tool_history_delta(
                            "",
                            state=tool_history_filter_state,
                            final=True,
                        )
                        if text:
                            has_yielded_model_data = True
                            if not text_block_started:
                                text_block_index = content_block_index
                                content_block_index += 1
                                text_block_started = True
                                yield anthropic_sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": text_block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                            yield anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_block_index,
                                    "delta": {"type": "text_delta", "text": text},
                                },
                            )
                        break
                    except UsageLimitExceeded as exc:
                        runtime_state.record(model, "rate_limited")
                        record_rotator_event("rate_limited")
                        if not has_yielded_model_data and stream_attempt < MAX_RETRIES - 1 and await try_switch_account():
                            logger.warning("Anthropic stream 429 限流，已切换账号，重试 %d/%d", stream_attempt + 1, MAX_RETRIES)
                            continue
                        raise exc
                    except RequestError as exc:
                        if exc.status == 204 and stream_attempt == 0:
                            logger.warning("Anthropic stream 收到 204，刷新 capture 后重试一次")
                            continue
                        raise
                    except AuthError as exc:
                        if stream_attempt == 0:
                            logger.warning("Anthropic stream 鉴权异常，刷新 capture 后重试一次: %s", exc)
                            continue
                        raise

                if not saw_tool_use and not text_block_started and thinking_fragments:
                    recovered_calls = _prepare_anthropic_function_calls(
                        _fallback_function_calls_from_thinking(
                            thinking="".join(thinking_fragments),
                            allowed_names=tool_names,
                            thought_signature=latest_thought_signature,
                        )
                    )
                    for function_call in recovered_calls:
                        saw_tool_use = True
                        tool_index = content_block_index
                        content_block_index += 1
                        yield anthropic_sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": function_call.get("anthropic_tool_use_id")
                                    or f"toolu_{new_chat_id().removeprefix('chatcmpl-')}",
                                    "name": function_call.get("name", "unknown"),
                                    "input": {},
                                },
                            },
                        )
                        yield anthropic_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(function_call_args(function_call), ensure_ascii=False),
                                },
                            },
                        )
                        yield anthropic_sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": tool_index},
                        )

                if text_block_started:
                    yield anthropic_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_block_index},
                    )

                record_rotator_event("success")
                runtime_state.record(model, "success", final_usage)
                yield anthropic_sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "tool_use" if saw_tool_use else "end_turn",
                            "stop_sequence": None,
                        },
                        "usage": anthropic_usage(final_usage).model_dump(mode="json"),
                    },
                )
                yield anthropic_sse("message_stop", {"type": "message_stop"})
            except Exception as exc:
                logger.error("Anthropic stream error: %s", exc, exc_info=True)
                if not isinstance(exc, UsageLimitExceeded):
                    record_rotator_event("error")
                runtime_state.record(model, "errors")
                yield anthropic_error_sse(str(exc))
            finally:
                cleanup_files(cleanup_paths)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

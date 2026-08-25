"""OpenAI-compatible HTTP/SSE response helpers."""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

from pydantic import BaseModel

from aistudio_api.api.response_models import (
    AnthropicMessageResponse,
    AnthropicTextBlockResponse,
    AnthropicToolUseBlockResponse,
    AnthropicUsageResponse,
    ErrorDetail,
    ErrorResponse,
    GeminiFunctionCallPayload,
    GeminiFunctionResponsePayload,
    GeminiInlineDataResponse,
    GeminiPartResponse,
    GeminiUsageMetadata,
    OpenAIChatChoice,
    OpenAIChatChunkChoice,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionResponse,
    OpenAIChatDelta,
    OpenAIChatMessage,
    OpenAIFunctionCallPayload,
    OpenAIToolCall,
    OpenAIUsage,
)
from aistudio_api.domain.models import GeneratedImage


def new_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] in "+-" and stripped[1:].isdigit())):
            return int(stripped)
    return 0


def normalize_usage(usage: dict | None = None) -> OpenAIUsage:
    usage = usage or {}
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    prompt_tokens = _coerce_usage_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_usage_int(usage.get("completion_tokens"))
    total_tokens = _coerce_usage_int(usage.get("total_tokens"))
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    return OpenAIUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tokens_details={"reasoning_tokens": _coerce_usage_int(completion_details.get("reasoning_tokens"))},
    )


def to_gemini_usage_metadata(usage: dict | None = None) -> GeminiUsageMetadata:
    usage = usage or {}
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    reasoning_tokens = _coerce_usage_int(completion_details.get("reasoning_tokens"))
    visible_tokens = _coerce_usage_int(completion_details.get("visible_tokens"))
    candidates_tokens = visible_tokens or _coerce_usage_int(usage.get("completion_tokens"))
    prompt_tokens = _coerce_usage_int(usage.get("prompt_tokens"))
    total_tokens = _coerce_usage_int(usage.get("total_tokens"))
    if total_tokens == 0 and (prompt_tokens or candidates_tokens or reasoning_tokens):
        total_tokens = prompt_tokens + _coerce_usage_int(usage.get("completion_tokens"))
    return GeminiUsageMetadata(
        promptTokenCount=prompt_tokens,
        candidatesTokenCount=candidates_tokens,
        thoughtsTokenCount=reasoning_tokens,
        totalTokenCount=total_tokens,
    )


def sse_chunk(
    chat_id: str,
    model: str,
    content: str,
    finish: str | None = None,
    thinking: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    include_usage: bool = True,
) -> str:
    data = OpenAIChatCompletionChunk(
        id=chat_id,
        created=int(time.time()),
        model=model,
        choices=[
            OpenAIChatChunkChoice(
                index=0,
                delta=OpenAIChatDelta(
                    content=content or None,
                    thinking=thinking,
                    reasoning_content=thinking,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish,
            )
        ],
        usage=None,
    )
    payload = data.model_dump(mode="json", exclude_none=True)
    if include_usage:
        payload["usage"] = None
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_usage_chunk(chat_id: str, model: str, usage: dict | None = None) -> str:
    data = OpenAIChatCompletionChunk(
        id=chat_id,
        created=int(time.time()),
        model=model,
        choices=[],
        usage=normalize_usage(usage),
    )
    return f"data: {data.model_dump_json()}\n\n"


def sse_error(message: str) -> str:
    data = ErrorResponse(error=ErrorDetail(message=message, type="server_error"))
    return f"data: {data.model_dump_json()}\n\n"


def anthropic_usage(usage: dict | None = None) -> AnthropicUsageResponse:
    normalized = normalize_usage(usage)
    return AnthropicUsageResponse(
        input_tokens=normalized.prompt_tokens,
        output_tokens=normalized.completion_tokens,
    )


def anthropic_sse(event_type: str, payload: dict[str, Any] | BaseModel) -> str:
    if isinstance(payload, BaseModel):
        body = payload.model_dump_json()
    else:
        body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {body}\n\n"


def anthropic_error_sse(message: str) -> str:
    return anthropic_sse(
        "error",
        {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        },
    )


def function_call_args(function_call: dict[str, Any]) -> dict[str, Any]:
    if isinstance(function_call.get("args"), dict):
        return function_call["args"]
    if "arguments" in function_call:
        raw_args = function_call["arguments"]
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                return {"arguments": raw_args}
            return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    raw = function_call.get("raw")
    if isinstance(raw, list) and len(raw) > 1:
        raw_args = raw[1]
        return raw_args if isinstance(raw_args, dict) else {"arguments": raw_args}
    return {}


def anthropic_message_response(
    *,
    model: str,
    content: str,
    usage: dict | None = None,
    function_calls: list[dict[str, Any]] | None = None,
) -> AnthropicMessageResponse:
    blocks: list[AnthropicTextBlockResponse | AnthropicToolUseBlockResponse] = []
    if content:
        blocks.append(AnthropicTextBlockResponse(text=content))
    for function_call in function_calls or []:
        blocks.append(
            AnthropicToolUseBlockResponse(
                id=function_call.get("anthropic_tool_use_id") or f"toolu_{uuid.uuid4().hex[:24]}",
                name=function_call.get("name", "unknown"),
                input=function_call_args(function_call),
            )
        )
    if not blocks:
        blocks.append(AnthropicTextBlockResponse(text=""))

    return AnthropicMessageResponse(
        id=new_message_id(),
        model=model,
        content=blocks,
        stop_reason="tool_use" if function_calls else "end_turn",
        stop_sequence=None,
        usage=anthropic_usage(usage),
    )


def _function_call_arguments(function_call: dict[str, Any]) -> str:
    if "args" in function_call:
        return json.dumps(function_call["args"], ensure_ascii=False)
    if "arguments" in function_call:
        return str(function_call["arguments"])
    raw = function_call.get("raw")
    if isinstance(raw, list) and len(raw) > 1:
        second = raw[1]
        if isinstance(second, str):
            return second
        return json.dumps(second, ensure_ascii=False)
    return "{}"


def to_openai_tool_calls(function_calls: list[dict[str, Any]]) -> list[OpenAIToolCall]:
    tool_calls: list[OpenAIToolCall] = []
    for idx, function_call in enumerate(function_calls):
        tool_calls.append(
            OpenAIToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}_{idx}",
                function=OpenAIFunctionCallPayload(
                    name=function_call.get("name", "unknown"),
                    arguments=_function_call_arguments(function_call),
                ),
            )
        )
    return tool_calls


def to_gemini_parts(
    content: str,
    function_calls: list[dict[str, Any]] | None = None,
    function_responses: list[dict[str, Any]] | None = None,
    thinking: str = "",
    images: list[GeneratedImage] | None = None,
    reasoning_images: list[GeneratedImage] | None = None,
) -> list[GeminiPartResponse]:
    parts: list[GeminiPartResponse] = []
    if thinking:
        parts.append(GeminiPartResponse(text=thinking, thought=True))
    for image in reasoning_images or []:
        parts.append(
            GeminiPartResponse(
                thought=True,
                thoughtSignature=image.thought_signature or None,
                inlineData=GeminiInlineDataResponse(
                    mimeType=image.mime,
                    data=base64.b64encode(image.data).decode("ascii"),
                ),
            )
        )
    if content:
        parts.append(GeminiPartResponse(text=content))
    for image in images or []:
        parts.append(
            GeminiPartResponse(
                thoughtSignature=image.thought_signature or None,
                inlineData=GeminiInlineDataResponse(
                    mimeType=image.mime,
                    data=base64.b64encode(image.data).decode("ascii"),
                )
            )
        )
    for function_call in function_calls or []:
        payload = GeminiFunctionCallPayload(name=function_call.get("name", "unknown"))
        if "args" in function_call:
            payload.args = function_call["args"]
        elif "arguments" in function_call:
            payload.args = function_call["arguments"]
        elif isinstance(function_call.get("raw"), list) and len(function_call["raw"]) > 1:
            payload.args = function_call["raw"][1]
        parts.append(GeminiPartResponse(functionCall=payload))
    for function_response in function_responses or []:
        payload = GeminiFunctionResponsePayload(name=function_response.get("name", "unknown"))
        if "args" in function_response:
            payload.response = function_response["args"]
        elif "arguments" in function_response:
            payload.response = function_response["arguments"]
        elif isinstance(function_response.get("raw"), list) and len(function_response["raw"]) > 1:
            payload.response = function_response["raw"][1]
        parts.append(GeminiPartResponse(functionResponse=payload))
    if not parts:
        parts.append(GeminiPartResponse(text=""))
    return parts


def chat_completion_response(
    model: str,
    content: str,
    thinking: str = "",
    usage: dict | None = None,
    function_calls: list[dict[str, Any]] | None = None,
) -> OpenAIChatCompletionResponse:
    finish_reason = "tool_calls" if function_calls else "stop"
    return OpenAIChatCompletionResponse(
        id=new_chat_id(),
        created=int(time.time()),
        model=model,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatMessage(
                    content=content,
                    thinking=thinking or None,
                    reasoning_content=thinking or None,
                    tool_calls=to_openai_tool_calls(function_calls) if function_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=normalize_usage(usage),
    )

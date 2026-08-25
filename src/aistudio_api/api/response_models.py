"""Structured HTTP response models."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class OpenAICompletionTokenDetails(BaseModel):
    reasoning_tokens: int = 0


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: OpenAICompletionTokenDetails = Field(
        default_factory=OpenAICompletionTokenDetails
    )


class OpenAIFunctionCallPayload(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallPayload


class OpenAIChatMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    thinking: str | None = None
    # Conventional field read by DeepSeek/OpenRouter-style clients
    reasoning_content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: str


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: OpenAIUsage


class OpenAIChatDelta(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    thinking: str | None = None
    # Conventional field read by DeepSeek/OpenRouter-style clients
    reasoning_content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChatChunkChoice(BaseModel):
    index: int
    delta: OpenAIChatDelta
    finish_reason: str | None = None


class OpenAIChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatChunkChoice]
    usage: OpenAIUsage | None = None


class ErrorDetail(BaseModel):
    message: str
    type: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ImageResponseData(BaseModel):
    b64_json: str
    revised_prompt: str = ""


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageResponseData]


class AnthropicUsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicTextBlockResponse(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseBlockResponse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class AnthropicMessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicTextBlockResponse | AnthropicToolUseBlockResponse]
    stop_reason: str
    stop_sequence: str | None = None
    usage: AnthropicUsageResponse


class AnthropicCountTokensResponse(BaseModel):
    input_tokens: int


class GeminiUsageMetadata(BaseModel):
    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    thoughtsTokenCount: int = 0
    totalTokenCount: int = 0


class GeminiFunctionCallPayload(BaseModel):
    name: str
    args: Any | None = None


class GeminiFunctionResponsePayload(BaseModel):
    name: str
    response: Any | None = None


class GeminiInlineDataResponse(BaseModel):
    mimeType: str
    data: str


class GeminiPartResponse(BaseModel):
    text: str | None = None
    thought: bool | None = None
    inlineData: GeminiInlineDataResponse | None = None
    thoughtSignature: str | None = None
    functionCall: GeminiFunctionCallPayload | None = None
    functionResponse: GeminiFunctionResponsePayload | None = None


class GeminiContentResponse(BaseModel):
    role: Literal["model"] = "model"
    parts: list[GeminiPartResponse]


class GeminiCandidateResponse(BaseModel):
    content: GeminiContentResponse
    finishReason: str | None = None


class GeminiGenerateContentResponse(BaseModel):
    candidates: list[GeminiCandidateResponse]
    usageMetadata: GeminiUsageMetadata | None = None


class HealthResponse(BaseModel):
    status: str
    busy: bool


class ModelStatsResponse(BaseModel):
    requests: int
    success: int
    rate_limited: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    last_used: str | None = None


class StatsTotalsResponse(BaseModel):
    requests: int
    success: int
    rate_limited: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StatsResponse(BaseModel):
    models: dict[str, ModelStatsResponse]
    totals: StatsTotalsResponse


class ModelCardResponse(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCardResponse]

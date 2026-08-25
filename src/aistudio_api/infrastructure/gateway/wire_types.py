"""Typed representations for the reverse-engineered AI Studio wire body."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ThinkingLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MINIMAL = 4


class MediaResolution(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class ImageOutputType(IntEnum):
    IMAGE = 2


@dataclass(frozen=True)
class AistudioThinkingConfig:
    level: ThinkingLevel = ThinkingLevel.HIGH
    mode: int = 1

    def to_wire(self) -> list:
        return [self.mode, None, None, int(self.level)]

    @classmethod
    def default(cls) -> "AistudioThinkingConfig":
        return cls()


@dataclass(frozen=True)
class AistudioImageOutputMode:
    output_type: ImageOutputType = ImageOutputType.IMAGE
    include_text: bool = False

    def to_wire(self) -> list[int]:
        if self.include_text:
            return [int(self.output_type), 1]
        return [int(self.output_type)]

    @classmethod
    def image_only(cls) -> "AistudioImageOutputMode":
        return cls(include_text=False)

    @classmethod
    def text_and_image(cls) -> "AistudioImageOutputMode":
        return cls(include_text=True)


@dataclass
class AistudioGenerationConfig:
    values: list = field(default_factory=list)

    @property
    def stop_sequences(self):
        return self.values[1] if len(self.values) > 1 else None

    @stop_sequences.setter
    def stop_sequences(self, value):
        self._ensure_len(2)
        self.values[1] = value

    @property
    def max_tokens(self):
        return self.values[3] if len(self.values) > 3 else None

    @max_tokens.setter
    def max_tokens(self, value):
        self._ensure_len(4)
        self.values[3] = value

    @property
    def temperature(self):
        return self.values[4] if len(self.values) > 4 else None

    @temperature.setter
    def temperature(self, value):
        self._ensure_len(5)
        self.values[4] = value

    @property
    def top_p(self):
        return self.values[5] if len(self.values) > 5 else None

    @top_p.setter
    def top_p(self, value):
        self._ensure_len(6)
        self.values[5] = value

    @property
    def top_k(self):
        return self.values[6] if len(self.values) > 6 else None

    @top_k.setter
    def top_k(self, value):
        self._ensure_len(7)
        self.values[6] = value

    @property
    def response_mime_type(self):
        return self.values[7] if len(self.values) > 7 else None

    @response_mime_type.setter
    def response_mime_type(self, value):
        self._ensure_len(8)
        self.values[7] = value

    @property
    def response_schema(self):
        return self.values[8] if len(self.values) > 8 else None

    @response_schema.setter
    def response_schema(self, value):
        self._ensure_len(9)
        self.values[8] = value

    @property
    def presence_penalty(self):
        return self.values[9] if len(self.values) > 9 else None

    @presence_penalty.setter
    def presence_penalty(self, value):
        self._ensure_len(10)
        self.values[9] = value

    @property
    def frequency_penalty(self):
        return self.values[10] if len(self.values) > 10 else None

    @frequency_penalty.setter
    def frequency_penalty(self, value):
        self._ensure_len(11)
        self.values[10] = value

    @property
    def response_logprobs(self):
        return self.values[11] if len(self.values) > 11 else None

    @response_logprobs.setter
    def response_logprobs(self, value):
        self._ensure_len(12)
        self.values[11] = value

    @property
    def logprobs(self):
        return self.values[12] if len(self.values) > 12 else None

    @logprobs.setter
    def logprobs(self, value):
        self._ensure_len(13)
        self.values[12] = value

    @property
    def image_output_mode(self):
        return self.values[14] if len(self.values) > 14 else None

    @image_output_mode.setter
    def image_output_mode(self, value):
        self._ensure_len(15)
        if isinstance(value, AistudioImageOutputMode):
            value = value.to_wire()
        self.values[14] = value

    @property
    def thinking_config(self):
        return self.values[16] if len(self.values) > 16 else None

    @thinking_config.setter
    def thinking_config(self, value):
        self._ensure_len(17)
        self.values[16] = value

    @property
    def media_resolution(self):
        return self.values[17] if len(self.values) > 17 else None

    @media_resolution.setter
    def media_resolution(self, value):
        self._ensure_len(18)
        if isinstance(value, MediaResolution):
            value = int(value)
        self.values[17] = value

    @property
    def output_resolution(self):
        return self.values[26] if len(self.values) > 26 else None

    @output_resolution.setter
    def output_resolution(self, value):
        self._ensure_len(27)
        self.values[26] = value

    def clear_gemma_thinking_budget(self):
        if len(self.values) > 16:
            self.values[16] = None

    def enable_default_thinking(self):
        if self.thinking_config is None:
            self.thinking_config = AistudioThinkingConfig.default().to_wire()
        # Match the AI Studio frontend request: index 13 = 1 enables thought
        # summaries, while an explicit response_mime_type ("text/plain")
        # suppresses them on Gemini models.
        self._ensure_len(14)
        if self.values[13] is None:
            self.values[13] = 1
        if len(self.values) > 7 and self.values[7] == "text/plain":
            self.values[7] = None

    def sanitize_for_plain_text(self):
        self.response_mime_type = "text/plain"
        self.response_schema = None
        self.thinking_config = None

    def _ensure_len(self, size: int):
        while len(self.values) < size:
            self.values.append(None)


@dataclass
class AistudioPart:
    text: str | None = None
    inline_data: tuple[str, str] | None = None
    file_id: str | None = None
    function_call: tuple[str, object] | tuple[str, object, str] | None = None
    function_response: tuple[str, object] | tuple[str, object, str] | None = None
    thought_signature: str | None = None
    thought: bool = False

    def to_wire(self):
        if self.file_id:
            return [None, None, None, None, None, [self.file_id]]
        if self.inline_data:
            mime, b64 = self.inline_data
            part = [None, None, [mime, b64]]
            if self.thought_signature:
                while len(part) <= 14:
                    part.append(None)
                part[14] = self.thought_signature
            return part
        if self.function_call:
            name, args = self.function_call[0], self.function_call[1]
            call_id = self.function_call[2] if len(self.function_call) > 2 else None
            function_call = [name, _encode_wire_args(args)]
            if call_id:
                function_call.append(call_id)
            part = [None] * 11
            part[10] = function_call
            if self.thought_signature:
                while len(part) <= 14:
                    part.append(None)
                part[14] = self.thought_signature
            return part
        if self.function_response:
            name, response = self.function_response[0], self.function_response[1]
            call_id = self.function_response[2] if len(self.function_response) > 2 else None
            function_response = [name, _encode_wire_args(response)]
            if call_id:
                function_response.append(call_id)
            part = [None] * 12
            part[11] = function_response
            return part
        # Text part — mark as thinking when thought=True (wire index 12 = 1).
        if self.thought:
            part: list = [None, self.text]
            while len(part) <= 12:
                part.append(None)
            part[12] = 1
            return part
        return [None, self.text]


def _encode_wire_args(value):
    if isinstance(value, dict):
        return [[[key, _encode_wire_value(val)] for key, val in value.items()]]
    return value


def _encode_wire_value(value):
    if isinstance(value, dict):
        return [None, _encode_wire_args(value)]
    if isinstance(value, list):
        return [None, None, [_encode_wire_value(item) for item in value]]
    return [None, None, value]


@dataclass
class AistudioContent:
    role: str
    parts: list[AistudioPart]

    def to_wire(self):
        return [[part.to_wire() for part in self.parts], self.role]


@dataclass
class AistudioRequest:
    model: str
    contents: list[AistudioContent]
    safety_settings: list | None
    generation_config: AistudioGenerationConfig
    snapshot: str | None
    system_instruction: AistudioContent | None
    tools: list[list] | None
    request_flag: int | None = None
    cached_content: str | None = None
    location: list | None = None
    raw_body: list = field(default_factory=list)

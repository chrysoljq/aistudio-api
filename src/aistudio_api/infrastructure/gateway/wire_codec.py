"""Codec between typed request objects and reverse-engineered AI Studio wire arrays."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import replace

from aistudio_api.config import DEFAULT_TEXT_MODEL

from .model_defaults import resolve_model_defaults
from .wire_types import AistudioContent, AistudioGenerationConfig, AistudioPart, AistudioRequest

TOOLS_TEMPLATES = {
    "code_execution": [[]],
    "google_search": [None, None, None, [None, [[]]]],
    "google_maps": [None, None, None, None, None, None, None, None, None, None, []],
    "url_context": [None, None, None, None, None, None, None, []],
}


def build_image_generation_search_tool(*, google_search: bool = False, image_search: bool = False) -> list | None:
    if not google_search and not image_search:
        return None
    if google_search and image_search:
        return [None, None, None, [None, [[], []]]]
    if google_search:
        return TOOLS_TEMPLATES["google_search"]
    return [None, None, None, [None, [None, []]]]


def _normalize_tool_name(raw_name: str, *, is_image_model: bool = False) -> str:
    name = str(raw_name).strip().lower()
    if is_image_model:
        if name in {"google_search_and_image_search", "image_google_search_and_image_search"}:
            return "google_search_and_image_search"
        if name in {"image_search", "google_image_search"}:
            return "image_search"
        if name in {"google_search", "image_google_search"}:
            return "google_search"
    return name


def _allowed_builtin_tools_for_model(model: str | None, *, is_image_model: bool = False) -> set[str]:
    if is_image_model:
        return {"google_search", "image_search", "google_search_and_image_search"}
    normalized_model = str(model or "").removeprefix("models/").lower()
    if normalized_model.startswith("gemma-"):
        return {"google_search", "code_execution"}
    if normalized_model.startswith("gemini-"):
        return {"google_search", "code_execution", "google_maps", "url_context"}
    return {"google_search", "code_execution"}


def build_tools_from_names(
    tool_names: list[str] | tuple[str, ...],
    *,
    model: str | None = None,
    is_image_model: bool = False,
) -> list[list]:
    allowed = _allowed_builtin_tools_for_model(model, is_image_model=is_image_model)
    if is_image_model:
        google_search = False
        image_search = False
        for raw_name in tool_names:
            name = _normalize_tool_name(raw_name, is_image_model=True)
            if not name:
                continue
            if name not in allowed:
                raise ValueError(f"Tool {raw_name!r} is not allowed for model {model or 'unknown'}")
            if name == "google_search_and_image_search":
                google_search = True
                image_search = True
            elif name == "google_search":
                google_search = True
            elif name == "image_search":
                image_search = True
        tool = build_image_generation_search_tool(
            google_search=google_search,
            image_search=image_search,
        )
        return [tool] if tool is not None else []

    tools: list[list] = []
    for raw_name in tool_names:
        name = _normalize_tool_name(raw_name, is_image_model=False)
        if not name:
            continue
        if name not in allowed:
            raise ValueError(f"Tool {raw_name!r} is not allowed for model {model or 'unknown'}")
        if name not in TOOLS_TEMPLATES:
            raise ValueError(f"Unsupported tool name: {raw_name!r}")
        tools.append(TOOLS_TEMPLATES[name])
    return tools


def _encode_image(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as file:
        data = file.read()
    return mime, base64.b64encode(data).decode("ascii")


class AistudioWireCodec:
    MODEL_INDEX = 0
    CONTENTS_INDEX = 1
    SAFETY_INDEX = 2
    GENERATION_CONFIG_INDEX = 3
    SNAPSHOT_INDEX = 4
    SYSTEM_INSTRUCTION_INDEX = 5
    TOOLS_INDEX = 6
    REQUEST_FLAG_INDEX = 10
    CACHED_CONTENT_INDEX = 11
    TIMEZONE_INDEX = 13

    def decode(self, raw_body: str) -> AistudioRequest:
        body = json.loads(raw_body)
        return AistudioRequest(
            model=body[self.MODEL_INDEX],
            contents=self._decode_contents(body[self.CONTENTS_INDEX]),
            safety_settings=body[self.SAFETY_INDEX],
            generation_config=AistudioGenerationConfig(list(body[self.GENERATION_CONFIG_INDEX])),
            snapshot=body[self.SNAPSHOT_INDEX] if len(body) > self.SNAPSHOT_INDEX else None,
            system_instruction=self._decode_system_instruction(body[self.SYSTEM_INSTRUCTION_INDEX] if len(body) > self.SYSTEM_INSTRUCTION_INDEX else None),
            tools=body[self.TOOLS_INDEX] if len(body) > self.TOOLS_INDEX else None,
            request_flag=body[self.REQUEST_FLAG_INDEX] if len(body) > self.REQUEST_FLAG_INDEX else None,
            cached_content=body[self.CACHED_CONTENT_INDEX] if len(body) > self.CACHED_CONTENT_INDEX else None,
            location=body[self.TIMEZONE_INDEX] if len(body) > self.TIMEZONE_INDEX else None,
            raw_body=body,
        )

    def encode(self, request: AistudioRequest) -> str:
        body = list(request.raw_body)
        self._ensure_len(body, self.TIMEZONE_INDEX + 1)

        model = request.model
        if not model.startswith("models/"):
            model = f"models/{model}"
        body[self.MODEL_INDEX] = model
        body[self.CONTENTS_INDEX] = [content.to_wire() for content in request.contents]
        body[self.SAFETY_INDEX] = request.safety_settings
        body[self.GENERATION_CONFIG_INDEX] = request.generation_config.values
        body[self.SNAPSHOT_INDEX] = request.snapshot
        body[self.SYSTEM_INSTRUCTION_INDEX] = request.system_instruction.to_wire() if request.system_instruction else None
        body[self.TOOLS_INDEX] = request.tools
        body[self.REQUEST_FLAG_INDEX] = request.request_flag
        body[self.CACHED_CONTENT_INDEX] = request.cached_content
        body[self.TIMEZONE_INDEX] = request.location

        model_defaults = resolve_model_defaults(request.model)
        if model_defaults.is_image_model:
            gc = body[self.GENERATION_CONFIG_INDEX]
            if isinstance(gc, list):
                # Different image-capable models reject some text-model carry-over slots.
                for idx in model_defaults.clear_generation_config_indexes:
                    if idx < len(gc):
                        gc[idx] = None
            if model_defaults.disable_safety_settings:
                body[self.SAFETY_INDEX] = None
        else:
            if request.tools:
                self._ensure_len(body, self.TIMEZONE_INDEX + 1)
                body[self.TIMEZONE_INDEX] = request.location or [[None, None, "Asia/Shanghai"]]
            else:
                body = body[:11]

        return json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    def rewrite(
        self,
        original_body: str,
        model: str = DEFAULT_TEXT_MODEL,
        snapshot: str | None = None,
        prompt: str | None = None,
        contents: list[AistudioContent] | None = None,
        system_instruction: str | None = None,
        system_instruction_content: AistudioContent | None = None,
        tools: list[list] | None = None,
        safety_settings: list[list] | None = None,
        images: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        generation_config_overrides: dict | None = None,
        sanitize_plain_text: bool = True,
        safety_off: bool = False,
    ) -> str:
        request = self.decode(original_body)
        request.model = model
        model_defaults = resolve_model_defaults(model)
        if snapshot is not None:
            request.snapshot = snapshot

        if contents is not None:
            request.contents = contents
        elif prompt is not None:
            request.contents = [self._build_user_content(prompt=prompt, images=images)]

        if system_instruction_content is not None:
            request.system_instruction = system_instruction_content
        else:
            request.system_instruction = (
                AistudioContent(role="user", parts=[AistudioPart(text=system_instruction)])
                if system_instruction
                else None
            )

        defaults = model_defaults.generation_config_overrides()
        for attr, value in defaults.items():
            setattr(request.generation_config, attr, value)
        if model_defaults.safety_settings is not None:
            request.safety_settings = model_defaults.safety_settings_overrides()
        if safety_settings is not None:
            request.safety_settings = safety_settings

        if max_tokens is not None:
            request.generation_config.max_tokens = max_tokens
        if temperature is not None:
            request.generation_config.temperature = temperature
        if top_p is not None:
            request.generation_config.top_p = top_p
        if top_k is not None:
            request.generation_config.top_k = top_k
        explicit_overrides = generation_config_overrides or {}
        for attr, value in explicit_overrides.items():
            if value is None or not hasattr(request.generation_config, attr):
                continue
            setattr(request.generation_config, attr, value)

        # The captured browser body usually carries a HIGH thinking config. That is
        # inherited state, not a caller choice, so the default thinking level stays
        # ours to size against the caller's output budget. A thinking config set by
        # the model profile (image models use MINIMAL) or by the caller is explicit
        # and must survive untouched.
        thinking_is_explicit = (
            "thinking_config" in defaults or "thinking_config" in explicit_overrides
        )
        request.generation_config.enable_default_thinking(
            max_tokens, force=not thinking_is_explicit, model=model
        )

        # OpenAI chat compatibility should not inherit browser-side structured output
        # or explicit reasoning settings from a previously captured AI Studio request.
        if sanitize_plain_text and not model_defaults.is_image_model:
            request.generation_config.sanitize_for_plain_text()
            request.generation_config.enable_default_thinking(
                max_tokens, force=not thinking_is_explicit, model=model
            )

        if safety_off:
            request.safety_settings = [[None, None, cat, 5] for cat in [7, 8, 9, 10]]

        request.tools = tools if tools else None

        if not model_defaults.is_image_model:
            if request.tools:
                request.generation_config.response_mime_type = None
                request.generation_config.response_schema = None

        return self.encode(request)

    def _build_user_content(self, prompt: str, images: list[str] | None) -> AistudioContent:
        parts = []
        for img_path in images or []:
            parts.append(AistudioPart(inline_data=_encode_image(img_path)))
        parts.append(AistudioPart(text=prompt))
        return AistudioContent(role="user", parts=parts)

    def _decode_contents(self, raw_contents) -> list[AistudioContent]:
        contents = []
        if not isinstance(raw_contents, list):
            return contents
        for item in raw_contents:
            if not isinstance(item, list) or len(item) < 2:
                continue
            raw_parts, role = item[0], item[1]
            parts = []
            if isinstance(raw_parts, list):
                for raw_part in raw_parts:
                    parts.append(self._decode_part(raw_part))
            contents.append(AistudioContent(role=role, parts=parts))
        return contents

    def _decode_part(self, raw_part) -> AistudioPart:
        if (
            isinstance(raw_part, list)
            and len(raw_part) > 5
            and isinstance(raw_part[5], list)
            and len(raw_part[5]) >= 1
            and isinstance(raw_part[5][0], str)
        ):
            return AistudioPart(file_id=raw_part[5][0])
        if (
            isinstance(raw_part, list)
            and len(raw_part) > 2
            and isinstance(raw_part[2], list)
            and len(raw_part[2]) >= 2
        ):
            return AistudioPart(inline_data=(raw_part[2][0], raw_part[2][1]))
        if isinstance(raw_part, list):
            raw_function_call = None
            if len(raw_part) > 10 and isinstance(raw_part[10], list):
                raw_function_call = raw_part[10]
            elif len(raw_part) > 3 and isinstance(raw_part[3], list):
                raw_function_call = raw_part[3]
            if raw_function_call is not None:
                name = raw_function_call[0] if raw_function_call and isinstance(raw_function_call[0], str) else "unknown"
                args = raw_function_call[1] if len(raw_function_call) > 1 else {}
                call_id = raw_function_call[2] if len(raw_function_call) > 2 and isinstance(raw_function_call[2], str) else None
                signature = raw_part[14] if len(raw_part) > 14 and isinstance(raw_part[14], str) else None
                function_call = (name, args, call_id) if call_id else (name, args)
                return AistudioPart(function_call=function_call, thought_signature=signature)
        if isinstance(raw_part, list) and (
            len(raw_part) > 11 and isinstance(raw_part[11], list) or len(raw_part) > 4 and isinstance(raw_part[4], list)
        ):
            raw_function_response = raw_part[11] if len(raw_part) > 11 and isinstance(raw_part[11], list) else raw_part[4]
            name = raw_function_response[0] if raw_function_response and isinstance(raw_function_response[0], str) else "unknown"
            response = raw_function_response[1] if len(raw_function_response) > 1 else {}
            return AistudioPart(function_response=(name, response))
        if isinstance(raw_part, list) and len(raw_part) > 1:
            return AistudioPart(text=raw_part[1])
        return AistudioPart()

    def _decode_system_instruction(self, raw_instruction) -> AistudioContent | None:
        if not raw_instruction or not isinstance(raw_instruction, list):
            return None
        if len(raw_instruction) >= 2 and isinstance(raw_instruction[1], str):
            decoded = self._decode_contents([raw_instruction])
            return decoded[0] if decoded else None
        parts = [self._decode_part(raw_part) for raw_part in raw_instruction if isinstance(raw_part, list)]
        if not parts:
            return None
        return AistudioContent(role="user", parts=parts)

    def _ensure_len(self, body: list, size: int):
        while len(body) < size:
            body.append(None)


_codec = AistudioWireCodec()


def modify_body(
    original_body: str,
    model: str = DEFAULT_TEXT_MODEL,
    snapshot: str | None = None,
    prompt: str | None = None,
    contents: list[AistudioContent] | None = None,
    system_instruction: str | None = None,
    system_instruction_content: AistudioContent | None = None,
    tools: list[list] | None = None,
    safety_settings: list[list] | None = None,
    images: list[str] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    generation_config_overrides: dict | None = None,
    sanitize_plain_text: bool = True,
    safety_off: bool = False,
) -> str:
    return _codec.rewrite(
        original_body=original_body,
        model=model,
        snapshot=snapshot,
        prompt=prompt,
        contents=contents,
        system_instruction=system_instruction,
        system_instruction_content=system_instruction_content,
        tools=tools,
        safety_settings=safety_settings,
        images=images,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        generation_config_overrides=generation_config_overrides,
        sanitize_plain_text=sanitize_plain_text,
        safety_off=safety_off,
    )

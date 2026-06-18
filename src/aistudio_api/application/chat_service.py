"""Application services for chat/image orchestration."""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from typing import Any, Optional

import httpx

from aistudio_api.config import DEFAULT_IMAGE_MODEL
from aistudio_api.domain.errors import RequestError
from aistudio_api.infrastructure.gateway.model_defaults import resolve_model_defaults
from aistudio_api.infrastructure.gateway.request_rewriter import build_tools_from_names
from aistudio_api.infrastructure.gateway.wire_types import (
    AistudioContent,
    AistudioImageOutputMode,
    AistudioPart,
    AistudioThinkingConfig,
    ThinkingLevel,
)
from aistudio_api.infrastructure.gateway.wire_codec import TOOLS_TEMPLATES


SCHEMA_TYPE_CODES = {
    "string": 1,
    "number": 2,
    "integer": 3,
    "boolean": 4,
    "array": 5,
    "object": 6,
}

def data_uri_to_file(uri: str, tmp_dir: str = "/tmp") -> str:
    match = re.match(r"data:(.+?);base64,(.+)", uri, re.DOTALL)
    if not match:
        raise ValueError("Invalid data URI")
    mime, b64 = match.group(1), match.group(2)
    ext = mime.split("/")[-1].replace("jpeg", "jpg")
    path = os.path.join(tmp_dir, f"aistudio_img_{uuid.uuid4().hex[:8]}.{ext}")
    with open(path, "wb") as file:
        file.write(base64.b64decode(b64))
    return path


def url_to_file(url: str, tmp_dir: str = "/tmp") -> str:
    path = os.path.join(tmp_dir, f"aistudio_img_{uuid.uuid4().hex[:8]}.jpg")
    with httpx.Client(timeout=30) as http:
        resp = http.get(url)
        resp.raise_for_status()
        with open(path, "wb") as file:
            file.write(resp.content)
    return path


def normalize_chat_request(messages, requested_model: str, tmp_dir: str = "/tmp") -> dict:
    system_texts: list[str] = []
    contents: list[AistudioContent] = []
    capture_texts: list[str] = []
    capture_images: list[str] = []
    cleanup_paths: list[str] = []
    saw_images = False

    # Build tool_call_id → function name map from assistant tool_calls
    tool_call_names: dict[str, str] = {}
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id and tc.function and tc.function.name:
                    tool_call_names[tc.id] = tc.function.name

    for msg in messages:
        role = (msg.role or "user").lower()
        if role in ("system", "developer"):
            text = _message_text_content(msg.content)
            if text:
                system_texts.append(text)
                capture_texts.append(text)
            continue

        parts: list[AistudioPart] = []
        text_parts: list[str] = []
        image_paths: list[str] = []

        # tool result → user role with marker
        if role == "tool":
            raw = _message_text_content(msg.content) or ""
            fname = tool_call_names.get(msg.tool_call_id or "", "")
            tagged = f"<tool_result name=\"{fname}\">\n{raw}\n</tool_result>" if fname else f"<tool_result>\n{raw}\n</tool_result>"
            parts.append(AistudioPart(text=tagged))
            text_parts.append(tagged)
            contents.append(AistudioContent(role="user", parts=parts))
            capture_texts.append(raw)
            continue

        # OpenAI 兼容格式的 reasoning_content：思考内容作为首个 thought Part 传入
        if role == "assistant" and msg.reasoning_content:
            parts.append(AistudioPart(text=msg.reasoning_content, thought=True))

        if isinstance(msg.content, str):
            if msg.content:
                parts.append(AistudioPart(text=msg.content))
                text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "text" and part.text:
                    parts.append(AistudioPart(text=part.text))
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    url = part.image_url["url"] if isinstance(part.image_url, dict) else part.image_url.url
                    if url.startswith("data:"):
                        path = data_uri_to_file(url, tmp_dir=tmp_dir)
                        image_paths.append(path)
                        cleanup_paths.append(path)
                    elif url.startswith("http"):
                        path = url_to_file(url, tmp_dir=tmp_dir)
                        image_paths.append(path)
                        cleanup_paths.append(path)


        for image_path in image_paths:
            parts.append(_image_path_to_part(image_path))

        if not parts:
            continue

        mapped_role = "model" if role == "assistant" else "user"
        contents.append(AistudioContent(role=mapped_role, parts=parts))
        capture_texts.extend(text_parts)
        if image_paths:
            saw_images = True
            capture_images.extend(image_paths)

    capture_prompt = "\n".join(capture_texts) if capture_texts else "你好"
    model = requested_model
    if model.startswith("gpt-") or model.startswith("openai/"):
        model = DEFAULT_IMAGE_MODEL if saw_images else requested_model

    return {
        "model": model,
        "system_instruction": "\n".join(system_texts) if system_texts else None,
        "contents": contents or [AistudioContent(role="user", parts=[AistudioPart(text="你好")])],
        "capture_prompt": capture_prompt,
        "capture_images": capture_images,
        "cleanup_paths": cleanup_paths,
    }


def _message_text_content(content) -> str | None:
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        texts = [part.text for part in content if part.type == "text" and part.text]
        return "\n".join(texts) if texts else None
    return None


def _image_path_to_part(path: str) -> AistudioPart:
    mime = "image/jpeg"
    if path.endswith(".png"):
        mime = "image/png"
    elif path.endswith(".webp"):
        mime = "image/webp"
    with open(path, "rb") as file:
        return AistudioPart(inline_data=(mime, base64.b64encode(file.read()).decode("ascii")))


def cleanup_files(paths: list[str]):
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def inline_data_to_file(mime_type: str, data: str, tmp_dir: str = "/tmp") -> str:
    ext = mime_type.split("/")[-1].replace("jpeg", "jpg")
    path = os.path.join(tmp_dir, f"aistudio_img_{uuid.uuid4().hex[:8]}.{ext}")
    with open(path, "wb") as file:
        file.write(base64.b64decode(data))
    return path


def encode_schema_to_wire(schema: dict, *, include_required: bool = True) -> list:
    schema_type = schema.get("type")
    type_code = SCHEMA_TYPE_CODES.get(schema_type, 0)
    wire = [type_code]

    if schema_type == "array" and isinstance(schema.get("items"), dict):
        while len(wire) <= 5:
            wire.append(None)
        wire[5] = encode_schema_to_wire(schema["items"], include_required=include_required)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        while len(wire) <= 6:
            wire.append(None)
        wire[6] = [
            [name, encode_schema_to_wire(prop, include_required=include_required)]
            for name, prop in properties.items()
            if isinstance(prop, dict)
        ]

    required = schema.get("required")
    if include_required and isinstance(required, list):
        while len(wire) <= 7:
            wire.append(None)
        wire[7] = list(required)

    property_ordering = schema.get("propertyOrdering")
    if isinstance(property_ordering, list):
        while len(wire) <= 22:
            wire.append(None)
        wire[22] = list(property_ordering)

    return wire


def encode_function_declaration_to_wire(declaration: dict) -> list:
    if not declaration.get("name"):
        raise ValueError("functionDeclarations[].name is required")

    wire = [declaration["name"]]
    if declaration.get("description") is not None:
        while len(wire) <= 1:
            wire.append(None)
        wire[1] = declaration["description"]

    parameters = declaration.get("parameters")
    if isinstance(parameters, dict):
        while len(wire) <= 2:
            wire.append(None)
        wire[2] = encode_schema_to_wire(parameters, include_required=False)

    return wire


def _normalize_gemini_modalities(value: Any) -> AistudioImageOutputMode | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("generationConfig.responseModalities must be a list")

    modalities = {str(item).strip().upper() for item in value if str(item).strip()}
    if not modalities:
        return None
    unknown = modalities - {"TEXT", "IMAGE"}
    if unknown:
        raise ValueError(f"Unsupported response modalities: {', '.join(sorted(unknown))}")
    if "IMAGE" not in modalities:
        return None
    if "TEXT" in modalities:
        return AistudioImageOutputMode.text_and_image()
    return AistudioImageOutputMode.image_only()


def _normalize_gemini_thinking_config(value: Any) -> list[Any] | dict[str, Any]:
    if value is None or isinstance(value, list):
        return value
    if not isinstance(value, dict):
        raise ValueError("generationConfig.thinkingConfig must be an object or wire array")

    raw_level = value.get("thinkingLevel", value.get("level", ThinkingLevel.HIGH))
    raw_mode = value.get("mode", 1)
    if isinstance(raw_level, ThinkingLevel):
        level = raw_level
    elif isinstance(raw_level, int):
        level = ThinkingLevel(raw_level)
    else:
        level = ThinkingLevel[str(raw_level).strip().upper()]
    return AistudioThinkingConfig(level=level, mode=int(raw_mode)).to_wire()


def _normalize_gemini_image_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("generationConfig.imageConfig must be an object")

    aspect_ratio = value.get("aspectRatio")
    image_size = value.get("imageSize")
    if isinstance(aspect_ratio, str) and not aspect_ratio.strip():
        aspect_ratio = None
    if isinstance(image_size, str):
        image_size = image_size.strip() or None
    person_generation = value.get("personGeneration")
    if person_generation not in (None, ""):
        raise ValueError("generationConfig.imageConfig.personGeneration is not supported yet")

    normalized: dict[str, Any] = {}
    if aspect_ratio is not None or image_size is not None:
        normalized["output_resolution"] = [aspect_ratio, image_size]
    return normalized


def _extract_google_search_tool_names(tool: Any, *, is_image_model: bool) -> list[str]:
    if tool.googleSearchRetrieval is not None:
        return ["google_search"]

    config = tool.googleSearch
    if config is None:
        return []
    if not is_image_model or not isinstance(config, dict):
        return ["google_search"]

    search_types = config.get("searchTypes")
    if not isinstance(search_types, dict):
        return ["google_search"]

    web_enabled = search_types.get("webSearch") is not None
    image_enabled = search_types.get("imageSearch") is not None
    if web_enabled and image_enabled:
        return ["google_search_and_image_search"]
    if image_enabled:
        return ["image_search"]
    return ["google_search"]


def _filter_default_tools_for_model(tool_names: tuple[str, ...], *, is_image_model: bool) -> list[str]:
    names = [str(name).strip() for name in tool_names if str(name).strip()]
    if not is_image_model:
        return names
    allowed = {"google_search", "image_search", "google_search_and_image_search"}
    return [name for name in names if name in allowed]


_GEMINI_SAFETY_CATEGORY_MAP = {
    "HARM_CATEGORY_HARASSMENT": 7,
    "HARM_CATEGORY_HATE_SPEECH": 8,
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": 9,
    "HARM_CATEGORY_DANGEROUS_CONTENT": 10,
}

_GEMINI_SAFETY_THRESHOLD_MAP = {
    "BLOCK_LOW_AND_ABOVE": 1,
    "BLOCK_MEDIUM_AND_ABOVE": 2,
    "BLOCK_ONLY_HIGH": 3,
    "BLOCK_NONE": 4,
    "OFF": 5,
}


def _normalize_gemini_safety_settings(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("safetySettings must be a list")

    normalized: list[list[Any]] = []
    for item in value:
        if not hasattr(item, "category") or not hasattr(item, "threshold"):
            raise ValueError(f"Unsupported safety setting entry: {item!r}")

        category = _GEMINI_SAFETY_CATEGORY_MAP.get(str(item.category).strip().upper())
        if category is None:
            raise ValueError(f"Unsupported safety category: {item.category}")
        threshold = _GEMINI_SAFETY_THRESHOLD_MAP.get(str(item.threshold).strip().upper())
        if threshold is None:
            raise ValueError(f"Unsupported safety threshold: {item.threshold}")
        normalized.append([None, None, category, threshold])
    return normalized


def normalize_openai_tools(tools) -> list[list] | None:
    if not tools:
        return None

    function_declarations: list[dict] = []
    for tool in tools:
        if tool.type != "function":
            raise ValueError(f"unsupported tool type: {tool.type}")
        if tool.function is None:
            raise ValueError("tools[].function is required when type=function")

        function_declarations.append(
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            }
        )

    if not function_declarations:
        return None

    return [[None, [encode_function_declaration_to_wire(decl) for decl in function_declarations]]]


ANTHROPIC_TOOL_HISTORY_INSTRUCTION = (
    "The conversation may include INTERNAL TOOL HISTORY records converted from "
    "Anthropic tool_use/tool_result blocks. Treat those records only as prior "
    "tool-call context. Do not quote, repeat, translate, or expose their wrapper "
    "text in the final answer. Use the recorded tool results to answer the user's "
    "actual request."
)

ANTHROPIC_TOOL_HISTORY_START = "<internal_anthropic_tool_history>"
ANTHROPIC_TOOL_HISTORY_END = "</internal_anthropic_tool_history>"


def normalize_anthropic_request(req, tmp_dir: str = "/tmp", tool_context: dict[str, dict] | None = None) -> dict:
    system_text = _anthropic_system_text(req.system)
    contents: list[AistudioContent] = []
    capture_texts: list[str] = [system_text] if system_text else []
    capture_images: list[str] = []
    cleanup_paths: list[str] = []
    pending_tool_parts: list[AistudioPart] = []
    tool_id_to_name = _anthropic_tool_id_name_map(req.messages)
    tool_context = tool_context or {}

    def flush_tool_parts():
        if pending_tool_parts:
            contents.append(AistudioContent(role="user", parts=list(pending_tool_parts)))
            pending_tool_parts.clear()

    for message in req.messages:
        role = (message.role or "user").lower()
        content = message.content

        if role == "user" and isinstance(content, list):
            tool_results = [block for block in content if block.type == "tool_result"]
            other_blocks = [block for block in content if block.type != "tool_result"]
            for block in tool_results:
                tool_info = _anthropic_tool_context(block.tool_use_id, tool_context)
                function_name = (
                    tool_info.get("name")
                    or tool_id_to_name.get(block.tool_use_id or "")
                    or block.name
                    or "unknown_function"
                )
                capture_text = _anthropic_tool_result_text(block.content)
                text = capture_text or json.dumps(_anthropic_tool_result_response(block.content), ensure_ascii=False)
                pending_tool_parts.append(
                    AistudioPart(text=_anthropic_tool_result_transcript(function_name, text))
                )
                if capture_text:
                    capture_texts.append(capture_text)
            if tool_results and not other_blocks:
                continue
            if tool_results:
                flush_tool_parts()
                content = other_blocks
        else:
            flush_tool_parts()

        parts: list[AistudioPart] = []
        text_parts: list[str] = []
        image_paths: list[str] = []

        if isinstance(content, str):
            if content:
                parts.append(AistudioPart(text=content))
                text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.type == "thinking" and block.thinking:
                    # Anthropic 思考 block：作为 thought=True 的文本 Part 传递，
                    # 不加入 capture_texts（避免影响 snapshot 捕获）
                    parts.append(AistudioPart(text=block.thinking, thought=True))
                elif block.type == "text" and block.text:
                    parts.append(AistudioPart(text=block.text))
                    text_parts.append(block.text)
                elif block.type == "image" and block.source:
                    image_path = _anthropic_image_source_to_file(block.source, tmp_dir=tmp_dir)
                    if image_path:
                        image_paths.append(image_path)
                        cleanup_paths.append(image_path)
                elif role == "assistant" and block.type == "tool_use" and block.name:
                    # AI Studio rejects replayed historical tool calls. The matching
                    # tool_result carries the useful context without exposing inputs.
                    continue

        for image_path in image_paths:
            parts.append(_image_path_to_part(image_path))

        if not parts:
            continue

        mapped_role = "model" if role == "assistant" else "user"
        contents.append(AistudioContent(role=mapped_role, parts=parts))
        capture_texts.extend(text_parts)
        capture_images.extend(image_paths)

    flush_tool_parts()
    capture_prompt = "\n".join(text for text in capture_texts if text) or "你好"
    model = req.model

    system_instruction_text = _anthropic_system_instruction(system_text, contents)

    return {
        "model": model,
        "system_instruction": (
            AistudioContent(role="user", parts=[AistudioPart(text=system_instruction_text)])
            if system_instruction_text
            else None
        ),
        "contents": contents or [AistudioContent(role="user", parts=[AistudioPart(text="你好")])],
        "capture_prompt": capture_prompt,
        "capture_images": capture_images or None,
        "cleanup_paths": cleanup_paths,
        "tools": normalize_anthropic_tools(req.tools, req.tool_choice),
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "max_tokens": req.max_tokens,
    }


def normalize_anthropic_tools(tools, tool_choice=None) -> list[list] | None:
    if not tools or (tool_choice and getattr(tool_choice, "type", None) == "none"):
        return None

    wire_tools: list[list] = []
    function_declarations: list[dict[str, Any]] = []

    for tool in tools:
        tool_type = (tool.type or "").lower()
        name = tool.name or ""
        if tool_type.startswith("web_search") or name == "web_search":
            wire_tools.append(TOOLS_TEMPLATES["google_search"])
            continue
        if not name:
            continue
        declaration = {"name": name, "description": tool.description}
        if tool.input_schema:
            declaration["parameters"] = _sanitize_schema_for_wire(tool.input_schema)
        else:
            declaration["parameters"] = {"type": "object", "properties": {}}
        function_declarations.append(declaration)

    if function_declarations:
        wire_tools.insert(0, [None, [encode_function_declaration_to_wire(decl) for decl in function_declarations]])

    return wire_tools or None


def _anthropic_system_text(system) -> str | None:
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        texts: list[str] = []
        for item in system:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    texts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    texts.append(str(text))
        return "\n".join(texts) if texts else None
    return None


def _anthropic_tool_id_name_map(messages) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for message in messages:
        if (message.role or "").lower() != "assistant" or not isinstance(message.content, list):
            continue
        for block in message.content:
            if block.type == "tool_use" and block.id and block.name:
                mapping[block.id] = block.name
    return mapping


def _anthropic_tool_context(tool_use_id: str | None, tool_context: dict[str, dict]) -> dict[str, Any]:
    if not tool_use_id:
        return {}
    context = tool_context.get(tool_use_id)
    return context if isinstance(context, dict) else {}


def _anthropic_system_instruction(system_text: str | None, contents: list[AistudioContent]) -> str | None:
    has_tool_history = any(
        part.text and ANTHROPIC_TOOL_HISTORY_START in part.text
        for content in contents
        for part in content.parts
    )
    if not has_tool_history:
        return system_text
    if system_text:
        return f"{system_text}\n\n{ANTHROPIC_TOOL_HISTORY_INSTRUCTION}"
    return ANTHROPIC_TOOL_HISTORY_INSTRUCTION


def _anthropic_tool_result_transcript(function_name: str, text: str) -> str:
    return (
        f"{ANTHROPIC_TOOL_HISTORY_START}\n"
        f"user tool_result for: {function_name}\n"
        f"tool_result content:\n{text}\n"
        f"{ANTHROPIC_TOOL_HISTORY_END}"
    )


def _anthropic_image_source_to_file(source: dict[str, Any], tmp_dir: str = "/tmp") -> str | None:
    source_type = source.get("type")
    if source_type == "base64" and source.get("media_type") and source.get("data"):
        return inline_data_to_file(source["media_type"], source["data"], tmp_dir=tmp_dir)
    if source_type == "url" and source.get("url"):
        return url_to_file(source["url"], tmp_dir=tmp_dir)
    return None


def _anthropic_tool_result_response(content) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = _anthropic_tool_result_text(content)
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    if content is None:
        return {"result": ""}
    return {"result": content}


def _anthropic_tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    texts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    texts.append(str(text))
        return "\n".join(texts)
    return ""


def _sanitize_schema_for_wire(schema: dict | None) -> dict:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list):
        for variant in variants:
            if isinstance(variant, dict) and variant.get("type") != "null":
                return _sanitize_schema_for_wire(variant)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)
    if schema_type not in SCHEMA_TYPE_CODES:
        if isinstance(schema.get("properties"), dict):
            schema_type = "object"
        elif isinstance(schema.get("items"), dict):
            schema_type = "array"
        else:
            schema_type = "string"

    sanitized: dict[str, Any] = {"type": schema_type}
    if schema_type == "object":
        properties: dict[str, Any] = {}
        raw_properties = schema.get("properties")
        if isinstance(raw_properties, dict):
            for name, prop in raw_properties.items():
                if isinstance(name, str):
                    properties[name] = _sanitize_schema_for_wire(prop if isinstance(prop, dict) else None)
        sanitized["properties"] = properties
        required = schema.get("required")
        if isinstance(required, list):
            sanitized["required"] = [name for name in required if isinstance(name, str) and name in properties]
    elif schema_type == "array":
        sanitized["items"] = _sanitize_schema_for_wire(schema.get("items") if isinstance(schema.get("items"), dict) else None)
    return sanitized


def normalize_gemini_request(req, requested_model: str, tmp_dir: str = "/tmp") -> dict:
    if not req.contents:
        raise ValueError("contents is required")

    model = requested_model if requested_model.startswith("models/") else f"models/{requested_model}"
    contents: list[AistudioContent] = []
    cleanup_paths: list[str] = []
    capture_prompt = "你好"
    capture_images: list[str] = []

    for content in req.contents:
        role = content.role or "user"
        parts: list[AistudioPart] = []
        text_parts: list[str] = []
        content_images: list[str] = []

        # 预先统计文本 Part 的位置索引，用于推断多 Part model 消息中的思考内容。
        # 约定：model 角色有 2 个及以上纯文本 Part 时，最后一个是正式回答，
        # 其余全是思考内容——即使客户端没有传 thought=true 字段。
        text_part_positions = [
            i for i, p in enumerate(content.parts) if p.text is not None
        ]
        infer_thinking = role == "model" and len(text_part_positions) >= 2

        for idx, part in enumerate(content.parts):
            if part.text is not None:
                # 显式 thought 字段优先；否则对 model 多文本 Part 按位置推断
                is_thought = bool(part.thought) or (
                    infer_thinking and idx != text_part_positions[-1]
                )
                parts.append(AistudioPart(
                    text=part.text,
                    thought=is_thought,
                ))
                text_parts.append(part.text)
                continue
            if part.inlineData is not None:
                parts.append(
                    AistudioPart(
                        inline_data=(part.inlineData.mimeType, part.inlineData.data),
                        thought_signature=part.thoughtSignature,
                    )
                )
                image_path = inline_data_to_file(part.inlineData.mimeType, part.inlineData.data, tmp_dir=tmp_dir)
                content_images.append(image_path)
                cleanup_paths.append(image_path)
                continue
            if part.fileData is not None:
                raise ValueError("fileData is not supported yet")

        contents.append(AistudioContent(role=role, parts=parts))

        if role == "user":
            if text_parts:
                capture_prompt = "\n".join(text_parts)
            if content_images:
                capture_images = content_images

    system_instruction = None
    if req.systemInstruction is not None:
        system_instruction = AistudioContent(
            role=req.systemInstruction.role or "user",
            parts=[
                AistudioPart(text=part.text)
                if part.text is not None
                else AistudioPart(
                    inline_data=(part.inlineData.mimeType, part.inlineData.data),
                    thought_signature=part.thoughtSignature,
                )
                for part in req.systemInstruction.parts
                if part.text is not None or part.inlineData is not None
            ],
        )

    model_defaults = resolve_model_defaults(model)
    tools = None
    if req.tools is not None:
        tools = []
        for tool in req.tools:
            builtin_tool_names: list[str] = []
            if tool.codeExecution is not None:
                builtin_tool_names.append("code_execution")
            if tool.functionDeclarations:
                tools.append([None, [encode_function_declaration_to_wire(decl) for decl in tool.functionDeclarations]])
            builtin_tool_names.extend(
                _extract_google_search_tool_names(tool, is_image_model=model_defaults.is_image_model)
            )
            if tool.googleMaps is not None:
                builtin_tool_names.append("google_maps")
            if tool.urlContext is not None:
                builtin_tool_names.append("url_context")
            if builtin_tool_names:
                tools.extend(
                    build_tools_from_names(
                        builtin_tool_names,
                        model=model,
                        is_image_model=model_defaults.is_image_model,
                    )
                )

    if req.tools is None and model_defaults.default_tools:
        default_tool_names = _filter_default_tools_for_model(
            model_defaults.default_tools,
            is_image_model=model_defaults.is_image_model,
        )
        tools = (
            build_tools_from_names(
                default_tool_names,
                model=model,
                is_image_model=model_defaults.is_image_model,
            )
            if default_tool_names
            else []
        )

    generation_config = req.generationConfig
    generation_config_overrides = {
        key: value
        for key, value in model_defaults.generation_config_overrides().items()
        if value is not None
    } or None
    if generation_config is not None:
        if generation_config_overrides is None:
            generation_config_overrides = {}
        if generation_config.stopSequences is not None:
            generation_config_overrides["stop_sequences"] = generation_config.stopSequences
        if generation_config.maxOutputTokens is not None:
            generation_config_overrides["max_tokens"] = generation_config.maxOutputTokens
        if generation_config.temperature is not None:
            generation_config_overrides["temperature"] = generation_config.temperature
        if generation_config.topP is not None:
            generation_config_overrides["top_p"] = generation_config.topP
        if generation_config.topK is not None:
            generation_config_overrides["top_k"] = generation_config.topK
        if generation_config.responseMimeType is not None:
            generation_config_overrides["response_mime_type"] = generation_config.responseMimeType
        if generation_config.responseSchema is not None:
            generation_config_overrides["response_schema"] = (
                encode_schema_to_wire(generation_config.responseSchema)
                if isinstance(generation_config.responseSchema, dict)
                else generation_config.responseSchema
            )
        if generation_config.presencePenalty is not None:
            generation_config_overrides["presence_penalty"] = generation_config.presencePenalty
        if generation_config.frequencyPenalty is not None:
            generation_config_overrides["frequency_penalty"] = generation_config.frequencyPenalty
        if generation_config.responseLogprobs is not None:
            generation_config_overrides["response_logprobs"] = generation_config.responseLogprobs
        if generation_config.logprobs is not None:
            generation_config_overrides["logprobs"] = generation_config.logprobs
        if generation_config.mediaResolution is not None:
            generation_config_overrides["media_resolution"] = generation_config.mediaResolution
        if generation_config.thinkingConfig is not None:
            generation_config_overrides["thinking_config"] = _normalize_gemini_thinking_config(
                generation_config.thinkingConfig
            )
        if generation_config.responseModalities is not None:
            image_output_mode = _normalize_gemini_modalities(generation_config.responseModalities)
            if image_output_mode is not None:
                generation_config_overrides["image_output_mode"] = image_output_mode
        if generation_config.imageConfig is not None:
            generation_config_overrides.update(_normalize_gemini_image_config(generation_config.imageConfig))

    return {
        "model": model,
        "contents": contents,
        "system_instruction": system_instruction,
        "tools": tools if tools is not None else None,
        "safety_settings": _normalize_gemini_safety_settings(req.safetySettings) if req.safetySettings is not None else None,
        "capture_prompt": capture_prompt,
        "capture_images": capture_images or None,
        "cleanup_paths": cleanup_paths,
        "temperature": generation_config.temperature if generation_config else None,
        "top_p": generation_config.topP if generation_config else None,
        "top_k": generation_config.topK if generation_config else None,
        "max_tokens": generation_config.maxOutputTokens if generation_config else None,
        "generation_config_overrides": generation_config_overrides or None,
    }

import pytest

from aistudio_api.api.schemas import GeminiContent, GeminiGenerateContentRequest, GeminiGenerationConfig, GeminiPart
from aistudio_api.application.chat_service import normalize_gemini_request, normalize_openai_tools
from aistudio_api.api.schemas import ChatRequest
from aistudio_api.infrastructure.gateway.wire_types import AistudioImageOutputMode


def test_normalize_gemini_request_exposes_generation_config_overrides():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
        generationConfig=GeminiGenerationConfig(
            stopSequences=["6"],
            temperature=1,
            topP=0.95,
            topK=64,
            maxOutputTokens=65536,
            responseMimeType="text/plain",
            responseSchema={
                "type": "object",
                "properties": {"test_response": {"type": "string"}},
                "propertyOrdering": ["test_response"],
            },
            presencePenalty=0.1,
            frequencyPenalty=0.2,
            responseLogprobs=True,
            logprobs=5,
            mediaResolution=2,
            thinkingConfig=[1, None, None, 3],
        ),
    )

    normalized = normalize_gemini_request(req, "models/gemini-3.1-flash-image-preview")

    assert normalized["generation_config_overrides"] == {
        "stop_sequences": ["6"],
        "max_tokens": 65536,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "response_mime_type": "text/plain",
        "response_schema": [6, None, None, None, None, None, [["test_response", [1]]], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ["test_response"]],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "response_logprobs": True,
        "logprobs": 5,
        "image_output_mode": AistudioImageOutputMode.image_only(),
        "media_resolution": 2,
        "thinking_config": [1, None, None, 3],
    }


def test_normalize_gemini_request_maps_official_image_generation_fields():
    req = GeminiGenerateContentRequest.model_validate(
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": "/9j/4AAQSkZJRgABAQAA...."}},
                        {"text": "INSERT_INPUT_HERE"},
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "thinkingConfig": {"thinkingLevel": "HIGH"},
                "imageConfig": {
                    "aspectRatio": "9:16",
                    "imageSize": "4K",
                    "personGeneration": "",
                },
            },
            "tools": [
                {
                    "googleSearch": {
                        "searchTypes": {
                            "webSearch": {},
                            "imageSearch": {},
                        }
                    }
                }
            ],
        }
    )

    normalized = normalize_gemini_request(req, "models/gemini-3.1-flash-image-preview")

    assert normalized["tools"] == [[None, None, None, [None, [[], []]]]]
    assert normalized["generation_config_overrides"] == {
        "image_output_mode": AistudioImageOutputMode.text_and_image(),
        "thinking_config": [1, None, None, 3],
        "output_resolution": ["9:16", "4K"],
    }
    assert normalized["capture_images"] is not None
    assert len(normalized["capture_images"]) == 1
    assert normalized["contents"][0].parts[0].inline_data == ("image/jpeg", "/9j/4AAQSkZJRgABAQAA....")


def test_normalize_gemini_request_encodes_function_declarations_to_wire_tools():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
        tools=[
            {
                "functionDeclarations": [
                    {
                        "name": "getWeather",
                        "description": "gets the weather for a requested city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "propertyOrdering": ["city"],
                        },
                    }
                ]
            }
        ],
    )

    normalized = normalize_gemini_request(req, "models/gemma-4-31b-it")

    assert normalized["tools"] == [
        [
            None,
            [
                [
                    "getWeather",
                    "gets the weather for a requested city",
                    [6, None, None, None, None, None, [["city", [1]]], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ["city"]],
                ]
            ],
        ]
    ]


def test_normalize_gemini_request_applies_gemma_default_tools():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
    )

    normalized = normalize_gemini_request(req, "models/gemma-4-31b-it")

    assert normalized["tools"] == [[None, None, None, [None, [[]]]]]


def test_normalize_gemini_request_encodes_builtin_tools_to_wire():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
        tools=[
            {
                "googleSearch": {},
                "googleMaps": {},
                "urlContext": {},
                "codeExecution": {},
            }
        ],
    )

    normalized = normalize_gemini_request(req, "models/gemini-3.5-flash")

    assert normalized["tools"] == [
        [[]],
        [None, None, None, [None, [[]]]],
        [None, None, None, None, None, None, None, None, None, None, []],
        [None, None, None, None, None, None, None, []],
    ]


def test_normalize_gemini_request_rejects_gemma_unsupported_builtin_tool():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
        tools=[
            {
                "googleMaps": {},
            }
        ],
    )

    with pytest.raises(ValueError, match="not allowed"):
        normalize_gemini_request(req, "models/gemma-4-31b-it")


def test_normalize_gemini_request_empty_tools_disables_model_defaults():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
        tools=[],
    )

    normalized = normalize_gemini_request(req, "models/gemma-4-31b-it")

    assert normalized["tools"] == []


def test_normalize_gemini_request_rejects_unsupported_person_generation():
    req = GeminiGenerateContentRequest.model_validate(
        {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {
                "imageConfig": {
                    "personGeneration": "ALLOW_ADULT",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="personGeneration"):
        normalize_gemini_request(req, "models/gemini-3.1-flash-image-preview")


def test_normalize_gemini_request_maps_official_text_model_fields():
    req = GeminiGenerateContentRequest.model_validate(
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "INSERT_INPUT_HERE"},
                    ],
                }
            ],
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": "HIGH"},
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ],
            "tools": [
                {"urlContext": {}},
                {"codeExecution": {}},
                {"googleSearch": {}},
            ],
        }
    )

    normalized = normalize_gemini_request(req, "models/gemini-3.5-flash")

    assert normalized["tools"] == [
        [None, None, None, None, None, None, None, []],
        [[]],
        [None, None, None, [None, [[]]]],
    ]
    assert normalized["generation_config_overrides"] == {
        "thinking_config": [1, None, None, 3],
    }
    assert normalized["safety_settings"] == [
        [None, None, 7, 1],
        [None, None, 8, 4],
        [None, None, 9, 3],
        [None, None, 10, 2],
    ]


def test_normalize_openai_tools_encodes_function_tools_to_wire():
    req = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "getWeather",
                    "description": "gets the weather for a requested city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "propertyOrdering": ["city"],
                    },
                },
            }
        ],
    )

    assert normalize_openai_tools(req.tools) == [
        [
            None,
            [
                [
                    "getWeather",
                    "gets the weather for a requested city",
                    [6, None, None, None, None, None, [["city", [1]]], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ["city"]],
                ]
            ],
        ]
    ]


def test_normalize_openai_tools_omits_required_from_function_schema_wire():
    req = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click by ref",
                    "parameters": {
                        "type": "object",
                        "properties": {"ref": {"type": "string"}},
                        "required": ["ref"],
                    },
                },
            }
        ],
    )

    schema = normalize_openai_tools(req.tools)[0][1][0][2]
    assert len(schema) <= 7 or schema[7] is None


def _iter_wire_schemas(schema):
    if not isinstance(schema, list) or not schema:
        return
    yield schema
    if len(schema) > 5:
        yield from _iter_wire_schemas(schema[5])
    if len(schema) > 6 and isinstance(schema[6], list):
        for property_entry in schema[6]:
            if isinstance(property_entry, list) and len(property_entry) >= 2:
                yield from _iter_wire_schemas(property_entry[1])


def test_normalize_openai_tools_sanitizes_nullable_unions_and_refs():
    req = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "screen_stock",
                    "description": "Screen stocks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "window": {"type": ["integer", "null"]},
                            "filters": {"type": "array", "items": {"$ref": "#/$defs/Filter"}},
                            "mode": {"enum": ["fast", "deep"]},
                            "payload": {"allOf": [{"$ref": "#/$defs/Payload"}]},
                        },
                        "required": ["symbol"],
                        "propertyOrdering": ["symbol", "window", "filters", "mode", "payload"],
                        "$defs": {
                            "Filter": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "threshold": {"oneOf": [{"type": "number"}, {"type": "null"}]},
                                    "enabled": {"type": ["boolean", "null"]},
                                },
                                "required": ["field", "threshold"],
                            },
                            "Payload": {
                                "type": "object",
                                "properties": {"notes": {"type": ["string", "null"]}},
                            },
                        },
                    },
                },
            }
        ],
    )

    schema = normalize_openai_tools(req.tools)[0][1][0][2]
    properties = {name: value for name, value in schema[6]}

    assert properties["symbol"] == [1]
    assert properties["window"] == [3]
    assert properties["mode"] == [1]
    assert properties["filters"][0] == 5
    filter_properties = {name: value for name, value in properties["filters"][5][6]}
    assert filter_properties["field"] == [1]
    assert filter_properties["threshold"] == [2]
    assert filter_properties["enabled"] == [4]
    payload_properties = {name: value for name, value in properties["payload"][6]}
    assert payload_properties["notes"] == [1]
    assert all(item[0] != 0 for item in _iter_wire_schemas(schema))
    assert len(schema) <= 7 or schema[7] is None


def test_normalize_gemini_function_declarations_sanitize_schema_before_encoding():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
        tools=[
            {
                "functionDeclarations": [
                    {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            },
                        },
                    }
                ]
            }
        ],
    )

    schema = normalize_gemini_request(req, "models/gemini-3.5-flash")["tools"][0][1][0][2]
    assert schema[6] == [["query", [1]]]


def test_normalize_gemini_response_schema_sanitizes_nullable_unions():
    req = GeminiGenerateContentRequest(
        contents=[GeminiContent(role="user", parts=[GeminiPart(text="hello")])],
        generationConfig=GeminiGenerationConfig(
            responseSchema={
                "type": "object",
                "properties": {
                    "answer": {"type": ["string", "null"]},
                    "score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
            },
        ),
    )

    schema = normalize_gemini_request(req, "models/gemini-3.5-flash")["generation_config_overrides"][
        "response_schema"
    ]
    assert schema[6] == [["answer", [1]], ["score", [2]]]

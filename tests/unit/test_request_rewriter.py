import json

from aistudio_api.infrastructure.gateway.request_rewriter import (
    AistudioWireCodec,
    build_image_generation_search_tool,
    build_tools_from_names,
    modify_body,
)


def test_modify_body_updates_generation_config_and_prompt():
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16],"!snap",null,null]'
    rewritten = modify_body(
        original,
        model="models/new",
        prompt="new prompt",
        system_instruction="sys",
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
        top_k=32,
    )

    assert '"models/new"' in rewritten
    assert '"new prompt"' in rewritten
    assert '"sys"' in rewritten
    body = json.loads(rewritten)
    assert body[3][3] == 256
    assert body[3][4] == 0.2
    assert body[3][5] == 0.9
    assert body[3][6] == 32
    assert body[3][16] == [1, None, None, 3]
    assert len(body[3]) <= 17 or body[3][17] is None


def test_build_image_generation_search_tool_variants():
    assert build_image_generation_search_tool(google_search=True, image_search=False) == [None, None, None, [None, [[]]]]
    assert build_image_generation_search_tool(google_search=False, image_search=True) == [None, None, None, [None, [None, []]]]
    assert build_image_generation_search_tool(google_search=True, image_search=True) == [None, None, None, [None, [[], []]]]
    assert build_image_generation_search_tool(google_search=False, image_search=False) is None


def test_build_tools_from_names_supports_image_tool_aliases():
    assert build_tools_from_names(["google_search"], model="models/gemini-3.1-flash-image-preview", is_image_model=True) == [[None, None, None, [None, [[]]]]]
    assert build_tools_from_names(["image_search"], model="models/gemini-3.1-flash-image-preview", is_image_model=True) == [[None, None, None, [None, [None, []]]]]
    assert build_tools_from_names(["google_search_and_image_search"], model="models/gemini-3.1-flash-image-preview", is_image_model=True) == [
        [None, None, None, [None, [[], []]]]
    ]


def test_build_tools_from_names_merges_image_search_flags_into_single_tool():
    assert build_tools_from_names(
        ["google_search", "image_search"],
        model="models/gemini-3.1-flash-image-preview",
        is_image_model=True,
    ) == [[None, None, None, [None, [[], []]]]]


def test_build_tools_from_names_restricts_gemma_builtin_tools():
    assert build_tools_from_names(["google_search", "code_execution"], model="models/gemma-4-31b-it") == [
        [None, None, None, [None, [[]]]],
        [[]],
    ]

    try:
        build_tools_from_names(["google_maps"], model="models/gemma-4-31b-it")
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_tools_from_names_allows_four_gemini_builtin_tools():
    assert build_tools_from_names(
        ["google_search", "code_execution", "google_maps", "url_context"],
        model="models/gemini-3.5-flash",
    ) == [
        [None, None, None, [None, [[]]]],
        [[]],
        [None, None, None, None, None, None, None, None, None, None, []],
        [None, None, None, None, None, None, None, []],
    ]


def test_build_tools_from_names_restricts_unknown_model_to_safe_subset():
    try:
        build_tools_from_names(["google_maps"], model="models/learnlm-1.5-pro-experimental")
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_wire_codec_decodes_semantic_fields():
    codec = AistudioWireCodec()
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16],"!snap",[[[null,"sys"]],"user"],[[[]]]]'

    decoded = codec.decode(original)

    assert decoded.model == "models/original"
    assert decoded.snapshot == "!snap"
    assert decoded.contents[0].role == "user"
    assert decoded.contents[0].parts[-1].text == "old"
    assert decoded.system_instruction is not None
    assert decoded.system_instruction.parts[0].text == "sys"
    assert decoded.generation_config.max_tokens == 128


def test_modify_body_sanitizes_plain_text_generation_config():
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16,"application/json",[6],null,null,null,null,null,null,null,[1,null,null,3]],"!snap",null,null]'
    rewritten = modify_body(
        original,
        model="models/gemma-4-31b-it",
        prompt="hello",
    )

    assert '"text/plain"' in rewritten
    assert '"application/json"' not in rewritten
    assert '[6]' not in rewritten
    assert json.loads(rewritten)[3][16] == [1, None, None, 3]
    assert len(json.loads(rewritten)[3]) <= 17 or json.loads(rewritten)[3][17] is None


def test_modify_body_enables_thinking_for_any_model():
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16],"!snap",null,null]'
    rewritten = modify_body(
        original,
        model="models/new-text-model",
        prompt="hello",
    )

    body = json.loads(rewritten)
    assert body[3][16] == [1, None, None, 3]
    assert len(body[3]) <= 17 or body[3][17] is None


def test_modify_body_can_override_image_output_resolution():
    original = json.dumps(
        [
            "models/original",
            [[[[None, "old"]], "user"]],
            None,
            [None] * 27,
            "!snap",
            None,
            None,
        ]
    )
    rewritten = modify_body(
        original,
        model="models/gemini-3.1-flash-image-preview",
        prompt="hello",
        generation_config_overrides={"output_resolution": ["16:9", "1K"]},
    )

    body = json.loads(rewritten)
    assert body[3][26] == ["16:9", "1K"]


def test_modify_body_applies_model_safety_defaults():
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16],"!snap",null,null]'
    rewritten = modify_body(
        original,
        model="models/gemma-4-31b-it",
        prompt="hello",
    )

    body = json.loads(rewritten)
    assert body[2] == [
        [None, None, 7, 5],
        [None, None, 8, 5],
        [None, None, 9, 5],
        [None, None, 10, 5],
    ]


def test_modify_body_applies_image_model_defaults():
    original = json.dumps(
        [
            "models/original",
            [[[[None, "old"]], "user"]],
            [[None, None, 7, 4]],
            [None] * 27,
            "!snap",
            None,
            None,
        ]
    )
    rewritten = modify_body(
        original,
        model="models/gemini-3.1-flash-image-preview",
        prompt="hello",
    )

    body = json.loads(rewritten)
    assert body[2] is None
    assert body[3][14] == [2]
    assert body[3][16] == [1, None, None, 4]


def test_modify_body_keeps_image_model_tools():
    original = json.dumps(
        [
            "models/original",
            [[[[None, "old"]], "user"]],
            None,
            [None] * 27,
            "!snap",
            None,
            None,
        ]
    )
    rewritten = modify_body(
        original,
        model="models/gemini-3.1-flash-image-preview",
        prompt="hello",
        tools=[[None, None, None, [None, [[], []]]]],
    )

    body = json.loads(rewritten)
    assert body[6] == [[None, None, None, [None, [[], []]]]]


def test_modify_body_safety_off_uses_off_threshold():
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16],"!snap",null,null]'
    rewritten = modify_body(
        original,
        model="models/gemma-4-31b-it",
        prompt="hello",
        safety_off=True,
    )

    body = json.loads(rewritten)
    assert body[2] == [
        [None, None, 7, 5],
        [None, None, 8, 5],
        [None, None, 9, 5],
        [None, None, 10, 5],
    ]


def test_modify_body_can_override_safety_settings():
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,null,null,128,0.5,0.8,16],"!snap",null,null]'
    rewritten = modify_body(
        original,
        model="models/gemini-3.5-flash",
        prompt="hello",
        safety_settings=[
            [None, None, 7, 1],
            [None, None, 8, 4],
            [None, None, 9, 3],
            [None, None, 10, 2],
        ],
    )

    body = json.loads(rewritten)
    assert body[2] == [
        [None, None, 7, 1],
        [None, None, 8, 4],
        [None, None, 9, 3],
        [None, None, 10, 2],
    ]


def test_modify_body_keeps_structured_generation_config_for_gemini_mode():
    original = '["models/original",[[[[null,"old"]],"user"]],null,[null,["6"],null,128,0.5,0.8,16,"application/json",[6],0.1,0.2,true,5,null,[2,1],null,[1,null,null,3],2],"!snap",null,null,null,null,null,1,"cached",null,[[null,null,"Asia/Shanghai"],null,1]]'
    rewritten = modify_body(
        original,
        model="models/gemini-2.5-pro-preview-05-06",
        prompt="hello",
        generation_config_overrides={
            "stop_sequences": ["STOP"],
            "response_mime_type": "application/json",
            "response_schema": [6],
            "presence_penalty": 0.3,
            "frequency_penalty": 0.4,
            "response_logprobs": True,
            "logprobs": 7,
            "image_output_mode": [2, 1],
            "thinking_config": [1, None, None, 3],
            "media_resolution": 2,
        },
        sanitize_plain_text=False,
    )

    body = json.loads(rewritten)
    assert body[3][1] == ["STOP"]
    assert body[3][7] == "application/json"
    assert body[3][8] == [6]
    assert body[3][9] == 0.3
    assert body[3][10] == 0.4
    assert body[3][11] is True
    assert body[3][12] == 7
    assert body[3][14] == [2, 1]
    assert body[3][16] == [1, None, None, 3]
    assert body[3][17] == 2

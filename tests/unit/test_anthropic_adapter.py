from aistudio_api.api.responses import anthropic_message_response
from aistudio_api.api.schemas import AnthropicMessageRequest
from aistudio_api.application.api_service_anthropic import _filter_internal_tool_history_delta
from aistudio_api.application.chat_service import (
    ANTHROPIC_TOOL_HISTORY_END,
    ANTHROPIC_TOOL_HISTORY_START,
    normalize_anthropic_request,
)
from aistudio_api.infrastructure.gateway.wire_codec import AistudioWireCodec
from aistudio_api.infrastructure.gateway.wire_types import AistudioPart


def test_aistudio_part_encodes_function_call_and_response():
    call = AistudioPart(function_call=("Read", {"file_path": "navigation/a.py"}, "toolu_1"), thought_signature="sig")
    response = AistudioPart(function_response=("Read", {"result": "content"}))

    assert call.to_wire()[10] == ["Read", [[["file_path", [None, None, "navigation/a.py"]]]], "toolu_1"]
    assert call.to_wire()[14] == "sig"
    assert response.to_wire()[11] == ["Read", [[["result", [None, None, "content"]]]]]


def test_wire_codec_decodes_function_call_and_response_parts():
    codec = AistudioWireCodec()

    call = codec._decode_part([None, None, None, None, None, None, None, None, None, None, ["Read", {"file_path": "navigation/a.py"}, "toolu_1"]])
    response = codec._decode_part([None, None, None, None, ["Read", {"result": "content"}]])

    assert call.function_call == ("Read", {"file_path": "navigation/a.py"}, "toolu_1")
    assert response.function_response == ("Read", {"result": "content"})


def test_normalize_anthropic_request_maps_tool_use_and_tool_result_to_transcript_parts():
    req = AnthropicMessageRequest(
        model="gemini-3.1-pro-preview",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "navigation/a.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": '[{"type":"text","text":"file content"}]',
                    }
                ],
            },
        ],
        tools=[
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
    )

    normalized = normalize_anthropic_request(
        req,
        tool_context={
            "toolu_1": {
                "call_id": "real_call_1",
                "thought_signature": "real_signature",
                "name": "Read",
            }
        },
    )

    assert normalized["contents"][0].role == "user"
    assert len(normalized["contents"]) == 1
    tool_history = normalized["contents"][0].parts[0].text
    assert tool_history.startswith(ANTHROPIC_TOOL_HISTORY_START)
    assert tool_history.endswith(ANTHROPIC_TOOL_HISTORY_END)
    assert "user tool_result for: Read" in tool_history
    assert "file content" in tool_history
    assert "Tool call Read input" not in tool_history
    assert "INTERNAL TOOL HISTORY" in normalized["system_instruction"].parts[0].text
    assert normalized["tools"][0][1][0][0] == "Read"
    schema = normalized["tools"][0][1][0][2]
    assert len(schema) <= 7 or schema[7] is None


def test_anthropic_message_response_maps_function_calls_to_tool_use_blocks():
    response = anthropic_message_response(
        model="gemini-3.1-pro-preview",
        content="",
        function_calls=[{"name": "Read", "args": {"file_path": "navigation/a.py"}}],
    )

    assert response.stop_reason == "tool_use"
    assert response.content[0].type == "tool_use"
    assert response.content[0].name == "Read"
    assert response.content[0].input == {"file_path": "navigation/a.py"}


def test_internal_tool_history_filter_handles_split_markers():
    state = {"in_tool_history": False}

    assert _filter_internal_tool_history_delta("answer ", state=state) == "answer "
    assert _filter_internal_tool_history_delta("<internal_anthropic", state=state) == ""
    assert _filter_internal_tool_history_delta("_tool_history>secret", state=state) == ""
    assert _filter_internal_tool_history_delta("</internal_anthropic_tool", state=state) == ""
    assert _filter_internal_tool_history_delta("_history> done", state=state, final=True) == " done"

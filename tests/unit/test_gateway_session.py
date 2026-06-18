from aistudio_api.infrastructure.gateway.session import BrowserSession, _snapshot_content_hash
from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart


def test_resolve_captured_info_preserves_partial_overrides():
    session = BrowserSession.__new__(BrowserSession)
    session._templates = {
        "model": {
            "url": "https://fallback.example/rpc",
            "headers": {
                "Authorization": "Bearer token",
                "Host": "ignored.example",
                "Content-Length": "123",
            },
        }
    }

    url, headers = session._resolve_captured_info("https://override.example/rpc", None)
    assert url == "https://override.example/rpc"
    assert headers == {"Authorization": "Bearer token"}

    url, headers = session._resolve_captured_info(None, {"X-Test": "1"})
    assert url == "https://fallback.example/rpc"
    assert headers == {"X-Test": "1"}


def test_snapshot_content_hash_includes_thought_signature():
    base = AistudioContent(
        role="model",
        parts=[AistudioPart(function_call=("Read", {"file_path": "a.py"}, "call_1"))],
    )
    signed = AistudioContent(
        role="model",
        parts=[AistudioPart(function_call=("Read", {"file_path": "a.py"}, "call_1"), thought_signature="sig")],
    )

    assert _snapshot_content_hash([base]) != _snapshot_content_hash([signed])

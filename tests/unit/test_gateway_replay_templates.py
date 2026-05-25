import asyncio

from aistudio_api.infrastructure.gateway.capture import CapturedRequest, RequestCaptureService
from aistudio_api.infrastructure.gateway.replay import RequestReplayService
from aistudio_api.infrastructure.gateway.streaming import StreamingGateway


def _body(model: str = "models/gemma-4-31b-it", snapshot: str = "snap") -> str:
    return f'["{model}",[[[[null,"old"]],"user"]],null,[],"{snapshot}"]'


def test_capture_service_refreshes_template_from_session():
    class _Session:
        def __init__(self):
            self.calls = 0

        async def capture_template(self, model):
            self.calls += 1
            return {
                "url": "https://fresh.example/GenerateContent",
                "headers": {"content-type": "application/json"},
                "body": _body(model),
            }

    session = _Session()
    service = RequestCaptureService(session, snapshot_cache=object())
    service._templates["models/gemma-4-31b-it"] = CapturedRequest(
        url="https://stale.example/GenerateContent",
        headers={},
        body=_body("models/stale"),
    )

    template = asyncio.run(service._ensure_template("models/gemma-4-31b-it"))

    assert session.calls == 1
    assert template.url == "https://fresh.example/GenerateContent"


def test_replay_uses_url_and_headers_from_captured_request():
    class _Session:
        def __init__(self):
            self.kwargs = None

        async def send_hooked_request(self, **kwargs):
            self.kwargs = kwargs
            return 200, b"ok"

    session = _Session()
    captured = CapturedRequest(
        url="https://fresh.example/GenerateContent",
        headers={
            "content-type": "application/json",
            "host": "stale.example",
            "content-length": "123",
        },
        body=_body(),
    )

    status, raw = asyncio.run(RequestReplayService(session).replay(captured, body=_body()))

    assert status == 200
    assert raw == b"ok"
    assert session.kwargs["captured_url"] == "https://fresh.example/GenerateContent"
    assert session.kwargs["captured_headers"] == {"content-type": "application/json"}


def test_streaming_uses_url_and_headers_from_captured_request():
    class _Session:
        def __init__(self):
            self.kwargs = None

        async def send_streaming_request(self, **kwargs):
            self.kwargs = kwargs
            yield "status", 200

    async def _collect():
        session = _Session()
        captured = CapturedRequest(
            url="https://fresh.example/GenerateContent",
            headers={"content-type": "application/json"},
            body=_body(),
        )
        events = [
            event
            async for event in StreamingGateway(session).stream_chat(
                captured=captured,
                model="models/gemma-4-31b-it",
                system_instruction=None,
            )
        ]
        return session.kwargs, events

    kwargs, events = asyncio.run(_collect())

    assert kwargs["captured_url"] == "https://fresh.example/GenerateContent"
    assert kwargs["captured_headers"] == {"content-type": "application/json"}
    assert events[-1] == ("done", None)

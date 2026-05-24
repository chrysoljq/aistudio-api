import asyncio
from contextlib import asynccontextmanager

from aistudio_api.api.schemas import ChatRequest
from aistudio_api.application.api_service import handle_chat


class _CaptureChatClient:
    def __init__(self):
        self.calls = []

    async def generate_content(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})

        class _Output:
            text = "ok"
            thinking = ""
            usage = {}
            function_calls = []

        return _Output()


def test_handle_chat_empty_tools_disables_model_defaults(monkeypatch):
    from aistudio_api.application import api_service_openai

    @asynccontextmanager
    async def _request_client(client, attempt=0):
        yield client

    monkeypatch.setattr(api_service_openai, "request_client", _request_client)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemma-4-31b-it",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    asyncio.run(handle_chat(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["tools"] == []

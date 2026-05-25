import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException

from aistudio_api.api.schemas import ImageRequest
from aistudio_api.application.api_service import handle_image_edit, handle_image_generation


class _UnusedClient:
    async def generate_image(self, *args, **kwargs):
        raise AssertionError("generate_image should not be called for invalid requests")


def test_handle_image_generation_rejects_unsupported_n():
    req = ImageRequest(prompt="hello", n=2)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(handle_image_generation(req, _UnusedClient()))

    assert exc.value.status_code == 400
    assert exc.value.detail["type"] == "invalid_request_error"
    assert exc.value.detail["message"] == "Only n=1 is currently supported"


def test_handle_image_edit_rejects_unsupported_size():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            handle_image_edit(
                prompt="hello",
                image_files=[],
                mask_file=None,
                model="gemini-3.1-flash-image-preview",
                n=1,
                size="800x600",
                client=_UnusedClient(),
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["type"] == "invalid_request_error"
    assert exc.value.detail["message"] == "Unsupported image size '800x600'"


class _CaptureImageClient:
    def __init__(self):
        self.calls = []

    async def generate_image(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})

        class _Output:
            images = []
            text = ""
            usage = None

        return _Output()


def test_handle_image_generation_passes_split_search_flags(monkeypatch):
    from aistudio_api.application import api_service_openai

    @asynccontextmanager
    async def _request_client(client, attempt=0):
        yield client

    monkeypatch.setattr(api_service_openai, "request_client", _request_client)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureImageClient()
    req = ImageRequest(prompt="hello", google_search=True, image_search=True)

    asyncio.run(handle_image_generation(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["google_search"] is True
    assert client.calls[0]["kwargs"]["image_search"] is True
    assert client.calls[0]["kwargs"]["use_default_tools"] is False


def test_handle_image_generation_allows_default_tools_when_search_flags_omitted(monkeypatch):
    from aistudio_api.application import api_service_openai

    @asynccontextmanager
    async def _request_client(client, attempt=0):
        yield client

    monkeypatch.setattr(api_service_openai, "request_client", _request_client)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureImageClient()
    req = ImageRequest(prompt="hello")

    asyncio.run(handle_image_generation(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["use_default_tools"] is True

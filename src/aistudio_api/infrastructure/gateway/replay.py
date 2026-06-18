"""Captured request replay workflow."""

from __future__ import annotations

import logging

from aistudio_api.config import settings
from aistudio_api.infrastructure.gateway.capture import CapturedRequest
from aistudio_api.infrastructure.gateway.session import BrowserSession

logger = logging.getLogger("aistudio")


class RequestReplayService:
    def __init__(self, session: BrowserSession | None):
        self._session = session

    async def replay(self, captured: CapturedRequest | None, body: str, timeout: int | None = None) -> tuple[int, bytes]:
        if not captured:
            return 0, b""

        if timeout is None:
            timeout = settings.timeout_replay

        headers = {k: v for k, v in captured.headers.items() if k.lower() not in ("host", "content-length")}

        try:
            if self._session is not None:
                return await self._session.send_hooked_request(
                    body=body,
                    timeout_ms=timeout * 1000,
                    url=captured.url,
                    headers=headers,
                )

            import aiohttp

            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(
                    captured.url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    raw = await resp.read()
                    return resp.status, raw
        except Exception as exc:
            logger.error("Replay error: %s", exc)
            return 0, str(exc).encode()

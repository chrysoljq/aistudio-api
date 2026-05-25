"""Per-account browser client pool."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from aistudio_api.application.account_rotator import AccountRotator, RotationMode
from aistudio_api.infrastructure.account.account_store import AccountMeta, AccountStore
from aistudio_api.infrastructure.cache.snapshot_cache import SnapshotCache
from aistudio_api.infrastructure.gateway.client import AIStudioClient

logger = logging.getLogger("aistudio.pool")


@dataclass(slots=True)
class AccountClientEntry:
    account: AccountMeta
    auth_file: str
    client: AIStudioClient
    snapshot_cache: SnapshotCache
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_acquired: float = 0.0


@dataclass(frozen=True, slots=True)
class AccountClientLease:
    account_id: str
    account_name: str
    client: AIStudioClient


class AccountClientPool:
    """Owns one long-lived browser-backed client per pooled account."""

    def __init__(
        self,
        *,
        account_store: AccountStore,
        rotator: AccountRotator,
        port: int,
        size: int,
        account_selectors: tuple[str, ...] = (),
    ) -> None:
        self._account_store = account_store
        self._rotator = rotator
        self._port = port
        self._configured_size = max(0, size)
        self._account_selectors = tuple(selector.strip() for selector in account_selectors if selector.strip())
        self._entries: list[AccountClientEntry] = []
        self._entries_by_id: dict[str, AccountClientEntry] = {}
        self._select_lock = asyncio.Lock()
        self._rr_index = 0
        self._load_entries()

    @property
    def enabled(self) -> bool:
        return bool(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    def contains(self, account_id: str) -> bool:
        return account_id in self._entries_by_id

    def _load_entries(self) -> None:
        accounts = self._select_accounts(self._account_store.list_accounts())
        limit = self._configured_size or len(accounts)
        for account in accounts:
            if len(self._entries) >= limit:
                break
            auth_path = self._account_store.get_auth_path_optional(account.id, require_exists=True)
            if auth_path is None:
                logger.warning("账号 %s 缺少 auth.json，跳过池化", account.id)
                continue
            snapshot_cache = SnapshotCache()
            client = AIStudioClient(
                port=self._port,
                auth_file=str(auth_path),
                snapshot_cache=snapshot_cache,
            )
            entry = AccountClientEntry(
                account=account,
                auth_file=str(auth_path),
                client=client,
                snapshot_cache=snapshot_cache,
            )
            self._entries.append(entry)
            self._entries_by_id[account.id] = entry

        if limit > 0:
            logger.info(
                "浏览器池初始化: configured=%d, selected=%d, loaded=%d",
                limit,
                len(accounts),
                len(self._entries),
            )

    def _select_accounts(self, accounts: list[AccountMeta]) -> list[AccountMeta]:
        if not self._account_selectors:
            return accounts

        by_key: dict[str, AccountMeta] = {}
        for account in accounts:
            by_key[account.id] = account
            by_key[account.name.lower()] = account
            if account.email:
                by_key[account.email.lower()] = account

        selected: list[AccountMeta] = []
        seen: set[str] = set()
        for selector in self._account_selectors:
            account = by_key.get(selector) or by_key.get(selector.lower())
            if account is None:
                logger.warning("浏览器池配置的账号不存在，跳过: %s", selector)
                continue
            if account.id in seen:
                continue
            selected.append(account)
            seen.add(account.id)
        return selected

    async def warmup(self) -> None:
        async def _warm_entry(entry: AccountClientEntry) -> None:
            try:
                await entry.client.warmup()
                logger.info("浏览器池账号预热完成: %s (%s)", entry.account.id, entry.account.name)
            except Exception as exc:
                logger.warning("浏览器池账号预热失败: %s (%s): %s", entry.account.id, entry.account.name, exc)

        await asyncio.gather(*(_warm_entry(entry) for entry in self._entries))

    async def close(self) -> None:
        await asyncio.gather(*(entry.client.close() for entry in self._entries), return_exceptions=True)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[AccountClientLease]:
        entry = await self._select_entry()
        await entry.lock.acquire()
        try:
            yield AccountClientLease(
                account_id=entry.account.id,
                account_name=entry.account.name,
                client=entry.client,
            )
        finally:
            entry.lock.release()

    async def _select_entry(self) -> AccountClientEntry:
        while True:
            async with self._select_lock:
                if not self._entries:
                    raise RuntimeError("account client pool has no available accounts")

                stats = self._rotator.get_all_stats()
                available = [
                    entry
                    for entry in self._entries
                    if stats.get(entry.account.id, {}).get("is_available", True)
                ]
                if available:
                    unlocked = [entry for entry in available if not entry.lock.locked()]
                    entry = self._pick_entry(unlocked or available, stats)
                    entry.last_acquired = time.time()
                    logger.info(
                        "浏览器池选择账号: %s (mode=%s, busy=%s)",
                        entry.account.name,
                        self._mode_value(),
                        entry.lock.locked(),
                    )
                    return entry

                wait_seconds = min(
                    max(0, int(stats.get(entry.account.id, {}).get("cooldown_remaining", 1)))
                    for entry in self._entries
                )

            await asyncio.sleep(max(0.1, wait_seconds))

    def _pick_entry(self, candidates: list[AccountClientEntry], stats: dict[str, dict]) -> AccountClientEntry:
        mode = self._rotator.mode
        if mode == RotationMode.ROUND_ROBIN:
            return self._pick_round_robin(candidates)
        if mode == RotationMode.LEAST_RECENTLY_USED:
            return min(candidates, key=lambda entry: entry.last_acquired)
        if mode == RotationMode.LEAST_RATE_LIMITED:
            return min(
                candidates,
                key=lambda entry: (
                    int(stats.get(entry.account.id, {}).get("rate_limited", 0)),
                    entry.last_acquired,
                ),
            )
        return candidates[0]

    def _pick_round_robin(self, candidates: list[AccountClientEntry]) -> AccountClientEntry:
        candidate_ids = {entry.account.id for entry in candidates}
        for offset in range(len(self._entries)):
            idx = (self._rr_index + offset) % len(self._entries)
            entry = self._entries[idx]
            if entry.account.id in candidate_ids:
                self._rr_index = (idx + 1) % len(self._entries)
                return entry
        return candidates[0]

    def _mode_value(self) -> str:
        mode = self._rotator.mode
        return mode.value if hasattr(mode, "value") else str(mode)

    def status(self) -> dict:
        stats = self._rotator.get_all_stats()
        return {
            "enabled": self.enabled,
            "configured_size": self._configured_size,
            "configured_accounts": list(self._account_selectors),
            "size": len(self._entries),
            "accounts": [
                {
                    "id": entry.account.id,
                    "name": entry.account.name,
                    "email": entry.account.email,
                    "busy": entry.lock.locked(),
                    "last_acquired": entry.last_acquired or None,
                    "is_available": stats.get(entry.account.id, {}).get("is_available", True),
                    "cooldown_remaining": stats.get(entry.account.id, {}).get("cooldown_remaining", 0),
                }
                for entry in self._entries
            ],
        }

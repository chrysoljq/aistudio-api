"""Account rotation for multi-account load balancing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aistudio_api.infrastructure.account.account_store import AccountStore, AccountMeta

logger = logging.getLogger("aistudio.rotator")


class RotationMode(str, Enum):
    """轮询模式。"""
    ROUND_ROBIN = "round_robin"          # 顺序轮询
    LEAST_RECENTLY_USED = "lru"         # 最久未用
    LEAST_RATE_LIMITED = "least_rl"     # 最少限流


@dataclass
class AccountStats:
    """单账号的运行统计。"""
    account_id: str
    requests: int = 0
    success: int = 0
    rate_limited: int = 0
    errors: int = 0
    last_used: float = 0.0           # timestamp
    last_rate_limited: float = 0.0   # timestamp
    cooldown_until: float = 0.0      # timestamp, 429 后冷却期

    def is_available(self) -> bool:
        """检查账号是否可用（不在冷却期）。"""
        return time.time() >= self.cooldown_until

    def record_success(self) -> None:
        self.requests += 1
        self.success += 1
        self.last_used = time.time()

    def record_rate_limited(self, cooldown_seconds: int = 60) -> None:
        self.requests += 1
        self.rate_limited += 1
        self.last_rate_limited = time.time()
        self.cooldown_until = time.time() + cooldown_seconds

    def record_error(self) -> None:
        self.requests += 1
        self.errors += 1
        self.last_used = time.time()


class AccountRotator:
    """多账号轮询管理器。

    支持三种模式：
    - round_robin: 顺序轮询，429 时跳过冷却中的账号
    - lru: 最久未用优先，适合均匀分配负载
    - least_rl: 最少限流优先，适合最大化吞吐
    """

    def __init__(
        self,
        account_store: AccountStore,
        mode: RotationMode = RotationMode.ROUND_ROBIN,
        cooldown_seconds: int = 60,
    ) -> None:
        self._store = account_store
        self._mode = mode
        self._cooldown_seconds = cooldown_seconds
        self._stats: dict[str, AccountStats] = {}
        self._current_index: int = 0
        self._lock = asyncio.Lock()

        # 初始化已有账号的统计
        for account in self._store.list_accounts():
            if account.id not in self._stats:
                self._stats[account.id] = AccountStats(account_id=account.id)

    @property
    def mode(self) -> RotationMode:
        return self._mode

    @mode.setter
    def mode(self, value: RotationMode) -> None:
        logger.info("轮询模式切换: %s -> %s", self._mode, value)
        self._mode = value

    @property
    def cooldown_seconds(self) -> int:
        return self._cooldown_seconds

    @cooldown_seconds.setter
    def cooldown_seconds(self, value: int) -> None:
        self._cooldown_seconds = value

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """获取所有账号的统计信息。"""
        result = {}
        for account in self._store.list_accounts():
            stats = self._stats.get(account.id, AccountStats(account_id=account.id))
            result[account.id] = {
                "name": account.name,
                "email": account.email,
                "requests": stats.requests,
                "success": stats.success,
                "rate_limited": stats.rate_limited,
                "errors": stats.errors,
                "last_used": datetime.fromtimestamp(stats.last_used, tz=timezone.utc).isoformat() if stats.last_used else None,
                "last_rate_limited": datetime.fromtimestamp(stats.last_rate_limited, tz=timezone.utc).isoformat() if stats.last_rate_limited else None,
                "is_available": stats.is_available(),
                "cooldown_remaining": max(0, int(stats.cooldown_until - time.time())),
            }
        return result

    def _get_available_accounts(self) -> list[tuple[AccountMeta, AccountStats]]:
        """获取所有可用的账号（不在冷却期）。"""
        accounts = self._store.list_accounts()
        available = []
        for account in accounts:
            stats = self._stats.get(account.id, AccountStats(account_id=account.id))
            if stats.is_available():
                available.append((account, stats))
        return available

    def _pick_round_robin(self, available: list[tuple[AccountMeta, AccountStats]]) -> tuple[AccountMeta, AccountStats] | None:
        """Round-robin 选择，基于全量账号索引，避免 available 变化导致跳过或重复。"""
        if not available:
            return None
        available_ids = {a.id for a, _ in available}
        all_accounts = self._store.list_accounts()
        if not all_accounts:
            return None
        total = len(all_accounts)
        for i in range(total):
            idx = (self._current_index + i) % total
            if all_accounts[idx].id in available_ids:
                self._current_index = (idx + 1) % total
                return next((a, s) for a, s in available if a.id == all_accounts[idx].id)
        return available[0]

    def _pick_lru(self, available: list[tuple[AccountMeta, AccountStats]]) -> tuple[AccountMeta, AccountStats] | None:
        """最久未用优先。"""
        if not available:
            return None
        return min(available, key=lambda x: x[1].last_used if x[1].last_used > 0 else 0.0)

    def _pick_least_rl(self, available: list[tuple[AccountMeta, AccountStats]]) -> tuple[AccountMeta, AccountStats] | None:
        """最少限流优先。"""
        if not available:
            return None
        return min(available, key=lambda x: x[1].rate_limited)

    async def get_next_account(self) -> AccountMeta | None:
        """获取下一个可用的账号。"""
        async with self._lock:
            available = self._get_available_accounts()

            if not available:
                # 所有账号都在冷却期，找一个冷却时间最短的
                all_accounts = self._store.list_accounts()
                if not all_accounts:
                    return None
                # 选冷却结束最早的
                earliest = min(
                    [(a, self._stats.get(a.id, AccountStats(account_id=a.id))) for a in all_accounts],
                    key=lambda x: x[1].cooldown_until,
                )
                account, stats = earliest
                wait_time = max(0, stats.cooldown_until - time.time())
                logger.warning("所有账号都在冷却期，等待 %.1fs 使用 %s", wait_time, account.name)
                await asyncio.sleep(wait_time)
                return account

            # 根据模式选择
            if self._mode == RotationMode.ROUND_ROBIN:
                pick = self._pick_round_robin(available)
            elif self._mode == RotationMode.LEAST_RECENTLY_USED:
                pick = self._pick_lru(available)
            elif self._mode == RotationMode.LEAST_RATE_LIMITED:
                pick = self._pick_least_rl(available)
            else:
                pick = available[0]

            if pick is None:
                return None

            account, stats = pick
            logger.info("轮询选择账号: %s (mode=%s)", account.name, self._mode)
            return account

    async def get_next_account_with_stats(self) -> tuple[AccountMeta, AccountStats] | None:
        """获取下一个可用的账号及其统计。"""
        async with self._lock:
            available = self._get_available_accounts()
            if not available:
                all_accounts = self._store.list_accounts()
                if not all_accounts:
                    return None
                earliest = min(
                    [(a, self._stats.get(a.id, AccountStats(account_id=a.id))) for a in all_accounts],
                    key=lambda x: x[1].cooldown_until,
                )
                account, stats = earliest
                wait_time = max(0, stats.cooldown_until - time.time())
                await asyncio.sleep(wait_time)
                return account, stats

            if self._mode == RotationMode.ROUND_ROBIN:
                pick = self._pick_round_robin(available)
            elif self._mode == RotationMode.LEAST_RECENTLY_USED:
                pick = self._pick_lru(available)
            elif self._mode == RotationMode.LEAST_RATE_LIMITED:
                pick = self._pick_least_rl(available)
            else:
                pick = available[0]
            return pick

    def record_success(self, account_id: str) -> None:
        """记录成功请求。"""
        if account_id not in self._stats:
            self._stats[account_id] = AccountStats(account_id=account_id)
        self._stats[account_id].record_success()

    def record_rate_limited(self, account_id: str) -> None:
        """记录 429 限流。"""
        if account_id not in self._stats:
            self._stats[account_id] = AccountStats(account_id=account_id)
        self._stats[account_id].record_rate_limited(self._cooldown_seconds)
        logger.warning("账号 %s 被限流，冷却 %ds", account_id, self._cooldown_seconds)

    def record_error(self, account_id: str) -> None:
        """记录错误。"""
        if account_id not in self._stats:
            self._stats[account_id] = AccountStats(account_id=account_id)
        self._stats[account_id].record_error()

    def add_account(self, account_id: str) -> None:
        """添加新账号时初始化统计。"""
        if account_id not in self._stats:
            self._stats[account_id] = AccountStats(account_id=account_id)

    def remove_account(self, account_id: str) -> None:
        """删除账号时清理统计。"""
        self._stats.pop(account_id, None)


# 全局轮询器实例
_rotator: AccountRotator | None = None


def get_rotator() -> AccountRotator | None:
    """获取全局轮询器。"""
    return _rotator


def init_rotator(account_store: AccountStore, **kwargs) -> AccountRotator:
    """初始化全局轮询器。"""
    global _rotator
    _rotator = AccountRotator(account_store, **kwargs)
    return _rotator

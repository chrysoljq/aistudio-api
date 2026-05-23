import asyncio
from types import SimpleNamespace

from aistudio_api.api.state import runtime_state
from aistudio_api.application import api_service_common
from aistudio_api.application.account_rotator import AccountRotator, RotationMode
from aistudio_api.infrastructure.account.account_store import AccountMeta


class _Store:
    def __init__(self, accounts):
        self._accounts = accounts

    def list_accounts(self):
        return list(self._accounts)


def _account(account_id: str) -> AccountMeta:
    return AccountMeta(
        id=account_id,
        name=f"{account_id}@example.com",
        email=f"{account_id}@example.com",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_lru_prefers_never_used_accounts():
    rotator = AccountRotator(
        _Store([_account("a"), _account("b"), _account("c")]),
        mode=RotationMode.LEAST_RECENTLY_USED,
    )
    rotator.record_success("a")

    selected = asyncio.run(rotator.get_next_account())

    assert selected is not None
    assert selected.id == "b"


def test_try_switch_account_skips_reactivating_current_account():
    active = _account("a")

    class _Rotator:
        async def get_next_account(self):
            return active

    class _AccountService:
        def __init__(self):
            self.activations = 0

        def get_active_account(self):
            return active

        async def activate_account(self, *args, **kwargs):
            self.activations += 1
            return active

    account_service = _AccountService()
    original = (
        runtime_state.rotator,
        runtime_state.account_service,
        runtime_state.client,
        runtime_state.snapshot_cache,
    )
    runtime_state.rotator = _Rotator()
    runtime_state.account_service = account_service
    runtime_state.client = SimpleNamespace(_session=object())
    runtime_state.snapshot_cache = object()
    try:
        assert asyncio.run(api_service_common.try_switch_account()) is True
        assert account_service.activations == 0
    finally:
        (
            runtime_state.rotator,
            runtime_state.account_service,
            runtime_state.client,
            runtime_state.snapshot_cache,
        ) = original


def test_ensure_active_account_uses_rotator_even_when_account_is_active():
    active = _account("a")
    target = _account("b")

    class _Rotator:
        async def get_next_account(self):
            return target

    class _AccountService:
        def __init__(self):
            self.activated = []

        def get_active_account(self):
            return active

        async def activate_account(self, account_id, *args, **kwargs):
            self.activated.append(account_id)
            return target

    account_service = _AccountService()
    original = (
        runtime_state.rotator,
        runtime_state.account_service,
        runtime_state.client,
        runtime_state.snapshot_cache,
    )
    runtime_state.rotator = _Rotator()
    runtime_state.account_service = account_service
    runtime_state.client = SimpleNamespace(_session=object())
    runtime_state.snapshot_cache = object()
    try:
        asyncio.run(api_service_common.ensure_active_account(0))
        assert account_service.activated == ["b"]
    finally:
        (
            runtime_state.rotator,
            runtime_state.account_service,
            runtime_state.client,
            runtime_state.snapshot_cache,
        ) = original

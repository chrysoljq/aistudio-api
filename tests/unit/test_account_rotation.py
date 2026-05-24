import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from aistudio_api.api.state import runtime_state
from aistudio_api.application import account_client_pool
from aistudio_api.application import api_service_common
from aistudio_api.application.account_client_pool import AccountClientPool
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


def test_least_rate_limited_tie_breaks_by_recent_use():
    rotator = AccountRotator(
        _Store([_account("a"), _account("b"), _account("c")]),
        mode=RotationMode.LEAST_RATE_LIMITED,
    )
    rotator.record_success("a")

    selected = asyncio.run(rotator.get_next_account())

    assert selected is not None
    assert selected.id == "b"


def test_account_client_pool_skips_busy_entries(monkeypatch, tmp_path):
    accounts = [_account("a"), _account("b"), _account("c")]

    class _PoolStore(_Store):
        def get_auth_path_optional(self, account_id, *, require_exists=False):
            path = tmp_path / account_id / "auth.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text("{}")
            return path

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def warmup(self):
            return None

        async def close(self):
            return None

    store = _PoolStore(accounts)
    rotator = AccountRotator(store, mode=RotationMode.LEAST_RATE_LIMITED)
    monkeypatch.setattr(account_client_pool, "AIStudioClient", _FakeClient)
    pool = AccountClientPool(account_store=store, rotator=rotator, port=9222, size=3)

    async def _run():
        async with pool.acquire() as first:
            async with pool.acquire() as second:
                return first.account_id, second.account_id

    first_id, second_id = asyncio.run(_run())

    assert first_id == "a"
    assert second_id == "b"


def test_account_client_pool_respects_explicit_account_selectors(monkeypatch, tmp_path):
    accounts = [_account("a"), _account("b"), _account("c")]
    accounts[1].email = "b@example.com"

    class _PoolStore(_Store):
        def get_auth_path_optional(self, account_id, *, require_exists=False):
            path = tmp_path / account_id / "auth.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text("{}")
            return path

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def close(self):
            return None

    store = _PoolStore(accounts)
    rotator = AccountRotator(store, mode=RotationMode.ROUND_ROBIN)
    monkeypatch.setattr(account_client_pool, "AIStudioClient", _FakeClient)
    pool = AccountClientPool(
        account_store=store,
        rotator=rotator,
        port=9222,
        size=0,
        account_selectors=("c", "b@example.com"),
    )

    assert [account["id"] for account in pool.status()["accounts"]] == ["c", "b"]
    assert pool.status()["configured_accounts"] == ["c", "b@example.com"]


def test_record_rotator_event_uses_pooled_account_context():
    class _Pool:
        enabled = True

        @asynccontextmanager
        async def acquire(self):
            yield SimpleNamespace(account_id="pooled", client=object())

    class _Rotator:
        def __init__(self):
            self.successes = []

        def record_success(self, account_id):
            self.successes.append(account_id)

    rotator = _Rotator()
    original = (
        runtime_state.client_pool,
        runtime_state.rotator,
        runtime_state.account_service,
    )
    runtime_state.client_pool = _Pool()
    runtime_state.rotator = rotator
    runtime_state.account_service = None

    async def _run():
        async with api_service_common.request_client(object()) as leased_client:
            assert leased_client is not None
            api_service_common.record_rotator_event("success")

    try:
        asyncio.run(_run())
        assert rotator.successes == ["pooled"]
    finally:
        (
            runtime_state.client_pool,
            runtime_state.rotator,
            runtime_state.account_service,
        ) = original


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


def test_try_switch_account_skips_failed_activation():
    broken = _account("broken")
    working = _account("working")

    class _Rotator:
        def __init__(self):
            self.accounts = [broken, working]
            self.errors = []

        async def get_next_account(self):
            return self.accounts.pop(0)

        def get_all_stats(self):
            return {broken.id: {}, working.id: {}}

        def record_error(self, account_id):
            self.errors.append(account_id)

    class _AccountService:
        def __init__(self):
            self.activations = []

        def get_active_account(self):
            return None

        async def activate_account(self, account_id, *args, **kwargs):
            self.activations.append(account_id)
            if account_id == broken.id:
                raise RuntimeError("profile launch failed")
            return working

    rotator = _Rotator()
    account_service = _AccountService()
    original = (
        runtime_state.rotator,
        runtime_state.account_service,
        runtime_state.client,
        runtime_state.snapshot_cache,
    )
    runtime_state.rotator = rotator
    runtime_state.account_service = account_service
    runtime_state.client = SimpleNamespace(_session=object())
    runtime_state.snapshot_cache = object()
    try:
        assert asyncio.run(api_service_common.try_switch_account()) is True
        assert account_service.activations == ["broken", "working"]
        assert rotator.errors == ["broken"]
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


def test_ensure_active_account_respects_sticky_window(monkeypatch):
    active = _account("a")
    active.last_used = "2026-01-01T00:00:00+00:00"

    class _Rotator:
        async def get_next_account(self):
            raise AssertionError("sticky active account should skip rotation")

    class _AccountService:
        def get_active_account(self):
            return active

    class _FixedDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)

        @staticmethod
        def fromisoformat(value):
            return datetime.fromisoformat(value)

    from datetime import datetime, timezone

    original = (
        runtime_state.rotator,
        runtime_state.account_service,
        runtime_state.client,
        runtime_state.snapshot_cache,
    )
    monkeypatch.setattr(api_service_common.settings, "account_rotation_sticky_seconds", 60)
    monkeypatch.setattr(api_service_common, "datetime", _FixedDatetime)
    runtime_state.rotator = _Rotator()
    runtime_state.account_service = _AccountService()
    runtime_state.client = SimpleNamespace(_session=object())
    runtime_state.snapshot_cache = object()
    try:
        asyncio.run(api_service_common.ensure_active_account(0))
    finally:
        (
            runtime_state.rotator,
            runtime_state.account_service,
            runtime_state.client,
            runtime_state.snapshot_cache,
        ) = original

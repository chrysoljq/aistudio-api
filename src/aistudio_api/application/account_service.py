"""账号管理应用服务，协调 account_store 和 login_service。"""

from __future__ import annotations

import logging
from typing import Any

from aistudio_api.infrastructure.account.account_store import AccountStore, AccountMeta
from aistudio_api.infrastructure.account.login_service import LoginService, LoginSession

logger = logging.getLogger("aistudio.account")


class AccountService:
    """账号管理服务。"""

    def __init__(
        self,
        account_store: AccountStore,
        login_service: LoginService,
    ) -> None:
        self._store = account_store
        self._login = login_service

    def list_accounts(self) -> list[AccountMeta]:
        """列出所有账号。"""
        return self._store.list_accounts()

    def get_account(self, account_id: str) -> AccountMeta | None:
        """获取单个账号。"""
        return self._store.get_account(account_id)

    def get_active_account(self) -> AccountMeta | None:
        """获取当前活跃账号。"""
        return self._store.get_active_account()

    async def start_login(
        self,
        name: str | None = None,
        *,
        headless: bool = False,
        ui_locale: str | None = None,
    ) -> str:
        """启动登录流程，返回 session_id。"""
        return await self._login.start_login(
            self._store,
            name,
            headless=headless,
            ui_locale=ui_locale,
        )

    def get_login_status(self, session_id: str) -> LoginSession | None:
        """获取登录状态。"""
        return self._login.get_status(session_id)

    async def activate_account(
        self,
        account_id: str,
        browser_session: Any,
        busy_lock: Any = None,  # None = skip lock (caller already holds it)
    ) -> AccountMeta | None:
        """切换到指定账号。

        Args:
            account_id: 目标账号 ID
            browser_session: BrowserSession 实例
            busy_lock: asyncio.Lock，确保切换时无请求在飞行中。None 则跳过锁

        Returns:
            切换后的账号元数据，或 None（如果账号不存在）
        """
        # 验证账号存在
        account = self._store.get_account(account_id)
        if account is None:
            return None

        async def _do_switch():
            # 获取 auth 路径
            auth_path = self._store.get_auth_path_optional(account_id, require_exists=False)
            if auth_path is None:
                logger.error("账号 %s 的账号目录不存在", account_id)
                return None

            # 切换 BrowserSession 的 auth
            await browser_session.switch_auth(str(auth_path))
            await browser_session.ensure_context()

            # 更新注册表
            self._store.set_active_account(account_id)

            logger.info("已切换到账号: %s (%s)", account_id, account.name)
            return account

        # 获取 busy_lock 确保无请求在飞行中
        if busy_lock is not None:
            async with busy_lock:
                return await _do_switch()
        else:
            return await _do_switch()

    def delete_account(self, account_id: str) -> bool:
        """删除账号。"""
        return self._store.delete_account(account_id)

    def update_account(self, account_id: str, name: str) -> AccountMeta | None:
        """更新账号名称。"""
        return self._store.update_account(account_id, name)

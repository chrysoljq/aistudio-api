"""账号管理路由。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from aistudio_api.api.dependencies import get_account_service, get_runtime_state
from aistudio_api.infrastructure.account.cookie_parser import parse_cookie_string
import logging

log = logging.getLogger("aistudio.routes_accounts")

router = APIRouter(prefix="/accounts")


class LoginStartRequest(BaseModel):
    name: str | None = None


class LoginStartResponse(BaseModel):
    session_id: str


class AccountResponse(BaseModel):
    id: str
    name: str
    email: str | None
    created_at: str
    last_used: str | None


class LoginStatusResponse(BaseModel):
    session_id: str
    status: str
    account_id: str | None = None
    email: str | None = None
    error: str | None = None


class UpdateAccountRequest(BaseModel):
    name: str


class ImportCookiesRequest(BaseModel):
    cookies: str  # "key=value; key=value; ..." 格式
    name: str | None = None  # 可选的账号名称
    email: str | None = None  # 可选的邮箱
    account_id: str | None = None  # 可选的账号 ID（覆盖已有账号）


class ImportCookiesResponse(BaseModel):
    account_id: str
    name: str
    cookie_count: int
    domain_summary: dict[str, int]  # domain -> cookie 数量


@router.post("/login/start", response_model=LoginStartResponse)
async def login_start(
    req: LoginStartRequest,
    account_service=Depends(get_account_service),
):
    """启动 Google 登录流程。"""
    session_id = await account_service.start_login(req.name)
    return LoginStartResponse(session_id=session_id)


@router.get("/login/status/{session_id}", response_model=LoginStatusResponse)
async def login_status(
    session_id: str,
    account_service=Depends(get_account_service),
):
    """查询登录状态。"""
    session = account_service.get_login_status(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="登录会话不存在")
    return LoginStatusResponse(
        session_id=session.session_id,
        status=session.status.value,
        account_id=session.account_id,
        email=session.email,
        error=session.error,
    )


@router.get("", response_model=list[AccountResponse])
async def list_accounts(
    account_service=Depends(get_account_service),
):
    """列出所有账号。"""
    accounts = account_service.list_accounts()
    return [
        AccountResponse(
            id=a.id,
            name=a.name,
            email=a.email,
            created_at=a.created_at,
            last_used=a.last_used,
        )
        for a in accounts
    ]


@router.get("/active", response_model=AccountResponse)
async def get_active_account(
    account_service=Depends(get_account_service),
):
    """获取当前活跃账号。"""
    account = account_service.get_active_account()
    if account is None:
        raise HTTPException(status_code=404, detail="没有活跃账号")
    return AccountResponse(
        id=account.id,
        name=account.name,
        email=account.email,
        created_at=account.created_at,
        last_used=account.last_used,
    )


@router.post("/{account_id}/activate", response_model=AccountResponse)
async def activate_account(
    account_id: str,
    account_service=Depends(get_account_service),
    runtime_state=Depends(get_runtime_state),
):
    """切换到指定账号。"""
    # 从 runtime_state 获取 browser_session 和 busy_lock
    browser_session = runtime_state.client._session if runtime_state.client else None
    busy_lock = runtime_state.busy_lock

    if browser_session is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    account = await account_service.activate_account(account_id, browser_session, busy_lock)
    if account is None:
        raise HTTPException(status_code=404, detail="账号不存在或切换失败")
    return AccountResponse(
        id=account.id,
        name=account.name,
        email=account.email,
        created_at=account.created_at,
        last_used=account.last_used,
    )


@router.delete("/{account_id}")
async def delete_account(
    account_id: str,
    account_service=Depends(get_account_service),
):
    """删除账号。"""
    success = account_service.delete_account(account_id)
    if not success:
        raise HTTPException(status_code=404, detail="账号不存在")
    return {"ok": True}


@router.put("/{account_id}", response_model=AccountResponse)
async def update_account(
    account_id: str,
    req: UpdateAccountRequest,
    account_service=Depends(get_account_service),
):
    """更新账号名称。"""
    account = account_service.update_account(account_id, req.name)
    if account is None:
        raise HTTPException(status_code=404, detail="账号不存在")
    return AccountResponse(
        id=account.id,
        name=account.name,
        email=account.email,
        created_at=account.created_at,
        last_used=account.last_used,
    )


@router.post("/import-cookies", response_model=ImportCookiesResponse)
async def import_cookies(
    req: ImportCookiesRequest,
    account_service=Depends(get_account_service),
    runtime_state=Depends(get_runtime_state),
):
    """从 cookie 字符串导入账号。

    支持格式: `key=value; key=value; ...`（浏览器开发者工具或 Cookie 编辑扩展导出格式）

    流程: 解析 → 保存到 store → 注入浏览器 → 访问页面 → 导出 auth.json
    """
    # 1. 解析并保存到账号 store
    storage_state = parse_cookie_string(req.cookies)
    cookie_count = len(storage_state["cookies"])

    if cookie_count == 0:
        raise HTTPException(status_code=400, detail="未解析到有效 cookie")

    domain_summary: dict[str, int] = {}
    for c in storage_state["cookies"]:
        d = c["domain"]
        domain_summary[d] = domain_summary.get(d, 0) + 1

    name = req.name or "导入的账号"

    account = account_service._store.save_account(
        name=name,
        email=req.email,
        storage_state=storage_state,
        account_id=req.account_id,
    )

    # 2. 注入浏览器 + 访问页面 + 保存 auth.json
    try:
        browser_session = runtime_state.client._session if runtime_state.client else None
        if browser_session:
            auth_path = account_service._store.get_auth_path(account.id)
            count = await browser_session.import_cookies(
                req.cookies,
                auth_file=str(auth_path) if auth_path else None,
            )
            log.info("[import-cookies] injected %d cookies, saved auth.json", count)
    except Exception as e:
        log.warning("[import-cookies] browser injection failed: %s", e)

    return ImportCookiesResponse(
        account_id=account.id,
        name=account.name,
        cookie_count=cookie_count,
        domain_summary=domain_summary,
    )

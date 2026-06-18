"""System and metadata routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from aistudio_api.application.api_service import health_response, stats_response
from aistudio_api.api.response_models import HealthResponse, StatsResponse
from aistudio_api.api.dependencies import get_runtime_state

public_router = APIRouter()
protected_router = APIRouter()


@public_router.get("/health", response_model=HealthResponse)
async def health():
    return health_response()


@protected_router.get("/stats", response_model=StatsResponse)
async def stats():
    return stats_response()


# ========== 模型配置 API ==========

@protected_router.get("/config/model-defaults")
async def get_model_defaults_config():
    """获取模型默认配置。"""
    import yaml
    from aistudio_api.infrastructure.gateway.model_defaults import (
        _resolve_config_path,
        _default_config,
    )

    path = _resolve_config_path(None)
    if not path.exists():
        return _default_config().get("model_defaults", {})
    try:
        content = yaml.safe_load(path.read_text()) or {}
        return content.get("model_defaults", {})
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(500, detail=f"读取配置失败: {e}")


class ModelDefaultsConfigPayload(BaseModel):
    profiles: list[dict]
    models: dict[str, dict]


@protected_router.post("/config/model-defaults")
async def save_model_defaults_config(payload: ModelDefaultsConfigPayload):
    """保存模型默认配置并清理缓存。"""
    import yaml
    from aistudio_api.infrastructure.gateway.model_defaults import (
        _resolve_config_path,
        _compiled_profiles,
        _compiled_model_overrides,
    )

    path = _resolve_config_path(None)
    data = {
        "model_defaults": {
            "profiles": payload.profiles,
            "models": payload.models,
        }
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                data,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        _compiled_profiles.cache_clear()
        _compiled_model_overrides.cache_clear()
        return {"ok": True}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(500, detail=f"保存配置失败: {e}")


# ========== 轮询管理 API ==========

class RotationModeRequest(BaseModel):
    mode: str  # round_robin, lru, least_rl
    cooldown_seconds: int | None = None


@protected_router.get("/rotation")
async def get_rotation_status(runtime_state=Depends(get_runtime_state)):
    """获取轮询状态。"""
    rotator = runtime_state.rotator
    if rotator is None:
        return {"enabled": False, "message": "轮询器未初始化"}

    return {
        "enabled": True,
        "mode": rotator.mode.value,
        "cooldown_seconds": rotator.cooldown_seconds,
        "accounts": rotator.get_all_stats(),
    }


@protected_router.post("/rotation/mode")
async def set_rotation_mode(
    req: RotationModeRequest,
    runtime_state=Depends(get_runtime_state),
):
    """设置轮询模式。"""
    rotator = runtime_state.rotator
    if rotator is None:
        raise HTTPException(503, detail="轮询器未初始化")

    try:
        from aistudio_api.application.account_rotator import RotationMode
        rotator.mode = RotationMode(req.mode)
        if req.cooldown_seconds is not None:
            rotator.cooldown_seconds = req.cooldown_seconds
        return {
            "ok": True,
            "mode": rotator.mode.value,
            "cooldown_seconds": rotator.cooldown_seconds,
        }
    except ValueError:
        raise HTTPException(400, detail=f"无效的轮询模式: {req.mode}，可选: round_robin, lru, least_rl")


@protected_router.get("/rotation/accounts")
async def get_rotation_accounts(runtime_state=Depends(get_runtime_state)):
    """获取所有账号的轮询统计。"""
    rotator = runtime_state.rotator
    if rotator is None:
        raise HTTPException(503, detail="轮询器未初始化")

    return rotator.get_all_stats()


@protected_router.post("/rotation/next")
async def force_next_account(runtime_state=Depends(get_runtime_state)):
    """强制切换到下一个可用账号。"""
    rotator = runtime_state.rotator
    if rotator is None:
        raise HTTPException(503, detail="轮询器未初始化")

    # 获取下一个账号
    next_account = await rotator.get_next_account()
    if next_account is None:
        raise HTTPException(404, detail="没有可用的账号")

    # 切换账号
    account_service = runtime_state.account_service
    client = runtime_state.client
    busy_lock = runtime_state.busy_lock

    if not all([account_service, client, busy_lock]):
        raise HTTPException(503, detail="服务未就绪")

    result = await account_service.activate_account(
        next_account.id,
        client._session,
        busy_lock,
    )

    if result is None:
        raise HTTPException(500, detail="切换失败")

    return {
        "ok": True,
        "account": {
            "id": result.id,
            "name": result.name,
            "email": result.email,
        },
    }

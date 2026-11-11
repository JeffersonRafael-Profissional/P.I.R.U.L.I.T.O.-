from __future__ import annotations

from typing import Any

from .core import BudgetPolicy, ModelTier
from .ledger import BudgetLedger
from .runtime import AdaptiveBudgetRuntime
from .schemas import ADAPTIVE_BUDGET_SCHEMA


def _float_config(ctx: Any, key: str, default: float) -> float:
    value = ctx.get_config(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_config(ctx: Any, key: str, default: int) -> int:
    value = ctx.get_config(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_config(ctx: Any, key: str, default: bool) -> bool:
    value = ctx.get_config(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _tier(ctx: Any, name: str, rank: int) -> ModelTier:
    return ModelTier(
        name=name,
        model=str(ctx.get_config(f"{name}_model", "") or "").strip(),
        input_cost_per_million=_float_config(ctx, f"{name}_input_usd_per_million", 0.0),
        output_cost_per_million=_float_config(ctx, f"{name}_output_usd_per_million", 0.0),
        quality_rank=rank,
    )


def build_runtime(ctx: Any) -> AdaptiveBudgetRuntime:
    policy = BudgetPolicy(
        session_budget_usd=max(0.0, _float_config(ctx, "session_budget_usd", 0.50)),
        daily_budget_usd=max(0.0, _float_config(ctx, "daily_budget_usd", 5.00)),
        reserve_ratio=min(0.95, max(0.0, _float_config(ctx, "reserve_ratio", 0.10))),
        cost_safety_multiplier=max(1.0, _float_config(ctx, "cost_safety_multiplier", 4.0)),
        expected_output_tokens=max(1, _int_config(ctx, "expected_output_tokens", 800)),
        balanced_threshold=max(0, min(100, _int_config(ctx, "balanced_threshold", 35))),
        premium_threshold=max(0, min(100, _int_config(ctx, "premium_threshold", 65))),
        hard_stop=_bool_config(ctx, "hard_stop", True),
    )
    allowed = str(ctx.get_config("allowed_providers", "openrouter") or "")
    state = getattr(ctx, "state", None)
    data_dir = getattr(state, "data_dir", None)
    ledger_path = data_dir / "budget-router.sqlite3" if data_dir is not None else ":memory:"
    return AdaptiveBudgetRuntime(
        tiers=(_tier(ctx, "economy", 1), _tier(ctx, "balanced", 2), _tier(ctx, "premium", 3)),
        policy=policy,
        ledger=BudgetLedger(ledger_path),
        allowed_providers=(item.strip() for item in allowed.split(",")),
        enabled=_bool_config(ctx, "enabled", True),
    )


def register(ctx: Any) -> None:
    runtime = build_runtime(ctx)
    ctx.register_middleware("llm_request", runtime.on_llm_request)
    ctx.register_middleware("llm_execution", runtime.on_llm_execution)
    ctx.register_hook("pre_llm_call", runtime.on_pre_llm_call)
    ctx.register_hook("post_api_request", runtime.on_post_api_request)
    ctx.register_hook("api_request_error", runtime.on_api_request_error)
    ctx.register_tool(
        name="adaptive_budget",
        toolset="adaptive_budget_router",
        schema=ADAPTIVE_BUDGET_SCHEMA,
        handler=runtime.handle_tool,
        description="Inspect budget-aware routing state and simulate model-tier decisions.",
        emoji="💸",
    )
    ctx.register_command(
        "pirulito",
        handler=runtime.handle_command,
        description="Inspect, explain, or explicitly reset session accounting (P.I.R.U.L.I.T.O.).",
        args_hint="[status|explain|reset-session]",
    )
    ctx.register_command(
        "budget-router",
        handler=runtime.handle_command,
        description="Inspect, explain, or explicitly reset session accounting.",
        args_hint="[status|explain|reset-session]",
    )
    ctx.on_unload(runtime.close)


__all__ = ["register", "build_runtime", "AdaptiveBudgetRuntime", "BudgetPolicy", "ModelTier"]

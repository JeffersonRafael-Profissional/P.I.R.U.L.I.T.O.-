from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from typing import Any, Iterable, Mapping

from .core import BudgetPolicy, ModelTier, RouteDecision, choose_route, estimate_cost, remaining_budget
from .ledger import BudgetLedger


class BudgetExceededError(RuntimeError):
    """Raised before provider execution when no configured tier can be reserved safely."""


class AdaptiveBudgetRuntime:
    def __init__(
        self,
        *,
        tiers: Iterable[ModelTier],
        policy: BudgetPolicy,
        ledger: BudgetLedger,
        allowed_providers: Iterable[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.configured_tiers = tuple(sorted(tiers, key=lambda item: item.quality_rank))
        self.tiers = tuple(tier for tier in self.configured_tiers if tier.enabled)
        self.policy = policy
        self.ledger = ledger
        self.allowed_providers = {value.strip().lower() for value in (allowed_providers or ()) if value.strip()}
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._pending: dict[str, RouteDecision] = {}

    def _provider_allowed(self, provider: str) -> bool:
        if not self.allowed_providers or not provider:
            return True
        return "*" in self.allowed_providers or provider.strip().lower() in self.allowed_providers

    def _summary(self, session_id: str) -> dict[str, Any]:
        return self.ledger.summary(session_id or "default")

    def _route(self, request: Mapping[str, Any], session_id: str) -> RouteDecision:
        summary = self._summary(session_id)
        return choose_route(
            request,
            self.configured_tiers,
            self.policy,
            session_spent_usd=summary["session_spend_usd"] + summary["session_reserved_usd"],
            daily_spent_usd=summary["daily_spend_usd"] + summary["daily_reserved_usd"],
        )

    def on_llm_request(
        self,
        *,
        request: Mapping[str, Any],
        session_id: str = "",
        api_request_id: str = "",
        provider: str = "",
        **_: Any,
    ) -> dict[str, Any] | None:
        if not self.enabled or not self.tiers or not self._provider_allowed(provider):
            return None
        decision = self._route(request, session_id or "default")
        routed_request = dict(request)
        routed_request["model"] = decision.model
        if api_request_id:
            with self._lock:
                self._pending[api_request_id] = decision
        return {
            "request": routed_request,
            "source": "pirulito",
            "reason": f"{decision.reason}:{decision.tier}:score={decision.complexity_score}",
        }

    def on_llm_execution(
        self,
        *,
        request: Mapping[str, Any],
        next_call: Any,
        session_id: str = "",
        api_request_id: str = "",
        **_: Any,
    ) -> Any:
        if not self.enabled or not api_request_id:
            return next_call(dict(request))

        sid = session_id or "default"
        with self._lock:
            decision = self._pending.get(api_request_id)
        if decision is None:
            return next_call(dict(request))

        for _attempt in range(max(2, len(self.tiers) + 1)):
            projected_reservation = (
                decision.projected_cost_usd * max(1.0, self.policy.cost_safety_multiplier)
            )
            if self.ledger.try_reserve(
                api_request_id,
                sid,
                projected_reservation,
                self.policy.session_budget_usd,
                self.policy.daily_budget_usd,
                reserve_ratio=self.policy.reserve_ratio,
            ):
                routed_request = dict(request)
                routed_request["model"] = decision.model
                with self._lock:
                    self._pending[api_request_id] = decision
                try:
                    return next_call(routed_request)
                except BaseException:
                    self.ledger.release_reservation(api_request_id)
                    with self._lock:
                        self._pending.pop(api_request_id, None)
                    raise

            refreshed = self._route(request, sid)
            if refreshed == decision or refreshed.over_budget_risk:
                decision = refreshed
                break
            decision = refreshed

        with self._lock:
            self._pending.pop(api_request_id, None)
        if self.policy.hard_stop:
            raise BudgetExceededError(
                "Adaptive Budget Router blocked provider execution because no configured tier "
                "fits the remaining session and daily budget."
            )
        routed_request = dict(request)
        routed_request["model"] = decision.model
        return next_call(routed_request)

    def _tier_for(self, model: str, decision: RouteDecision | None) -> ModelTier | None:
        if decision is not None:
            for tier in self.tiers:
                if tier.name == decision.tier:
                    return tier
        for tier in self.tiers:
            if tier.model == model:
                return tier
        return None

    @staticmethod
    def _usage_tokens(
        usage: Mapping[str, Any] | None, decision: RouteDecision | None = None,
    ) -> tuple[int, int]:
        fallback_input = decision.estimated_input_tokens if decision is not None else 0
        fallback_output = decision.estimated_output_tokens if decision is not None else 0
        if not isinstance(usage, Mapping):
            return fallback_input, fallback_output
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", fallback_input))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", fallback_output))
        try:
            parsed_input = max(0, int(input_tokens if input_tokens is not None else fallback_input))
            parsed_output = max(0, int(output_tokens if output_tokens is not None else fallback_output))
            return parsed_input or fallback_input, parsed_output or fallback_output
        except (TypeError, ValueError):
            return fallback_input, fallback_output

    def on_post_api_request(
        self,
        *,
        session_id: str = "",
        api_request_id: str = "",
        usage: Mapping[str, Any] | None = None,
        response_model: str = "",
        ended_at: float | None = None,
        **_: Any,
    ) -> None:
        with self._lock:
            decision = self._pending.pop(api_request_id, None)
        tier = self._tier_for(response_model, decision)
        if tier is None or not api_request_id:
            return
        input_tokens, output_tokens = self._usage_tokens(usage, decision)
        self.ledger.record_success(
            api_request_id,
            session_id or "default",
            response_model or tier.model,
            tier.name,
            input_tokens,
            output_tokens,
            estimate_cost(tier, input_tokens, output_tokens),
            ended_at,
        )

    def on_api_request_error(
        self,
        *,
        session_id: str = "",
        api_request_id: str = "",
        model: str = "",
        ended_at: float | None = None,
        **_: Any,
    ) -> None:
        del session_id, model, ended_at
        # Failed attempts have no billable usage payload. Drop only the pending routing decision;
        # do not reserve the idempotency key because some providers reuse it for a successful retry.
        with self._lock:
            self._pending.pop(api_request_id, None)
        self.ledger.release_reservation(api_request_id)

    def on_pre_llm_call(self, *, session_id: str = "", **_: Any) -> dict[str, str] | None:
        if not self.enabled or not self.policy.hard_stop:
            return None
        summary = self._summary(session_id or "default")
        exhausted: list[str] = []
        session_committed = summary["session_spend_usd"] + summary["session_reserved_usd"]
        daily_committed = summary["daily_spend_usd"] + summary["daily_reserved_usd"]
        if self.policy.session_budget_usd > 0 and session_committed >= self.policy.session_budget_usd:
            exhausted.append("session")
        if self.policy.daily_budget_usd > 0 and daily_committed >= self.policy.daily_budget_usd:
            exhausted.append("daily")
        if not exhausted:
            return None
        return {
            "action": "block",
            "message": (
                "Adaptive Budget Router blocked this turn because the "
                + " and ".join(exhausted)
                + " budget is exhausted. Use /budget-router status or reset-session explicitly."
            ),
        }

    def status(self, session_id: str) -> dict[str, Any]:
        summary = self._summary(session_id or "default")
        remaining = remaining_budget(
            self.policy,
            session_spent_usd=summary["session_spend_usd"] + summary["session_reserved_usd"],
            daily_spent_usd=summary["daily_spend_usd"] + summary["daily_reserved_usd"],
        )
        return {
            "status": "ok",
            **summary,
            "routing": {
                "enabled": self.enabled,
                "configured_tiers": [tier.name for tier in self.tiers],
                "session_budget_usd": self.policy.session_budget_usd,
                "daily_budget_usd": self.policy.daily_budget_usd,
                "remaining_budget_usd": remaining,
                "hard_stop": self.policy.hard_stop,
            },
        }

    def explain(self, prompt: str, session_id: str) -> dict[str, Any]:
        if not self.tiers:
            return {"status": "inactive", "reason": "no model tiers configured"}
        decision = self._route({"messages": [{"role": "user", "content": prompt}]}, session_id or "default")
        return {"status": "ok", "decision": asdict(decision)}

    def handle_tool(self, args: Mapping[str, Any], *, session_id: str = "", **_: Any) -> str:
        action = str(args.get("action") or "status").strip().lower()
        sid = str(session_id or "default")
        if action == "status":
            payload = self.status(sid)
        elif action == "explain":
            payload = self.explain(str(args.get("prompt") or ""), sid)
        else:
            payload = {"status": "error", "error": f"unknown action: {action}"}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _active_session_id() -> str:
        try:
            from gateway.session_context import get_session_env

            return str(get_session_env("HERMES_SESSION_ID", "") or "default")
        except Exception:
            return str(os.getenv("HERMES_SESSION_ID") or "default")

    def handle_command(self, raw_args: str) -> str:
        sid = self._active_session_id()
        command, _, rest = raw_args.strip().partition(" ")
        command = command.lower() or "status"
        if command in {"help", "-h", "--help"}:
            return (
                "/budget-router status\n"
                "/budget-router explain <prompt>\n"
                "/budget-router reset-session"
            )
        if command == "status":
            return json.dumps(self.status(sid), ensure_ascii=False, indent=2, sort_keys=True)
        if command == "explain":
            return json.dumps(self.explain(rest, sid), ensure_ascii=False, indent=2, sort_keys=True)
        if command == "reset-session":
            return f"Adaptive Budget Router removed {self.ledger.reset_session(sid)} request record(s) for this session."
        return "Unknown subcommand. Use /budget-router help."

    def close(self) -> None:
        self.ledger.close()

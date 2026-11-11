from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_REASONING_TERMS = {
    "analyze", "analyse", "architecture", "benchmark", "compare", "debug",
    "design", "distributed", "formal", "implement", "migration", "optimize",
    "plan", "prove", "reason", "refactor", "security", "strategy", "test",
    "verify",
}
_CODE_MARKERS = ("```", "def ", "class ", "function ", "SELECT ", "CREATE TABLE", "import ")


@dataclass(frozen=True)
class ModelTier:
    name: str
    model: str
    input_cost_per_million: float
    output_cost_per_million: float
    quality_rank: int

    @property
    def enabled(self) -> bool:
        return (
            bool(self.model.strip())
            and self.input_cost_per_million >= 0
            and self.output_cost_per_million >= 0
            and (self.input_cost_per_million > 0 or self.output_cost_per_million > 0)
        )


@dataclass(frozen=True)
class BudgetPolicy:
    session_budget_usd: float = 0.50
    daily_budget_usd: float = 5.00
    reserve_ratio: float = 0.10
    cost_safety_multiplier: float = 4.0
    expected_output_tokens: int = 800
    balanced_threshold: int = 35
    premium_threshold: int = 65
    hard_stop: bool = True


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    model: str
    reason: str
    complexity_score: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    projected_cost_usd: float
    remaining_budget_usd: float | None
    over_budget_risk: bool


def _json_chars(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))
    except Exception:
        return len(str(value))


def _message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            chunks.append(str(message))
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        elif content is not None:
            chunks.append(str(content))
    return "\n".join(chunks)


def estimate_request_tokens(request: Mapping[str, Any]) -> int:
    """Cheap, deterministic estimate used only for routing and pre-call cost projection."""
    text_chars = len(_message_text(request.get("messages") or request.get("input")))
    tool_chars = _json_chars(request.get("tools") or [])
    instruction_chars = len(str(request.get("instructions") or request.get("system") or ""))
    return max(1, math.ceil((text_chars + tool_chars + instruction_chars) / 4))


def score_request_complexity(request: Mapping[str, Any]) -> int:
    text = _message_text(request.get("messages") or request.get("input"))
    lowered = text.lower()
    words = set(re.findall(r"[a-zA-Z][a-zA-Z_-]+", lowered))
    token_estimate = estimate_request_tokens(request)
    tools = request.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0

    length_score = min(45.0, token_estimate / 8.0)
    reasoning_score = min(28.0, sum(1 for term in _REASONING_TERMS if term in words) * 4.0)
    tool_score = min(20.0, tool_count * 2.0)
    code_score = 10.0 if any(marker.lower() in lowered for marker in _CODE_MARKERS) else 0.0
    structure_score = 5.0 if len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", text)) >= 3 else 0.0
    return max(0, min(100, round(length_score + reasoning_score + tool_score + code_score + structure_score)))


def estimate_cost(tier: ModelTier, input_tokens: int, output_tokens: int) -> float:
    return (
        max(0, input_tokens) * max(0.0, tier.input_cost_per_million)
        + max(0, output_tokens) * max(0.0, tier.output_cost_per_million)
    ) / 1_000_000


def remaining_budget(
    policy: BudgetPolicy, *, session_spent_usd: float, daily_spent_usd: float,
) -> float | None:
    remaining: list[float] = []
    if policy.session_budget_usd > 0:
        remaining.append(max(0.0, policy.session_budget_usd - max(0.0, session_spent_usd)))
    if policy.daily_budget_usd > 0:
        remaining.append(max(0.0, policy.daily_budget_usd - max(0.0, daily_spent_usd)))
    return min(remaining) if remaining else None


def _desired_rank(score: int, policy: BudgetPolicy, configured: Sequence[ModelTier]) -> int:
    ranks = sorted({tier.quality_rank for tier in configured})
    if score >= policy.premium_threshold:
        return ranks[-1]
    if score >= policy.balanced_threshold:
        return ranks[min(1, len(ranks) - 1)]
    return ranks[0]


def _desired_tier(
    score: int, policy: BudgetPolicy, configured: Sequence[ModelTier], enabled: Sequence[ModelTier],
) -> ModelTier:
    target_rank = _desired_rank(score, policy, configured)
    at_or_below = [tier for tier in enabled if tier.quality_rank <= target_rank]
    return max(at_or_below or enabled, key=lambda tier: tier.quality_rank)


def choose_route(
    request: Mapping[str, Any],
    tiers: Iterable[ModelTier],
    policy: BudgetPolicy,
    *,
    session_spent_usd: float,
    daily_spent_usd: float,
) -> RouteDecision:
    configured = tuple(tiers)
    enabled = tuple(sorted((tier for tier in configured if tier.enabled), key=lambda tier: tier.quality_rank))
    if not enabled:
        raise ValueError("At least one model tier must have a configured model")

    input_tokens = estimate_request_tokens(request)
    raw_max_tokens = request.get("max_tokens") or request.get("max_completion_tokens")
    output_tokens = policy.expected_output_tokens
    if isinstance(raw_max_tokens, int) and not isinstance(raw_max_tokens, bool) and raw_max_tokens > 0:
        output_tokens = max(output_tokens, raw_max_tokens)

    score = score_request_complexity(request)
    desired = _desired_tier(score, policy, configured, enabled)
    remaining = remaining_budget(
        policy, session_spent_usd=session_spent_usd, daily_spent_usd=daily_spent_usd,
    )

    candidates = [tier for tier in enabled if tier.quality_rank <= desired.quality_rank]
    chosen = desired
    reason = "complexity_match"
    over_budget_risk = False

    if remaining is not None:
        spendable = max(0.0, remaining * (1.0 - min(0.95, max(0.0, policy.reserve_ratio))))
        fitting = [
            tier for tier in candidates
            if estimate_cost(tier, input_tokens, output_tokens)
            * max(1.0, policy.cost_safety_multiplier) <= spendable
        ]
        if fitting:
            chosen = max(fitting, key=lambda tier: tier.quality_rank)
            if chosen.quality_rank < desired.quality_rank:
                reason = "budget_downgrade"
        else:
            chosen = min(enabled, key=lambda tier: tier.quality_rank)
            reason = "budget_downgrade"
            over_budget_risk = True

    return RouteDecision(
        tier=chosen.name,
        model=chosen.model,
        reason=reason,
        complexity_score=score,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        projected_cost_usd=estimate_cost(chosen, input_tokens, output_tokens),
        remaining_budget_usd=remaining,
        over_budget_risk=over_budget_risk,
    )

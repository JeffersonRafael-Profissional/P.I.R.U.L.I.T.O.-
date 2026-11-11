from __future__ import annotations

import unittest

from adaptive_budget_router.core import (
    BudgetPolicy,
    ModelTier,
    choose_route,
    estimate_request_tokens,
    score_request_complexity,
)


def tiers() -> tuple[ModelTier, ...]:
    return (
        ModelTier("economy", "openai/gpt-5-nano", 0.05, 0.40, 1),
        ModelTier("balanced", "openai/gpt-5-mini", 0.25, 2.00, 2),
        ModelTier("premium", "openai/gpt-5.2", 1.75, 14.00, 3),
    )


class ComplexityTests(unittest.TestCase):
    def test_complex_prompt_scores_higher_than_simple_prompt(self) -> None:
        simple = {"messages": [{"role": "user", "content": "Translate hello to Portuguese."}]}
        complex_request = {
            "messages": [{
                "role": "user",
                "content": (
                    "Analyze this distributed system architecture, compare three alternatives, "
                    "prove the safety properties, identify race conditions, and implement a migration plan. "
                    "```python\nasync def reconcile(state): ...\n```"
                ),
            }],
            "tools": [{"type": "function", "function": {"name": "terminal"}} for _ in range(8)],
        }

        self.assertLess(score_request_complexity(simple), score_request_complexity(complex_request))

    def test_token_estimate_counts_text_and_tool_schema(self) -> None:
        without_tools = {"messages": [{"role": "user", "content": "x" * 400}]}
        with_tools = {
            **without_tools,
            "tools": [{"type": "function", "function": {"name": "terminal", "description": "y" * 400}}],
        }

        self.assertGreaterEqual(estimate_request_tokens(without_tools), 100)
        self.assertGreater(estimate_request_tokens(with_tools), estimate_request_tokens(without_tools))


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = BudgetPolicy(
            session_budget_usd=1.0,
            daily_budget_usd=5.0,
            reserve_ratio=0.10,
            expected_output_tokens=500,
            balanced_threshold=35,
            premium_threshold=65,
        )

    def test_simple_request_uses_economy_tier(self) -> None:
        request = {"messages": [{"role": "user", "content": "Say hello."}]}

        decision = choose_route(request, tiers(), self.policy, session_spent_usd=0, daily_spent_usd=0)

        self.assertEqual(decision.tier, "economy")
        self.assertEqual(decision.model, "openai/gpt-5-nano")
        self.assertEqual(decision.reason, "complexity_match")

    def test_complex_request_uses_premium_when_budget_is_healthy(self) -> None:
        request = {
            "messages": [{"role": "user", "content": "Deeply analyze and implement a production-grade distributed consensus protocol with formal verification, tests, benchmarks, and a migration strategy." * 8}],
            "tools": [{"type": "function", "function": {"name": f"tool_{i}"}} for i in range(12)],
        }

        decision = choose_route(request, tiers(), self.policy, session_spent_usd=0, daily_spent_usd=0)

        self.assertEqual(decision.tier, "premium")
        self.assertGreaterEqual(decision.complexity_score, self.policy.premium_threshold)

    def test_router_downgrades_when_desired_tier_would_break_budget_reserve(self) -> None:
        request = {
            "messages": [{"role": "user", "content": "Analyze and implement a robust architecture with tests and migration details." * 20}],
            "max_tokens": 4000,
        }

        decision = choose_route(
            request,
            tiers(),
            self.policy,
            session_spent_usd=0.97,
            daily_spent_usd=0.0,
        )

        self.assertEqual(decision.tier, "economy")
        self.assertEqual(decision.reason, "budget_downgrade")
        self.assertFalse(decision.over_budget_risk)

    def test_zero_budget_caps_mean_unlimited(self) -> None:
        unlimited = BudgetPolicy(
            session_budget_usd=0,
            daily_budget_usd=0,
            reserve_ratio=0.25,
            expected_output_tokens=500,
            balanced_threshold=20,
            premium_threshold=40,
        )
        request = {"messages": [{"role": "user", "content": "Design, analyze, prove, benchmark, and implement." * 30}]}

        decision = choose_route(request, tiers(), unlimited, session_spent_usd=999, daily_spent_usd=999)

        self.assertEqual(decision.tier, "premium")
        self.assertIsNone(decision.remaining_budget_usd)

    def test_disabled_tier_is_skipped(self) -> None:
        configured = (
            ModelTier("economy", "cheap", 0.1, 0.2, 1),
            ModelTier("balanced", "", 0.2, 0.4, 2),
            ModelTier("premium", "premium", 1.0, 2.0, 3),
        )
        request = {"messages": [{"role": "user", "content": "Compare alternatives and produce a careful implementation plan." * 6}]}

        decision = choose_route(request, configured, self.policy, session_spent_usd=0, daily_spent_usd=0)

        self.assertNotEqual(decision.tier, "balanced")
        self.assertTrue(decision.model)

    def test_disabled_balanced_tier_does_not_promote_balanced_work_to_premium(self) -> None:
        configured = (
            ModelTier("economy", "cheap", 0.1, 0.2, 1),
            ModelTier("balanced", "", 0.2, 0.4, 2),
            ModelTier("premium", "premium", 1.0, 2.0, 3),
        )
        request = {"messages": [{
            "role": "user",
            "content": (
                "Analyze and compare this distributed architecture, implement a migration plan, "
                "test security, and verify benchmarks. " * 2
            ),
        }]}

        decision = choose_route(request, configured, self.policy, session_spent_usd=0, daily_spent_usd=0)

        self.assertGreaterEqual(decision.complexity_score, self.policy.balanced_threshold)
        self.assertLess(decision.complexity_score, self.policy.premium_threshold)
        self.assertEqual(decision.tier, "economy")

    def test_safe_budget_downgrade_is_not_marked_as_over_budget_risk(self) -> None:
        policy = BudgetPolicy(
            session_budget_usd=0.01,
            daily_budget_usd=1.0,
            reserve_ratio=0,
            cost_safety_multiplier=1,
            expected_output_tokens=1_000,
            balanced_threshold=10,
            premium_threshold=20,
        )
        configured = (
            ModelTier("economy", "cheap", 0.1, 0.2, 1),
            ModelTier("balanced", "balanced", 1.0, 2.0, 2),
            ModelTier("premium", "premium", 10.0, 20.0, 3),
        )
        request = {"messages": [{"role": "user", "content": "Analyze, compare, design, and implement." * 20}]}

        decision = choose_route(request, configured, policy, session_spent_usd=0, daily_spent_usd=0)

        self.assertEqual(decision.reason, "budget_downgrade")
        self.assertEqual(decision.tier, "balanced")
        self.assertFalse(decision.over_budget_risk)

    def test_tier_with_missing_prices_is_disabled(self) -> None:
        tier = ModelTier("economy", "cheap", 0.0, 0.0, 1)

        self.assertFalse(tier.enabled)


if __name__ == "__main__":
    unittest.main()

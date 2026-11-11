from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptive_budget_router import register
from adaptive_budget_router.core import BudgetPolicy, ModelTier
from adaptive_budget_router.ledger import BudgetLedger
from adaptive_budget_router.runtime import AdaptiveBudgetRuntime, BudgetExceededError


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.ledger = BudgetLedger(Path(self.tempdir.name) / "ledger.sqlite3")
        self.runtime = AdaptiveBudgetRuntime(
            tiers=(
                ModelTier("economy", "cheap", 1.0, 2.0, 1),
                ModelTier("premium", "premium", 10.0, 20.0, 2),
            ),
            policy=BudgetPolicy(
                session_budget_usd=1.0,
                daily_budget_usd=10.0,
                reserve_ratio=0,
                expected_output_tokens=100,
                balanced_threshold=30,
                premium_threshold=60,
                hard_stop=True,
            ),
            ledger=self.ledger,
        )

    def tearDown(self) -> None:
        self.runtime.close()
        self.tempdir.cleanup()

    def test_llm_request_rewrites_model_without_mutating_original(self) -> None:
        request = {"model": "original", "messages": [{"role": "user", "content": "hello"}]}

        result = self.runtime.on_llm_request(
            request=request,
            session_id="s1",
            api_request_id="req-1",
            provider="openrouter",
        )

        self.assertEqual(request["model"], "original")
        self.assertEqual(result["request"]["model"], "cheap")
        self.assertEqual(result["source"], "pirulito")

    def test_post_api_request_accounts_for_selected_tier(self) -> None:
        request = {"model": "original", "messages": [{"role": "user", "content": "hello"}]}
        self.runtime.on_llm_request(request=request, session_id="s1", api_request_id="req-1")

        self.runtime.on_post_api_request(
            session_id="s1",
            api_request_id="req-1",
            usage={"input_tokens": 1_000_000, "output_tokens": 100_000},
            response_model="cheap",
        )

        summary = self.ledger.summary("s1")
        self.assertEqual(summary["session_requests"], 1)
        self.assertAlmostEqual(summary["session_spend_usd"], 1.2)

    def test_failed_attempt_does_not_consume_idempotency_key_for_later_success(self) -> None:
        request = {"model": "original", "messages": [{"role": "user", "content": "hello"}]}
        self.runtime.on_llm_request(request=request, session_id="s1", api_request_id="req-1")
        self.runtime.on_api_request_error(session_id="s1", api_request_id="req-1", model="cheap")

        self.runtime.on_post_api_request(
            session_id="s1",
            api_request_id="req-1",
            usage={"input_tokens": 1_000, "output_tokens": 100},
            response_model="cheap",
        )

        summary = self.ledger.summary("s1")
        self.assertEqual(summary["session_requests"], 1)
        self.assertGreater(summary["session_spend_usd"], 0)

    def test_hard_stop_blocks_next_turn_after_budget_is_exhausted(self) -> None:
        self.ledger.record_success("prior", "s1", "cheap", "economy", 1, 1, 1.01)

        result = self.runtime.on_pre_llm_call(session_id="s1")

        self.assertEqual(result["action"], "block")
        self.assertIn("budget", result["message"].lower())

    def test_status_tool_returns_json(self) -> None:
        raw = self.runtime.handle_tool({"action": "status"}, session_id="s1")
        payload = json.loads(raw)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["session_id"], "s1")
        self.assertIn("routing", payload)

    def test_tool_cannot_read_an_arbitrary_session(self) -> None:
        self.ledger.record_success("other", "s2", "cheap", "economy", 1, 1, 0.25)

        raw = self.runtime.handle_tool(
            {"action": "status", "session_id": "s2"}, session_id="s1",
        )
        payload = json.loads(raw)

        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["session_spend_usd"], 0)

    def test_slash_command_uses_active_session_environment(self) -> None:
        self.ledger.record_success("active", "live-session", "cheap", "economy", 1, 1, 0.25)

        with mock.patch.object(self.runtime, "_active_session_id", return_value="live-session"):
            payload = json.loads(self.runtime.handle_command("status"))

        self.assertEqual(payload["session_id"], "live-session")
        self.assertAlmostEqual(payload["session_spend_usd"], 0.25)

    def test_slash_command_does_not_expose_global_daily_reset(self) -> None:
        result = self.runtime.handle_command("reset-day")

        self.assertIn("unknown", result.lower())

    def test_missing_provider_usage_falls_back_to_projected_tokens(self) -> None:
        request = {"model": "original", "messages": [{"role": "user", "content": "hello"}]}
        self.runtime.on_llm_request(request=request, session_id="s1", api_request_id="req-missing")

        self.runtime.on_post_api_request(
            session_id="s1", api_request_id="req-missing", usage=None, response_model="cheap",
        )

        summary = self.ledger.summary("s1")
        self.assertGreater(summary["input_tokens"], 0)
        self.assertGreater(summary["output_tokens"], 0)
        self.assertGreater(summary["session_spend_usd"], 0)

    def test_execution_reservation_blocks_concurrent_overspend(self) -> None:
        runtime = AdaptiveBudgetRuntime(
            tiers=(ModelTier("economy", "costly", 0.0, 600_000.0, 1),),
            policy=BudgetPolicy(
                session_budget_usd=1.0,
                daily_budget_usd=1.0,
                reserve_ratio=0,
                cost_safety_multiplier=1,
                expected_output_tokens=1,
                hard_stop=True,
            ),
            ledger=self.ledger,
        )
        request = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
        first = runtime.on_llm_request(request=request, session_id="s1", api_request_id="r1")
        second = runtime.on_llm_request(request=request, session_id="s1", api_request_id="r2")

        self.assertEqual(runtime.on_llm_execution(
            request=first["request"], next_call=lambda routed: routed["model"],
            session_id="s1", api_request_id="r1",
        ), "costly")
        with self.assertRaises(BudgetExceededError):
            runtime.on_llm_execution(
                request=second["request"], next_call=lambda routed: routed["model"],
                session_id="s1", api_request_id="r2",
            )

        summary = self.ledger.summary("s1")
        self.assertAlmostEqual(summary["session_reserved_usd"], 0.60)

    def test_execution_error_releases_reservation(self) -> None:
        request = {"messages": [{"role": "user", "content": "hello"}]}
        routed = self.runtime.on_llm_request(
            request=request, session_id="s1", api_request_id="req-error",
        )

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            self.runtime.on_llm_execution(
                request=routed["request"],
                next_call=lambda _: (_ for _ in ()).throw(RuntimeError("provider failed")),
                session_id="s1",
                api_request_id="req-error",
            )

        self.assertEqual(self.ledger.summary("s1")["session_reserved_usd"], 0)

    def test_runtime_preserves_disabled_tier_threshold_semantics(self) -> None:
        runtime = AdaptiveBudgetRuntime(
            tiers=(
                ModelTier("economy", "cheap", 0.1, 0.2, 1),
                ModelTier("balanced", "", 0.2, 0.4, 2),
                ModelTier("premium", "premium", 1.0, 2.0, 3),
            ),
            policy=BudgetPolicy(
                session_budget_usd=10,
                daily_budget_usd=10,
                reserve_ratio=0,
                cost_safety_multiplier=1,
                expected_output_tokens=100,
                balanced_threshold=35,
                premium_threshold=65,
            ),
            ledger=self.ledger,
        )
        prompt = (
            "Analyze and compare this distributed architecture, implement a migration plan, "
            "test security, and verify benchmarks. " * 2
        )

        result = runtime.on_llm_request(
            request={"messages": [{"role": "user", "content": prompt}]},
            session_id="s1",
            api_request_id="semantic-rank",
        )

        self.assertEqual(result["request"]["model"], "cheap")


class FakeState:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


class FakeContext:
    def __init__(self, data_dir: Path) -> None:
        self.state = FakeState(data_dir)
        self.middleware = []
        self.hooks = []
        self.tools = []
        self.commands = []
        self.unload = []

    def get_config(self, key, default=None):
        return default

    def register_middleware(self, kind, callback):
        self.middleware.append((kind, callback))

    def register_hook(self, name, callback):
        self.hooks.append((name, callback))

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands.append((name, handler, description, args_hint))

    def on_unload(self, callback):
        self.unload.append(callback)


class RegistrationTests(unittest.TestCase):
    def test_register_wires_structural_plugin_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ctx = FakeContext(Path(tempdir))

            register(ctx)

            self.assertEqual({name for name, _ in ctx.middleware}, {"llm_request", "llm_execution"})
            self.assertTrue({"pre_llm_call", "post_api_request", "api_request_error"}.issubset(
                {name for name, _ in ctx.hooks}
            ))
            self.assertEqual([tool["name"] for tool in ctx.tools], ["adaptive_budget"])
            self.assertEqual([name for name, *_ in ctx.commands], ["pirulito", "budget-router"])
            self.assertEqual(len(ctx.unload), 1)
            ctx.unload[0]()

    def test_register_supports_validator_context_without_state_facade(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            ctx = FakeContext(Path(tempdir))
            del ctx.state

            register(ctx)

            self.assertEqual([tool["name"] for tool in ctx.tools], ["adaptive_budget"])
            ctx.unload[0]()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from adaptive_budget_router.ledger import BudgetLedger


class BudgetLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.ledger = BudgetLedger(Path(self.tempdir.name) / "ledger.sqlite3")
        self.now = datetime(2026, 11, 11, 15, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.ledger.close()
        self.tempdir.cleanup()

    def test_record_success_is_idempotent_by_request_id(self) -> None:
        first = self.ledger.record_success(
            request_id="req-1",
            session_id="session-a",
            model="cheap",
            tier="economy",
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.001,
            occurred_at=self.now,
        )
        second = self.ledger.record_success(
            request_id="req-1",
            session_id="session-a",
            model="cheap",
            tier="economy",
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.001,
            occurred_at=self.now,
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertAlmostEqual(self.ledger.session_spend("session-a"), 0.001)

    def test_summary_separates_session_and_utc_day(self) -> None:
        self.ledger.record_success("r1", "s1", "cheap", "economy", 100, 10, 0.01, self.now)
        self.ledger.record_success("r2", "s2", "premium", "premium", 100, 10, 0.09, self.now)

        summary = self.ledger.summary("s1", now=self.now)

        self.assertAlmostEqual(summary["session_spend_usd"], 0.01)
        self.assertAlmostEqual(summary["daily_spend_usd"], 0.10)
        self.assertEqual(summary["session_requests"], 1)
        self.assertEqual(summary["daily_requests"], 2)
        self.assertEqual(summary["tiers"]["economy"]["requests"], 1)

    def test_reset_session_removes_only_target_session(self) -> None:
        self.ledger.record_success("r1", "s1", "cheap", "economy", 100, 10, 0.01, self.now)
        self.ledger.record_success("r2", "s2", "cheap", "economy", 100, 10, 0.02, self.now)

        deleted = self.ledger.reset_session("s1")

        self.assertEqual(deleted, 1)
        self.assertEqual(self.ledger.session_spend("s1"), 0)
        self.assertAlmostEqual(self.ledger.session_spend("s2"), 0.02)

    def test_reservations_are_atomic_across_connections(self) -> None:
        second = BudgetLedger(Path(self.tempdir.name) / "ledger.sqlite3")
        barrier = threading.Barrier(2)
        results: list[bool] = []

        def reserve(ledger: BudgetLedger, request_id: str) -> None:
            barrier.wait()
            results.append(ledger.try_reserve(
                request_id=request_id,
                session_id="s1",
                projected_cost_usd=0.60,
                session_budget_usd=1.00,
                daily_budget_usd=1.00,
                reserve_ratio=0,
                occurred_at=self.now,
            ))

        threads = [
            threading.Thread(target=reserve, args=(self.ledger, "r1")),
            threading.Thread(target=reserve, args=(second, "r2")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        second.close()

        self.assertEqual(sorted(results), [False, True])
        summary = self.ledger.summary("s1", now=self.now)
        self.assertAlmostEqual(summary["session_reserved_usd"], 0.60)
        self.assertAlmostEqual(summary["daily_reserved_usd"], 0.60)


    def test_success_settlement_replaces_reservation_with_actual_cost(self) -> None:
        reserved = self.ledger.try_reserve(
            "r1", "s1", 0.60, 1.00, 1.00, reserve_ratio=0, occurred_at=self.now,
        )

        recorded = self.ledger.record_success(
            "r1", "s1", "cheap", "economy", 100, 10, 0.25, self.now,
        )

        self.assertTrue(reserved)
        self.assertTrue(recorded)
        summary = self.ledger.summary("s1", now=self.now)
        self.assertAlmostEqual(summary["session_spend_usd"], 0.25)
        self.assertEqual(summary["session_reserved_usd"], 0)


if __name__ == "__main__":
    unittest.main()

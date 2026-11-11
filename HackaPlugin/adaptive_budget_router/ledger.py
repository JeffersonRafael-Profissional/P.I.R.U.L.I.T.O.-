from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetLedger:
    """Small SQLite ledger with idempotent request accounting."""

    def __init__(self, path: str | Path) -> None:
        raw_path = str(path)
        self.path = None if raw_path == ":memory:" else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            ":memory:" if self.path is None else str(self.path), check_same_thread=False, timeout=10
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    occurred_at REAL NOT NULL,
                    utc_day TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_budget_session ON requests(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_budget_day ON requests(utc_day)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    projected_cost_usd REAL NOT NULL,
                    occurred_at REAL NOT NULL,
                    utc_day TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reservation_session ON reservations(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reservation_day ON reservations(utc_day)"
            )

    @staticmethod
    def _moment(value: datetime | float | int | None = None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    def record_success(
        self,
        request_id: str,
        session_id: str,
        model: str,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        occurred_at: datetime | float | int | None = None,
    ) -> bool:
        moment = self._moment(occurred_at)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO requests (
                        request_id, session_id, model, tier, input_tokens, output_tokens,
                        cost_usd, occurred_at, utc_day
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(request_id), str(session_id or "default"), str(model or "unknown"), str(tier),
                        max(0, int(input_tokens or 0)), max(0, int(output_tokens or 0)),
                        max(0.0, float(cost_usd or 0.0)), moment.timestamp(), moment.date().isoformat(),
                    ),
                )
                self._conn.execute("DELETE FROM reservations WHERE request_id = ?", (str(request_id),))
                self._conn.commit()
                return cursor.rowcount == 1
            except Exception:
                self._conn.rollback()
                raise

    def try_reserve(
        self,
        request_id: str,
        session_id: str,
        projected_cost_usd: float,
        session_budget_usd: float,
        daily_budget_usd: float,
        *,
        reserve_ratio: float = 0.0,
        occurred_at: datetime | float | int | None = None,
    ) -> bool:
        """Atomically reserve projected cost across threads and Hermes processes."""
        moment = self._moment(occurred_at)
        request_key = str(request_id)
        sid = str(session_id or "default")
        day = moment.date().isoformat()
        projected = max(0.0, float(projected_cost_usd or 0.0))
        reserve = min(0.95, max(0.0, float(reserve_ratio or 0.0)))

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._conn.execute(
                    "SELECT 1 FROM reservations WHERE request_id = ?", (request_key,),
                ).fetchone() is not None:
                    self._conn.commit()
                    return True
                if self._conn.execute(
                    "SELECT 1 FROM requests WHERE request_id = ?", (request_key,),
                ).fetchone() is not None:
                    self._conn.commit()
                    return False

                session_row = self._conn.execute(
                    """
                    SELECT
                        (SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE session_id = ?) +
                        (SELECT COALESCE(SUM(projected_cost_usd), 0) FROM reservations WHERE session_id = ?)
                        AS total
                    """,
                    (sid, sid),
                ).fetchone()
                daily_row = self._conn.execute(
                    """
                    SELECT
                        (SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE utc_day = ?) +
                        (SELECT COALESCE(SUM(projected_cost_usd), 0) FROM reservations WHERE utc_day = ?)
                        AS total
                    """,
                    (day, day),
                ).fetchone()
                limits = (
                    (max(0.0, float(session_budget_usd or 0.0)), float(session_row["total"] or 0.0)),
                    (max(0.0, float(daily_budget_usd or 0.0)), float(daily_row["total"] or 0.0)),
                )
                fits = all(
                    limit <= 0 or projected <= max(0.0, limit - spent) * (1.0 - reserve) + 1e-12
                    for limit, spent in limits
                )
                if not fits:
                    self._conn.rollback()
                    return False

                self._conn.execute(
                    """
                    INSERT INTO reservations (
                        request_id, session_id, projected_cost_usd, occurred_at, utc_day
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (request_key, sid, projected, moment.timestamp(), day),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def release_reservation(self, request_id: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM reservations WHERE request_id = ?", (str(request_id),)
            )
            return cursor.rowcount == 1

    def record_error(
        self, request_id: str, session_id: str, model: str, tier: str,
        occurred_at: datetime | float | int | None = None,
    ) -> bool:
        return self.record_success(request_id, session_id, model, tier, 0, 0, 0.0, occurred_at)

    def session_spend(self, session_id: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM requests WHERE session_id = ?",
                (str(session_id or "default"),),
            ).fetchone()
        return float(row["total"] or 0.0)

    def daily_spend(self, now: datetime | float | int | None = None) -> float:
        day = self._moment(now).date().isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM requests WHERE utc_day = ?",
                (day,),
            ).fetchone()
        return float(row["total"] or 0.0)

    def summary(self, session_id: str, now: datetime | float | int | None = None) -> dict[str, Any]:
        sid = str(session_id or "default")
        day = self._moment(now).date().isoformat()
        with self._lock:
            session = self._conn.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0) AS spend, COUNT(*) AS requests,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens
                FROM requests WHERE session_id = ?
                """,
                (sid,),
            ).fetchone()
            daily = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS spend, COUNT(*) AS requests FROM requests WHERE utc_day = ?",
                (day,),
            ).fetchone()
            session_reserved = self._conn.execute(
                "SELECT COALESCE(SUM(projected_cost_usd), 0) AS total FROM reservations WHERE session_id = ?",
                (sid,),
            ).fetchone()
            daily_reserved = self._conn.execute(
                "SELECT COALESCE(SUM(projected_cost_usd), 0) AS total FROM reservations WHERE utc_day = ?",
                (day,),
            ).fetchone()
            tier_rows = self._conn.execute(
                """
                SELECT tier, COUNT(*) AS requests, COALESCE(SUM(cost_usd), 0) AS spend
                FROM requests WHERE session_id = ? GROUP BY tier ORDER BY tier
                """,
                (sid,),
            ).fetchall()
        return {
            "session_id": sid,
            "utc_day": day,
            "session_spend_usd": float(session["spend"] or 0.0),
            "daily_spend_usd": float(daily["spend"] or 0.0),
            "session_reserved_usd": float(session_reserved["total"] or 0.0),
            "daily_reserved_usd": float(daily_reserved["total"] or 0.0),
            "session_requests": int(session["requests"] or 0),
            "daily_requests": int(daily["requests"] or 0),
            "input_tokens": int(session["input_tokens"] or 0),
            "output_tokens": int(session["output_tokens"] or 0),
            "tiers": {
                row["tier"]: {"requests": int(row["requests"]), "spend_usd": float(row["spend"] or 0.0)}
                for row in tier_rows
            },
        }

    def reset_session(self, session_id: str) -> int:
        with self._lock, self._conn:
            sid = str(session_id or "default")
            cursor = self._conn.execute(
                "DELETE FROM requests WHERE session_id = ?", (sid,)
            )
            return max(0, cursor.rowcount)

    def reset_day(self, now: datetime | float | int | None = None) -> int:
        day = self._moment(now).date().isoformat()
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM requests WHERE utc_day = ?", (day,))
            self._conn.execute("DELETE FROM reservations WHERE utc_day = ?", (day,))
            return max(0, cursor.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

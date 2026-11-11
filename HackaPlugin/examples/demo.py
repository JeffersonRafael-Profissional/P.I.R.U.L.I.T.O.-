from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_budget_router.core import BudgetPolicy, ModelTier, choose_route


TIERS = (
    ModelTier("economy", "demo/economy", 0.10, 0.40, 1),
    ModelTier("balanced", "demo/balanced", 0.50, 2.00, 2),
    ModelTier("premium", "demo/premium", 2.00, 12.00, 3),
)
POLICY = BudgetPolicy(
    session_budget_usd=0.10,
    daily_budget_usd=1.00,
    reserve_ratio=0.10,
    cost_safety_multiplier=4.0,
    expected_output_tokens=800,
    balanced_threshold=35,
    premium_threshold=65,
)
SCENARIOS = (
    ("simple", "Translate this sentence to Portuguese.", 0.00),
    (
        "complex",
        (
            "Analyze, design, implement, test, benchmark, and verify a distributed event-processing "
            "architecture with failure recovery, security controls, and a migration plan. "
        ) * 8,
        0.00,
    ),
    (
        "budget-pressure",
        (
            "Analyze, design, implement, test, benchmark, and verify a distributed consensus system "
            "with formal safety reasoning and a production migration plan. "
        ) * 8,
        0.095,
    ),
)


def main() -> None:
    rows = []
    for name, prompt, spent in SCENARIOS:
        decision = choose_route(
            {"messages": [{"role": "user", "content": prompt}]},
            TIERS,
            POLICY,
            session_spent_usd=spent,
            daily_spent_usd=spent,
        )
        rows.append({"scenario": name, "session_spent_usd": spent, **asdict(decision)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Changelog

## 0.1.0

- Added deterministic request-complexity scoring.
- Added three configurable model tiers with provider guardrails.
- Added session and UTC-day budget caps, reserve ratio, conservative cost projections, and atomic cross-process reservations.
- Added SQLite usage accounting keyed idempotently by API request id.
- Added `llm_request` and `llm_execution` middleware, API observer hooks, hard-stop guard, `adaptive_budget` tool, and a session-scoped `/budget-router` command.
- Added tests, offline demo, Hermes validation metadata, and documentation.

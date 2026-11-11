<div align="center">

<img src="assets/mascot.jpg" alt="P.I.R.U.L.I.T.O. Mascot" width="300" style="border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);" />

# P.I.R.U.L.I.T.O.
### Plugin Inspection, Rate Uniformity & Local Integrity Token Optimizer

[![Hermes Agent](https://img.shields.io/badge/Hermes%20Agent-%3E%3D0.21.5-blue.svg)](https://github.com/NousResearch/hermes-agent)
[![Hackathon Submission](https://img.shields.io/badge/Nous%20Research-Hermes%20Hackathon-8A2BE2.svg)](https://nousresearch.com)
[![GitHub Repository](https://img.shields.io/badge/GitHub-JeffersonRafael--Profissional%2Fpirulito-181717.svg?logo=github)](https://github.com/JeffersonRafael-Profissional/pirulito)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-29%20Passed-success.svg)]()

<p><em>Meet the official mascot of P.I.R.U.L.I.T.O. — keeping your token spend cool and your model routing optimized under maximum agent workload!</em> 🎧⚡🍭</p>

---

</div>

**P.I.R.U.L.I.T.O.** is a native **Hermes Agent middleware plugin** that inspects task complexity and dynamically routes each main-loop LLM request to the most cost-effective configured model tier while enforcing atomic session and daily budget caps.

It is structural—not prompt-only: the plugin operates at the `llm_request` and `llm_execution` middleware boundaries, rewrites provider request kwargs, observes real token usage, and persists a profile-scoped SQLite ledger.

## Why it is useful

- **Adaptive Quality:** Simple tasks use an economy model; complex multi-step reasoning tasks scale up to balanced or premium tiers.
- **Rate Uniformity & Spend Guard:** Enforces strict per-session and daily budget caps with atomic SQLite reservations preventing concurrent overspend.
- **Local Integrity:** 100% standard-library Python; no external telemetry, no remote API dependencies, and no credential access required.
- **Auditable Decisions:** `/pirulito explain` and the `adaptive_budget` tool provide full visibility into scoring, token estimations, and routing justifications.

## Architecture

```text
User turn
   │
   ▼
pre_llm_call hook ── hard-stop check against SQLite ledger
   │
   ▼
llm_request middleware (P.I.R.U.L.I.T.O.)
   ├─ estimate prompt/tool-schema tokens
   ├─ score complexity (length, tools, reasoning terms, code, structure)
   ├─ choose desired quality tier
   ├─ project cost + reserve + safety multiplier
   └─ rewrite request["model"]
   │
   ▼
llm_execution middleware
   ├─ atomically reserve safety-adjusted projected cost
   ├─ re-route if another concurrent request changed the budget
   └─ block provider execution when no tier fits and hard_stop is enabled
   │
   ▼
Current Hermes provider executes the request
   │
   ▼
post_api_request / api_request_error hooks
   ├─ read actual input/output token usage
   ├─ estimate cost from configured prices
   └─ settle or release the reservation idempotently
```

## Installation

### From GitHub

```bash
hermes plugins install JeffersonRafael-Profissional/pirulito --enable
```

For official catalog admission, pin the full 40-character commit SHA in your catalog entry (`plugin-catalog/pirulito.yaml`).

### Local Development

Copy or symlink the repository into the active profile's plugin directory, then enable it:

```bash
mkdir -p "$HERMES_HOME/plugins/pirulito"
cp -R . "$HERMES_HOME/plugins/pirulito"
hermes plugins enable pirulito
```

Verify your installation:

```bash
hermes plugins validate . --json
hermes plugins doctor . --ci
```

## Configuration

Configure tiers and budgets via `hermes config set`:

```bash
# Provider guard
hermes config set plugins.entries.pirulito.settings.allowed_providers openrouter

# Budgets
hermes config set plugins.entries.pirulito.settings.session_budget_usd 0.50
hermes config set plugins.entries.pirulito.settings.daily_budget_usd 5.00
hermes config set plugins.entries.pirulito.settings.reserve_ratio 0.10
hermes config set plugins.entries.pirulito.settings.cost_safety_multiplier 4.0
hermes config set plugins.entries.pirulito.settings.hard_stop true

# Model tiers (example for OpenRouter)
hermes config set plugins.entries.pirulito.settings.economy_model openrouter/deepseek/deepseek-chat
hermes config set plugins.entries.pirulito.settings.economy_input_usd_per_million 0.14
hermes config set plugins.entries.pirulito.settings.economy_output_usd_per_million 0.28

hermes config set plugins.entries.pirulito.settings.balanced_model openrouter/anthropic/claude-3.5-haiku
hermes config set plugins.entries.pirulito.settings.balanced_input_usd_per_million 0.80
hermes config set plugins.entries.pirulito.settings.balanced_output_usd_per_million 4.00

hermes config set plugins.entries.pirulito.settings.premium_model openrouter/anthropic/claude-3.5-sonnet
hermes config set plugins.entries.pirulito.settings.premium_input_usd_per_million 3.00
hermes config set plugins.entries.pirulito.settings.premium_output_usd_per_million 15.00
```

## Usage

### Slash Command

```text
/pirulito status
/pirulito explain Design a distributed event-driven microservice architecture with Kafka
/pirulito reset-session
```
*(Alias `/budget-router` is also supported)*

### Agent Tool

The plugin registers the `adaptive_budget` tool:
- `status` — current session/day spend and tier totals.
- `explain` — simulate and explain a routing decision for a prompt.

## Local Demo

```bash
python examples/demo.py
```

The demo runs completely offline and deterministically without network calls.

## Verification

```bash
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).

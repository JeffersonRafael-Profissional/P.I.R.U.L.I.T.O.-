ADAPTIVE_BUDGET_SCHEMA = {
    "name": "adaptive_budget",
    "description": (
        "Inspect Adaptive Budget Router spend and routing, or simulate which configured tier "
        "would be selected for a prompt. This tool never resets budgets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "explain"],
                "description": "status returns current spend; explain simulates a routing decision.",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt to score when action=explain.",
            },
        },
        "required": ["action"],
    },
}

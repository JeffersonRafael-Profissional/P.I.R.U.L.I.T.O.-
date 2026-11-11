"""Repository-root entry point used by the Hermes native plugin loader."""

from .adaptive_budget_router import register

__all__ = ["register"]

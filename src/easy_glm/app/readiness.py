"""Readiness helpers used by the Streamlit workbench entry point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def split_ready(state_module: Any) -> bool:
    """Return whether split-gated pages should be unlocked.

    Prefer ``state.split_ready()`` when available. Fall back to the stable
    ``state.status()['split']`` contract for compatibility with older state
    modules where ``split_ready`` is absent.
    """
    helper = getattr(state_module, "split_ready", None)
    if callable(helper):
        return bool(helper())

    status = getattr(state_module, "status", None)
    if not callable(status):
        return False
    current = status()
    if not isinstance(current, Mapping):
        return False
    return bool(current.get("split", False))

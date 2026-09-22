"""Keep saved screening recipes independent of changing interactive defaults."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def resolved_selection_options(
    options: dict[str, Any],
    result: dict[str, Any],
    *,
    legacy: bool = False,
) -> dict[str, Any]:
    """Record the resolved link and scoring sample, preserving old replays.

    Version-one recipes written before sampling was introduced used every training
    row for importance and a log link for Gaussian unless explicitly overridden.
    Saved result metadata takes precedence over defaults, never over an explicit
    option. Neither argument is mutated.
    """
    resolved = deepcopy(options)
    if resolved.get("link") is None:
        link = result.get("link")
        if not isinstance(link, str) or not link:
            family = str(resolved.get("family", "poisson")).lower()
            link = (
                "logit"
                if family == "binomial"
                else (
                    "identity"
                    if family in ("gaussian", "normal") and not legacy
                    else "log"
                )
            )
        resolved["link"] = link
    if "importance_sample_pct" not in resolved:
        resolved["importance_sample_pct"] = result.get(
            "importance_sample_pct", 100.0 if legacy else 30.0
        )
    return resolved

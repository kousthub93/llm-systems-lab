"""Compare a fresh run's summary to a stored baseline and flag regressions."""

from __future__ import annotations

from dataclasses import dataclass

# metric -> (better direction, how the tolerance is measured)
#   "abs": allowed change is an absolute difference (quality metrics in 0..1)
#   "rel": allowed change is a fraction of the baseline value (cost metrics)
GATED_METRICS: dict[str, tuple[str, str]] = {
    "EM": ("higher", "abs"),
    "F1": ("higher", "abs"),
    "retrieval_hit": ("higher", "abs"),
    "F1_if_hit": ("higher", "abs"),
    "cite_valid": ("higher", "abs"),
    "cite_correct": ("higher", "abs"),
    "prompt_tokens": ("lower", "rel"),
    "p50_gen_s": ("lower", "rel"),
}

DEFAULT_ABS_TOL = 0.05
DEFAULT_REL_TOL = 0.25
_EPS = 1e-9


@dataclass
class Finding:
    mode: str
    metric: str
    baseline: float | None
    current: float | None
    change: float | None  # absolute difference, or fraction of baseline for "rel" metrics
    allowed: float
    status: str  # "ok" | "regression" | "skipped" | "missing"


def compare(
    current: dict[str, dict],
    baseline: dict[str, dict],
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    overrides: dict[str, float] | None = None,
) -> list[Finding]:
    """Check every gated metric of every baseline mode against the current run.

    A metric regresses when it moves in the wrong direction by more than its
    tolerance. Improvements never fail. A baseline mode absent from the current
    run is reported as "missing".
    """
    overrides = overrides or {}
    findings: list[Finding] = []
    for mode, base_summary in baseline.items():
        if mode not in current:
            findings.append(Finding(mode, "*", None, None, None, 0.0, "missing"))
            continue
        cur_summary = current[mode]
        for metric, (direction, kind) in GATED_METRICS.items():
            if metric not in base_summary:
                continue
            allowed = overrides.get(metric, abs_tol if kind == "abs" else rel_tol)
            b, c = base_summary[metric], cur_summary.get(metric)
            if b is None or c is None:
                findings.append(Finding(mode, metric, b, c, None, allowed, "skipped"))
                continue
            if kind == "abs":
                change = c - b
            elif b != 0:
                change = (c - b) / b
            else:
                change = 0.0 if c == 0 else float("inf")
            worse = -change if direction == "higher" else change
            status = "regression" if worse > allowed + _EPS else "ok"
            findings.append(Finding(mode, metric, b, c, change, allowed, status))
    return findings


def has_regression(findings: list[Finding]) -> bool:
    return any(f.status in ("regression", "missing") for f in findings)

"""Command line: summarize a run, write a baseline, or check a run against one.

Exit codes: 0 all good, 1 regression found, 2 bad input (missing or malformed file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .regression import DEFAULT_ABS_TOL, DEFAULT_REL_TOL, compare, has_regression
from .report import load_records, summarize_by_mode

SUMMARY_COLUMNS = ["n", "EM", "F1", "retrieval_hit", "cite_valid", "cite_correct", "idk_rate", "p50_gen_s"]


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _print_summary(summaries: dict[str, dict]) -> None:
    width = max([len("mode")] + [len(m) for m in summaries])
    print("mode".ljust(width), *[c.rjust(13) for c in SUMMARY_COLUMNS])
    for mode, row in summaries.items():
        print(mode.ljust(width), *[_fmt(row.get(c)).rjust(13) for c in SUMMARY_COLUMNS])


def _print_findings(findings) -> None:
    width = max([len("mode")] + [len(f.mode) for f in findings])
    print("mode".ljust(width), "metric".ljust(15), *[h.rjust(9) for h in ("baseline", "current", "change", "allowed")], " status")
    for f in findings:
        print(
            f.mode.ljust(width),
            f.metric.ljust(15),
            *[_fmt(v).rjust(9) for v in (f.baseline, f.current, f.change, f.allowed)],
            "",
            f.status,
        )


def _cmd_summarize(args) -> int:
    summaries = summarize_by_mode(load_records(args.records))
    _print_summary(summaries)
    return 0


def _cmd_baseline(args) -> int:
    summaries = summarize_by_mode(load_records(args.records))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"description": f"Baseline written from {Path(args.records).name}", "modes": summaries}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote baseline for {len(summaries)} modes to {out}")
    return 0


def _cmd_check(args) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = summarize_by_mode(load_records(args.records))
    findings = compare(
        current,
        baseline["modes"],
        abs_tol=args.tolerance,
        rel_tol=args.rel_tolerance,
        overrides=baseline.get("tolerances"),
    )
    _print_findings(findings)
    bad = [f for f in findings if f.status in ("regression", "missing")]
    if has_regression(findings):
        print(f"\nFAILED: {len(bad)} regression(s)")
        return 1
    print("\nOK: no regressions")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragevals", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("summarize", help="print per-mode metrics for a records file")
    p.add_argument("records")
    p.set_defaults(func=_cmd_summarize)

    p = sub.add_parser("baseline", help="write a baseline JSON from a records file")
    p.add_argument("records")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_baseline)

    p = sub.add_parser("check", help="compare a records file to a baseline; exit 1 on regression")
    p.add_argument("records")
    p.add_argument("--baseline", required=True)
    p.add_argument("--tolerance", type=float, default=DEFAULT_ABS_TOL, help="allowed absolute drop for quality metrics")
    p.add_argument("--rel-tolerance", type=float, default=DEFAULT_REL_TOL, help="allowed relative rise for cost metrics")
    p.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as exc:  # missing file, bad JSON, bad baseline shape
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""Load per-question records and summarize them per mode."""

from __future__ import annotations

import json
from typing import Iterable

CLOSED_BOOK_MODES = {"closed-book"}


def load_records(path) -> list[dict]:
    """Read a JSONL file, one record per line. Blank lines are skipped."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
    return records


def _mean(values: Iterable) -> float | None:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def summarize(recs: list[dict], closed_book: bool = False) -> dict:
    """Aggregate scored records into one row of metrics.

    Retrieval and citation metrics are left out for closed-book runs, where no
    chunks are given to the model. A metric whose denominator is empty is None.
    """
    if not recs:
        raise ValueError("no records to summarize")
    n = len(recs)
    out: dict = {
        "n": n,
        "EM": _mean(r["em"] for r in recs),
        "F1": _mean(r["f1"] for r in recs),
        "idk_rate": _mean(r["idk"] for r in recs),
    }
    if not closed_book:
        hit = [r for r in recs if r["hit"]]
        miss = [r for r in recs if not r["hit"]]
        out["retrieval_hit"] = len(hit) / n
        out["F1_if_hit"] = _mean(r["f1"] for r in hit)
        out["F1_if_miss"] = _mean(r["f1"] for r in miss)
        out["idk_if_miss"] = _mean(r["idk"] for r in miss)
        answered = [r for r in recs if not r["idk"]]
        out["cite_present"] = _mean(r["has_cite"] for r in answered)
        with_cite = [r for r in answered if r["has_cite"]]
        out["cite_valid"] = _mean(r["cite_valid"] for r in with_cite)
        out["cite_correct"] = _mean(r["cite_correct"] for r in with_cite)
    out["prompt_tokens"] = _mean(r["prompt_tokens"] for r in recs)
    times = sorted(r["gen_s"] for r in recs)
    out["p50_gen_s"] = times[len(times) // 2]
    return out


def summarize_by_mode(records: list[dict]) -> dict[str, dict]:
    """Group records by their 'mode' field (first-seen order) and summarize each group."""
    groups: dict[str, list[dict]] = {}
    for i, rec in enumerate(records):
        if "mode" not in rec:
            raise ValueError(f"record {i} has no 'mode' field")
        groups.setdefault(rec["mode"], []).append(rec)
    return {
        mode: summarize(recs, closed_book=(mode in CLOSED_BOOK_MODES))
        for mode, recs in groups.items()
    }

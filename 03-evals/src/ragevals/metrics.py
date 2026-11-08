"""Answer and retrieval metrics. Pure Python, no dependencies."""

from __future__ import annotations

import collections
import re
import string
from typing import Callable, Iterable, Sequence

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles, collapse spaces."""
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-overlap F1 between a predicted and a gold answer."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    overlap = sum((collections.Counter(p) & collections.Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(metric: Callable[[str, str], float], pred: str, golds: Iterable[str]) -> float:
    """Score against every acceptable gold answer and keep the best."""
    return max((metric(pred, g) for g in golds), default=0.0)


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """1.0 if any gold id is among the first k ranked ids, else 0.0."""
    gold = set(gold_ids)
    return 1.0 if any(cid in gold for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    """1 / rank of the first gold id, or 0.0 if none is retrieved."""
    gold = set(gold_ids)
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0

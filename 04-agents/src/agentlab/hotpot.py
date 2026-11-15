"""HotpotQA data prep and metrics.

HotpotQA questions need two Wikipedia paragraphs ("supporting" paragraphs) to answer.
In the distractor release each question ships with 10 paragraphs: the 2 supporting ones plus
8 distractors. We pool the paragraphs of many questions into one searchable corpus, so the
agent has to find the right paragraphs among thousands, not among its own 10.

Rows come in two shapes and both are accepted:
  raw JSON (CMU release):  context = [[title, [sentences]], ...]  supporting_facts = [[title, sent_id], ...]
  Hugging Face columns:    context = {"title": [...], "sentences": [[...]]}  supporting_facts = {"title": [...], ...}
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Iterable

from .retrieval import Passage

_SPECIAL = {"yes", "no", "noanswer"}
_PUNCT = set(string.punctuation)


# ----------------------------------------------------------------------------- metrics

def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse spaces (same normalization as SQuAD/HotpotQA)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-overlap F1. yes/no/noanswer must match exactly, as in the official HotpotQA scorer."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in _SPECIAL or g in _SPECIAL:
        return float(p == g)
    p_tokens, g_tokens = p.split(), g.split()
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p_tokens), overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


# ----------------------------------------------------------------------------- data

def normalize_row(row: dict) -> dict:
    ctx = row["context"]
    if isinstance(ctx, dict):          # columnar; values may be numpy arrays when read from parquet
        pairs = [(str(t), [str(x) for x in s]) for t, s in zip(ctx["title"], ctx["sentences"])]
    else:
        pairs = [(str(t), [str(x) for x in s]) for t, s in ctx]
    sf = row.get("supporting_facts")
    if sf is None:
        sf = []
    sf_titles = [str(t) for t in sf["title"]] if isinstance(sf, dict) else [str(t) for t, _ in sf]
    return {
        "id": str(row.get("_id", row.get("id"))),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "type": str(row.get("type", "")),
        "level": str(row.get("level", "")),
        "gold_titles": list(dict.fromkeys(sf_titles)),
        "paragraphs": [(title, " ".join(" ".join(sents).split())) for title, sents in pairs],
    }


def build_hotpot(rows: Iterable[dict], n_eval: int, n_corpus: int, seed: int = 0):
    """Pick n_corpus questions at random, evaluate on the first n_eval of them.

    Returns (examples, passages, stats). The corpus is every paragraph of the n_corpus questions,
    de-duplicated by title, so each evaluation question's supporting paragraphs are in the corpus.
    Rows that are unusable (no answer, or a supporting title missing from the context) are dropped.
    """
    if n_corpus < n_eval:
        raise ValueError("n_corpus must be at least n_eval, so the evaluation questions' paragraphs are indexed")
    usable = []
    for raw in rows:
        r = normalize_row(raw)
        titles = {t for t, _ in r["paragraphs"]}
        if r["answer"].strip() and r["gold_titles"] and set(r["gold_titles"]) <= titles:
            usable.append(r)
    if len(usable) < n_corpus:
        raise ValueError(f"only {len(usable)} usable rows, need {n_corpus}")
    random.Random(seed).shuffle(usable)
    pool = usable[:n_corpus]
    examples = pool[:n_eval]

    by_title: dict[str, str] = {}
    conflicts = 0
    for r in pool:
        for title, text in r["paragraphs"]:
            if title in by_title:
                conflicts += by_title[title] != text
            else:
                by_title[title] = text
    passages = [Passage(id=title, title=title, text=text) for title, text in by_title.items()]
    stats = {"n_eval": len(examples), "n_corpus_questions": len(pool), "n_passages": len(passages),
             "title_conflicts": conflicts}
    return examples, passages, stats


def title_recall(retriever, examples: list[dict], k: int) -> dict:
    """Retrieval ceiling: search with the raw question and check which supporting titles come back.

    both  = fraction of questions where ALL supporting paragraphs are in the top k
    any   = fraction where at least one is
    A single-shot RAG pipeline cannot do better than `both`; this is what the agent has to beat.
    """
    both = any_ = 0
    for ex in examples:
        got = {p.title for p in retriever.search(ex["question"], k)}
        gold = set(ex["gold_titles"])
        both += gold <= got
        any_ += bool(gold & got)
    n = len(examples)
    return {"k": k, "both": both / n, "any": any_ / n}

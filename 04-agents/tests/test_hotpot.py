import pytest

from agentlab.hotpot import (build_hotpot, exact_match, f1_score, normalize_answer, normalize_row,
                             title_recall)
from agentlab.retrieval import Passage


def raw_row(i, answer="Paris", gold=("A", "B"), extra=3, qtype="bridge"):
    """CMU-style row: lists of pairs."""
    titles = list(gold) + [f"D{i}_{j}" for j in range(extra)]
    return {"_id": f"q{i}", "question": f"question {i}?", "answer": answer, "type": qtype, "level": "hard",
            "supporting_facts": [[g, 0] for g in gold] + [[gold[0], 3]],
            "context": [[t, [f"First sentence of {t}.", " Second sentence."]] for t in titles]}


def hf_row(i, **kw):
    """Hugging Face-style row: dicts of lists."""
    r = raw_row(i, **kw)
    return {"id": r["_id"], "question": r["question"], "answer": r["answer"], "type": r["type"], "level": r["level"],
            "supporting_facts": {"title": [t for t, _ in r["supporting_facts"]],
                                 "sent_id": [s for _, s in r["supporting_facts"]]},
            "context": {"title": [t for t, _ in r["context"]], "sentences": [s for _, s in r["context"]]}}


# ------------------------------------------------------------------ metrics

def test_normalize_answer():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("  An   apple, a day ") == "apple day"


def test_exact_match_ignores_case_articles_and_punctuation():
    assert exact_match("the Boris Yeltsin.", "Boris Yeltsin") == 1.0
    assert exact_match("President Boris Yeltsin", "Boris Yeltsin") == 0.0


def test_f1_partial_credit_matches_the_worked_example():
    assert f1_score("President Boris Yeltsin", "Boris Yeltsin") == pytest.approx(0.8)
    assert f1_score("Moscow", "Boris Yeltsin") == 0.0
    assert f1_score("", "Boris Yeltsin") == 0.0


def test_yes_no_must_match_exactly():
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("no", "yes") == 0.0
    assert f1_score("yes it is", "yes") == 0.0  # official scorer: special answers get no partial credit
    assert f1_score("yes", "yes it is") == 0.0


# ------------------------------------------------------------------ rows

def test_normalize_row_handles_both_shapes_identically():
    a, b = normalize_row(raw_row(1)), normalize_row(hf_row(1))
    assert a == b
    assert a["id"] == "q1" and a["gold_titles"] == ["A", "B"]  # duplicate supporting title collapsed
    assert a["paragraphs"][0] == ("A", "First sentence of A. Second sentence.")  # whitespace tidied


def test_build_hotpot_pools_and_dedupes_paragraphs():
    rows = [raw_row(i, gold=("Shared", f"G{i}")) for i in range(10)]
    examples, passages, stats = build_hotpot(rows, n_eval=3, n_corpus=8, seed=1)
    assert len(examples) == 3 and stats["n_corpus_questions"] == 8
    titles = [p.title for p in passages]
    assert len(titles) == len(set(titles)) == stats["n_passages"]
    assert titles.count("Shared") == 1                        # appears in every row, indexed once
    for ex in examples:                                       # supporting paragraphs are always searchable
        assert set(ex["gold_titles"]) <= set(titles)
    assert all(p.id == p.title for p in passages)


def test_build_hotpot_is_deterministic_per_seed():
    rows = [raw_row(i) for i in range(30)]
    ids = lambda seed: [e["id"] for e in build_hotpot(rows, 5, 20, seed=seed)[0]]
    assert ids(0) == ids(0) and ids(0) != ids(1)


def test_build_hotpot_counts_conflicting_paragraphs():
    r1, r2 = raw_row(1), raw_row(2)
    r2["context"][0] = ["A", ["A different version of the paragraph."]]
    _, _, stats = build_hotpot([r1, r2], n_eval=1, n_corpus=2)
    assert stats["title_conflicts"] == 1


def test_build_hotpot_drops_unusable_rows():
    good = raw_row(1)
    no_answer = raw_row(2, answer="  ")
    missing_gold = raw_row(3)
    missing_gold["supporting_facts"] = [["Not In Context", 0]]
    _, _, stats = build_hotpot([good, no_answer, missing_gold], n_eval=1, n_corpus=1)
    assert stats["n_corpus_questions"] == 1
    with pytest.raises(ValueError, match="only 1 usable"):
        build_hotpot([good, no_answer, missing_gold], n_eval=1, n_corpus=2)


def test_build_hotpot_requires_corpus_at_least_eval_size():
    with pytest.raises(ValueError, match="n_corpus"):
        build_hotpot([raw_row(1)], n_eval=2, n_corpus=1)


# ------------------------------------------------------------------ retrieval ceiling

class StubRetriever:
    def __init__(self, table):
        self.table = table

    def search(self, query, k):
        return [Passage(t, t, "") for t in self.table[query][:k]]


def test_title_recall_counts_both_and_any():
    exs = [{"question": "q1", "gold_titles": ["A", "B"]},   # both found
           {"question": "q2", "gold_titles": ["A", "B"]},   # one found
           {"question": "q3", "gold_titles": ["A", "B"]}]   # none found
    r = StubRetriever({"q1": ["B", "A", "X"], "q2": ["A", "X", "Y"], "q3": ["X", "Y", "Z"]})
    assert title_recall(r, exs, k=3) == {"k": 3, "both": pytest.approx(1 / 3), "any": pytest.approx(2 / 3)}
    assert title_recall(r, exs, k=1)["both"] == 0.0

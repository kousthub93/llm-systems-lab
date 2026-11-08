import pytest

from ragevals.metrics import (
    best_over_golds,
    exact_match,
    f1_score,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
)


def test_normalize_drops_case_punctuation_and_articles():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("  An   apple, a day ") == "apple day"


def test_exact_match_ignores_articles_and_case():
    assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0
    assert exact_match("Paris", "Lyon") == 0.0


def test_f1_partial_overlap():
    # prediction has 3 tokens, gold has 1, one shared -> precision 1/3, recall 1 -> F1 0.5
    assert f1_score("in Paris France", "Paris") == pytest.approx(0.5)


def test_f1_no_overlap_and_empty_prediction():
    assert f1_score("Berlin", "Paris") == 0.0
    assert f1_score("", "Paris") == 0.0


def test_best_over_golds_takes_the_best_match():
    assert best_over_golds(exact_match, "Paris", ["Lyon", "paris"]) == 1.0
    assert best_over_golds(exact_match, "Paris", []) == 0.0


@pytest.mark.parametrize(
    "ranked, gold, r1, r3, rr",
    [
        (["p3", "p1", "p7"], ["p1"], 0.0, 1.0, 0.5),   # gold at rank 2
        (["p5", "p2", "p9"], ["p5"], 1.0, 1.0, 1.0),   # gold at rank 1
        (["p1", "p2", "p3"], ["p9"], 0.0, 0.0, 0.0),   # gold not retrieved
        (["p4"], ["p4"], 1.0, 1.0, 1.0),               # single result
        (["x", "p2", "p9"], ["p2", "p9"], 0.0, 1.0, 0.5),  # several gold chunks: first one counts
    ],
)
def test_retrieval_metrics(ranked, gold, r1, r3, rr):
    assert recall_at_k(ranked, gold, 1) == r1
    assert recall_at_k(ranked, gold, 3) == r3
    assert reciprocal_rank(ranked, gold) == pytest.approx(rr)

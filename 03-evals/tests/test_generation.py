from ragevals.generation import parse_answer, score_answer


# The first three are real outputs from the Project 2 smoke test.
def test_parse_clean_answer_with_adjacent_citations():
    assert parse_answer("Boris Yeltsin [d31#98][d31#97]") == {
        "answer": "Boris Yeltsin", "cited": ["d31#98", "d31#97"], "idk": False,
    }


def test_parse_keeps_malformed_chunk_ids_as_citations():
    parsed = parse_answer("Sicilian Mafia [d125#36#3] [d125#3#3]")
    assert parsed["answer"] == "Sicilian Mafia"
    assert parsed["cited"] == ["d125#36#3", "d125#3#3"]  # counted, so scoring can mark them invalid


def test_parse_ignores_trailing_note_and_dedupes_citations():
    raw = "2011 [d140#92], [d140#91]\n\nNote: The answer is derived from the context chunks [d140#92] and [d"
    assert parse_answer(raw) == {"answer": "2011", "cited": ["d140#92", "d140#91"], "idk": False}


def test_parse_abstention_variants():
    assert parse_answer("I don't know.")["idk"]
    assert parse_answer("I don’t know")["idk"]
    assert parse_answer("I do not know [d1#1]")["idk"]
    assert parse_answer("India")["idk"] is False  # a word starting with "I" is not an abstention


def test_parse_edge_cases():
    assert parse_answer("") == {"answer": "", "cited": [], "idk": False}
    assert parse_answer("[d3#1] Paris")["answer"] == "Paris"
    assert parse_answer("Paris.\nSources: [d1#2]")["cited"] == ["d1#2"]
    assert parse_answer("Paris [Chunk d1#2] [1]")["cited"] == ["d1#2"]  # non-id tokens ignored


def test_score_correct_answer_and_citation():
    r = score_answer("Paris [d1#2]", ["Paris"], ["d1#2", "d4#0"], {"d1#2"})
    assert r["em"] == 1 and r["hit"] and r["cite_valid"] and r["cite_correct"]


def test_score_citation_of_chunk_that_was_not_provided_is_invalid():
    r = score_answer("Paris [d9#9]", ["Paris"], ["d1#2"], {"d1#2"})
    assert r["em"] == 1 and r["has_cite"] and not r["cite_valid"] and not r["cite_correct"]


def test_score_malformed_citation_counts_as_present_but_invalid():
    r = score_answer("Sicilian Mafia [d125#36#3]", ["the Sicilian Mafia"], ["d125#36"], {"d125#36"})
    assert r["em"] == 1 and r["has_cite"] and not r["cite_valid"] and not r["cite_correct"]


def test_score_abstention_when_retrieval_missed():
    r = score_answer("I don't know.", ["Paris"], ["d4#0"], {"d1#2"})
    assert r["idk"] and not r["hit"] and r["f1"] == 0 and r["answer"] == ""


def test_score_cites_valid_chunk_that_lacks_the_answer():
    r = score_answer("Paris [d4#0]", ["Paris"], ["d1#2", "d4#0"], {"d1#2"})
    assert r["cite_valid"] and not r["cite_correct"] and r["hit"]

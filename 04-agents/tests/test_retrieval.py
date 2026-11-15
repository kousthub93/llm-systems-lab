import numpy as np
import pytest

from agentlab import Agent, ToolRegistry
from agentlab.retrieval import HybridRetriever, Passage, make_search_tool, rrf_fuse, tokenize
from conftest import act

DIM = 64


def fake_embed(texts):
    """Hashed bag of words, L2-normalized. Crude, but deterministic and good enough to rank by overlap."""
    out = np.zeros((len(texts), DIM))
    for r, text in enumerate(texts):
        for tok in tokenize(text):
            out[r, sum(ord(c) for c in tok) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


def flat_embed(texts):
    """No dense signal at all: every text gets the same vector."""
    return np.ones((len(texts), DIM)) / np.sqrt(DIM)


PASSAGES = [
    Passage("p0", "Mikhail Gorbachev", "Mikhail Gorbachev was the last leader of the Soviet Union and resigned in 1991."),
    Passage("p1", "Boris Yeltsin", "Boris Yeltsin became the first president of Russia in 1991."),
    Passage("p2", "Eiffel Tower", "The Eiffel Tower is a wrought iron lattice tower in Paris, France."),
    Passage("p3", "Great Barrier Reef", "The Great Barrier Reef is the largest coral reef system, off Queensland, Australia."),
    Passage("p4", "Photosynthesis", "Photosynthesis converts light energy into chemical energy stored in glucose."),
    Passage("p5", "Mount Everest", "Mount Everest is Earth's highest mountain above sea level, in the Himalayas."),
]


def ids(passages):
    return [p.id for p in passages]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("Who was the President of Russia?") == ["president", "russia"]


def test_rrf_item_in_both_lists_beats_item_in_one():
    assert rrf_fuse([["a", "b"], ["c", "b"]], [1.0, 1.0], top_k=3) == ["b", "a", "c"]


def test_rrf_weights_and_top_k():
    assert rrf_fuse([["a"], ["b"]], [1.0, 3.0], top_k=1) == ["b"]


def test_search_ranks_the_matching_passage_first():
    r = HybridRetriever(PASSAGES, fake_embed)
    assert ids(r.search("first president of Russia", k=2))[0] == "p1"
    assert ids(r.search("Eiffel Tower in Paris", k=1)) == ["p2"]


def test_title_is_part_of_what_is_searched():
    # "Himalayas" is only in p5's text, "Everest" is in its title and text; a title-only word must still match
    r = HybridRetriever([Passage("a", "Zanzibar", "An island."), *PASSAGES[2:]], fake_embed)
    assert ids(r.search("Zanzibar", k=1)) == ["a"]


def test_sparse_signal_alone_still_surfaces_the_match():
    # Constant embeddings tie every dense score, so the dense list is just index order and RRF can
    # tie the exact top spot. The BM25 hit must still be in the results, ahead of the unrelated passages.
    r = HybridRetriever(PASSAGES, flat_embed)
    top2 = ids(r.search("Yeltsin", k=2))
    assert "p1" in top2 and not {"p3", "p4", "p5"} & set(top2)


def test_k_larger_than_corpus_returns_everything_once():
    r = HybridRetriever(PASSAGES, fake_embed)
    out = ids(r.search("Russia", k=50))
    assert sorted(out) == sorted(ids(PASSAGES))


def test_blank_query_and_bad_k_return_nothing():
    r = HybridRetriever(PASSAGES, fake_embed)
    assert r.search("   ") == [] and r.search("Russia", k=0) == []


def test_rejects_empty_corpus_and_duplicate_ids():
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([], fake_embed)
    with pytest.raises(ValueError, match="unique"):
        HybridRetriever([PASSAGES[0], PASSAGES[0]], fake_embed)


def test_reranker_reorders_its_pool():
    prefer_gorbachev = lambda pairs: [1.0 if "Gorbachev" in text else 0.0 for _, text in pairs]
    r = HybridRetriever(PASSAGES, fake_embed, prefer_gorbachev, pool=6)
    assert ids(r.search("first president of Russia", k=1)) == ["p0"]


def test_reranker_cannot_promote_a_passage_outside_its_pool():
    # Same invariant as Project 2: a re-ranker only reorders the candidates it is given.
    prefer_reef = lambda pairs: [1.0 if "Reef" in text else 0.0 for _, text in pairs]
    plain = HybridRetriever(PASSAGES, fake_embed)
    reranked = HybridRetriever(PASSAGES, fake_embed, prefer_reef, pool=2)
    assert "p3" not in ids(plain.search("first president of Russia", k=2))
    assert "p3" not in ids(reranked.search("first president of Russia", k=2))


def test_pool_is_at_least_k():
    seen = []
    def spy(pairs):
        seen.append(len(pairs))
        return [0.0] * len(pairs)
    HybridRetriever(PASSAGES, fake_embed, spy, pool=2).search("Russia", k=4)
    assert seen == [4]


def test_search_tool_formats_results_with_titles():
    tool = make_search_tool(HybridRetriever(PASSAGES, fake_embed), k=2)
    out = tool.fn("first president of Russia")
    assert out.splitlines()[0].startswith("[Boris Yeltsin] Boris Yeltsin became")
    assert len(out.splitlines()) == 2
    assert tool.name == "search" and list(tool.params) == ["query"]


def test_search_tool_truncates_long_passages_and_reports_no_results():
    long = Passage("L", "Long", "word " * 500)
    tool = make_search_tool(HybridRetriever([long], fake_embed), k=1, max_chars=50)
    out = tool.fn("word")
    assert out.endswith("...") and len(out) < 80
    assert tool.fn("   ") == "No results."


def test_agent_answers_a_two_hop_question_through_the_real_retriever(scripted):
    tools = ToolRegistry([make_search_tool(HybridRetriever(PASSAGES, fake_embed), k=1)])
    llm = scripted([
        act("search", query="last leader of the Soviet Union"),
        act("search", query="first president of Russia"),
        act("finish", answer="Boris Yeltsin"),
    ])
    result = Agent(llm, tools).run("Who led Russia after the last Soviet leader?")
    assert result.status == "finished" and result.n_tool_calls == 2
    assert "[Mikhail Gorbachev]" in result.steps[0].observation
    assert "[Boris Yeltsin]" in result.steps[1].observation

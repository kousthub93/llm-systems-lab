import pytest

from agentlab import Agent, Tool, ToolRegistry, build_system_prompt
from agentlab.retrieval import Passage
from agentlab.runner import (HOTPOT_EXTRA, agent_record, clean_answer, closed_book_prompt, format_trace,
                             rag_prompt, run_agent, run_closed_book, run_single_shot, seen_titles, summarize,
                             summarize_by_type)
from conftest import act

EX = {"id": "q1", "question": "Who led Russia after the last Soviet leader?", "answer": "Boris Yeltsin",
      "type": "bridge", "level": "hard", "gold_titles": ["Mikhail Gorbachev", "Boris Yeltsin"]}

P_GORB = Passage("Mikhail Gorbachev", "Mikhail Gorbachev", "Last leader of the Soviet Union.")
P_YELT = Passage("Boris Yeltsin", "Boris Yeltsin", "First president of Russia.")
P_EIFFEL = Passage("Eiffel Tower", "Eiffel Tower", "A tower in Paris.")


class StubRetriever:
    def __init__(self, by_query, default=()):
        self.by_query, self.default = by_query, list(default)

    def search(self, query, k=3):
        return list(self.by_query.get(query, self.default))[:k]


def search_tool(retriever, k=3):
    def search(query):
        hits = retriever.search(query, k)
        return "\n".join(f"[{p.title}] {p.text}" for p in hits) or "No results."
    return Tool("search", "Search.", {"query": "what to find"}, search)


# ------------------------------------------------------------------ small pure helpers

@pytest.mark.parametrize("raw,expected", [
    ("Boris Yeltsin", "Boris Yeltsin"),
    ("Answer: Boris Yeltsin.", "Boris Yeltsin"),
    ("  final answer - \"Boris Yeltsin\"  \n\nBecause he was the first president.", "Boris Yeltsin"),
    ("\n\nyes\n", "yes"),
    ("", ""),
    (None, ""),
])
def test_clean_answer(raw, expected):
    assert clean_answer(raw) == expected


def test_seen_titles_looks_for_the_bracketed_title():
    texts = ["[Mikhail Gorbachev] Last leader...", "[Eiffel Tower] A tower"]
    assert seen_titles(texts, EX["gold_titles"]) == ["Mikhail Gorbachev"]
    assert seen_titles(["Mikhail Gorbachev mentioned without brackets"], EX["gold_titles"]) == []


def test_prompts_contain_the_question_and_the_rules():
    assert "Question: " + EX["question"] in closed_book_prompt(EX["question"])
    p = rag_prompt(EX["question"], [P_GORB, P_YELT])
    assert "[Mikhail Gorbachev] Last leader" in p and "[Boris Yeltsin] First president" in p
    assert "yes or no" in p and p.endswith("Answer:")


# ------------------------------------------------------------------ the three modes

def test_closed_book_scores_the_cleaned_answer(scripted):
    llm = scripted(["Answer: Boris Yeltsin."])
    rec = run_closed_book(llm, EX)
    assert rec["mode"] == "closed-book" and rec["answer"] == "Boris Yeltsin" and rec["em"] == 1.0
    assert "seen_titles" not in rec  # nothing is retrieved in this mode
    assert llm.calls[0][0]["role"] == "user" and EX["question"] in llm.calls[0][0]["content"]


def test_single_shot_records_which_supporting_paragraphs_were_retrieved(scripted):
    retriever = StubRetriever({EX["question"]: [P_GORB, P_EIFFEL]})
    rec = run_single_shot(scripted(["Mikhail Gorbachev"]), retriever, EX, k=2)
    assert rec["mode"] == "single-shot RAG" and rec["em"] == 0.0
    assert rec["seen_titles"] == ["Mikhail Gorbachev"] and rec["gold_seen"] == 0.5 and rec["both_seen"] is False


def test_single_shot_respects_k(scripted):
    llm = scripted(["x"])
    run_single_shot(llm, StubRetriever({EX["question"]: [P_GORB, P_YELT, P_EIFFEL]}), EX, k=1)
    assert "Eiffel" not in llm.calls[0][0]["content"] and "Yeltsin" not in llm.calls[0][0]["content"]


def make_agent(scripted_replies, retriever):
    tools = ToolRegistry([search_tool(retriever)])
    llm = scripted_replies
    return Agent(llm, tools, max_steps=5, system_prompt=build_system_prompt(tools, HOTPOT_EXTRA))


def test_agent_two_hop_run_is_recorded_end_to_end(scripted):
    retriever = StubRetriever({"last Soviet leader": [P_GORB], "first president of Russia": [P_YELT]})
    llm = scripted([act("search", query="last Soviet leader"), act("search", query="first president of Russia"),
                    act("finish", answer="Boris Yeltsin.")])
    rec = run_agent(make_agent(llm, retriever), EX)
    assert rec["mode"] == "agent" and rec["status"] == "finished"
    assert rec["answer"] == "Boris Yeltsin" and rec["em"] == 1.0 and rec["f1"] == 1.0
    assert (rec["n_steps"], rec["n_tool_calls"], rec["n_errors"]) == (3, 2, 0)
    assert rec["both_seen"] is True and rec["gold_seen"] == 1.0
    assert [s["tool"] for s in rec["steps"]] == ["search", "search", "finish"]
    assert rec["steps"][0]["args"] == {"query": "last Soviet leader"} and "[Mikhail Gorbachev]" in rec["steps"][0]["observation"]


def test_agent_that_gives_up_has_an_empty_answer_and_zero_score(scripted):
    retriever = StubRetriever({}, default=[P_EIFFEL])
    llm = scripted([act("search", query=f"q{i}") for i in range(5)])
    rec = run_agent(make_agent(llm, retriever), EX)
    assert rec["status"] == "max_steps" and rec["answer"] == "" and rec["em"] == 0.0 and rec["f1"] == 0.0
    assert rec["both_seen"] is False and rec["gold_seen"] == 0.0


def test_agent_errors_are_counted_and_kept_in_the_trace(scripted):
    retriever = StubRetriever({"q": [P_YELT]})
    llm = scripted(["no json here", act("search", query="q"), act("finish", answer="Boris Yeltsin")])
    rec = run_agent(make_agent(llm, retriever), EX)
    assert rec["n_errors"] == 1 and rec["steps"][0]["tool"] is None and rec["steps"][0]["error"] == "parse"
    assert rec["status"] == "finished" and rec["seen_titles"] == ["Boris Yeltsin"]


def test_a_truncated_observation_only_counts_what_the_model_actually_saw(scripted):
    retriever = StubRetriever({"q": [P_EIFFEL, P_YELT]})
    llm = scripted([act("search", query="q"), act("finish", answer="x")])
    tools = ToolRegistry([search_tool(retriever)])
    agent = Agent(llm, tools, max_steps=3, max_obs_chars=30)   # cut before the second passage's title
    assert run_agent(agent, EX)["seen_titles"] == []


def test_format_trace_shows_raw_reply_parse_result_and_outcome(scripted):
    retriever = StubRetriever({"q": [P_YELT]})
    result = Agent(scripted(["oops", act("search", query="q"), act("finish", answer="Boris Yeltsin")]),
                   ToolRegistry([search_tool(retriever)])).run("Q?")
    text = format_trace(result)
    assert "model said 'oops'" in text and "(no valid action)" in text and "[error: parse]" in text
    assert "observation: [Boris Yeltsin] First president of Russia." in text
    assert text.splitlines()[-1].startswith("=> status=finished answer='Boris Yeltsin'")


# ------------------------------------------------------------------ summaries

def recs():
    base = {"id": "x", "level": "hard", "question": "q", "gold_answer": "g", "gold_titles": ["A"], "answer": ""}
    return [
        {**base, "mode": "closed-book", "type": "bridge", "em": 0.0, "f1": 0.2},
        {**base, "mode": "closed-book", "type": "comparison", "em": 1.0, "f1": 1.0},
        {**base, "mode": "agent", "type": "bridge", "em": 1.0, "f1": 1.0, "status": "finished", "n_steps": 3,
         "n_tool_calls": 2, "n_errors": 0, "gold_seen": 1.0, "both_seen": True, "wall_s": 10.0},
        {**base, "mode": "agent", "type": "bridge", "em": 0.0, "f1": 0.0, "status": "max_steps", "n_steps": 5,
         "n_tool_calls": 4, "n_errors": 2, "gold_seen": 0.5, "both_seen": False, "wall_s": 20.0},
    ]


def test_summarize_averages_per_mode_in_first_seen_order():
    s = summarize(recs())
    assert list(s) == ["closed-book", "agent"]
    assert s["closed-book"] == {"n": 2, "EM": 0.5, "F1": 0.6}  # no agent-only fields invented
    a = s["agent"]
    assert (a["n"], a["EM"], a["F1"]) == (2, 0.5, 0.5)
    assert (a["steps"], a["tool_calls"], a["errors"], a["wall_s"]) == (4.0, 3.0, 1.0, 15.0)
    assert (a["gold_seen"], a["both_seen"]) == (0.75, 0.5)
    assert (a["status_finished"], a["status_max_steps"], a["status_loop"]) == (0.5, 0.5, 0.0)


def test_summarize_by_type_splits_modes_and_question_types():
    s = summarize_by_type(recs())
    assert s["closed-book / bridge"] == {"n": 1, "EM": 0.0, "F1": 0.2}
    assert s["closed-book / comparison"]["EM"] == 1.0
    assert s["agent / bridge"]["n"] == 2

import pytest

from agentlab import Agent
from conftest import act


def test_search_then_finish(scripted, registry):
    llm = scripted([act("search", query="Boris Yeltsin"), act("finish", answer="1991")])
    result = Agent(llm, registry).run("When did Yeltsin become president?")
    assert result.status == "finished" and result.answer == "1991"
    assert (result.n_steps, result.n_tool_calls, result.n_errors) == (2, 1, 0)
    # the model saw the tool result before its second reply
    assert "first president of Russia" in llm.calls[1][-1]["content"]
    assert llm.calls[1][-1]["content"].startswith("Observation: ")


def test_system_prompt_and_question_open_the_conversation(scripted, registry):
    llm = scripted([act("finish", answer="x")])
    Agent(llm, registry).run("Q?")
    system, user = llm.calls[0]
    assert system["role"] == "system" and "finish" in system["content"]
    assert user == {"role": "user", "content": "Q?"}


def test_each_call_gets_its_own_snapshot_of_the_messages(scripted, registry):
    llm = scripted([act("search", query="Gorbachev"), act("finish", answer="x")])
    Agent(llm, registry).run("Q?")
    assert len(llm.calls[0]) == 2 and len(llm.calls[1]) == 4


def test_unparseable_reply_is_fed_back_and_the_run_recovers(scripted, registry):
    llm = scripted(["The answer is probably 1991.", act("finish", answer="1991")])
    result = Agent(llm, registry).run("Q?")
    assert result.status == "finished" and result.n_errors == 1
    assert result.steps[0].error == "parse" and result.steps[0].action is None
    assert "Error: no JSON object" in llm.calls[1][-1]["content"]


def test_unknown_tool_error_lists_available_tools(scripted, registry):
    llm = scripted([act("lookup", query="x"), act("finish", answer="x")])
    result = Agent(llm, registry).run("Q?")
    assert result.steps[0].error == "unknown_tool" and not result.steps[0].executed
    assert "Available tools: search" in result.steps[0].observation


def test_bad_args_are_an_error_the_model_can_fix(scripted, registry):
    llm = scripted([act("search"), act("search", query="Gorbachev"), act("finish", answer="USSR")])
    result = Agent(llm, registry).run("Q?")
    assert result.steps[0].error == "bad_args" and "missing required args" in result.steps[0].observation
    assert result.steps[1].executed and result.status == "finished"


def test_tool_exception_does_not_crash_the_run(scripted, registry):
    llm = scripted([act("boom", x=1), act("finish", answer="gave up")])
    result = Agent(llm, registry).run("Q?")
    assert result.steps[0].error == "tool_failed" and "backend down" in result.steps[0].observation
    assert result.status == "finished"


def test_finish_without_an_answer_is_rejected_then_accepted(scripted, registry):
    llm = scripted([act("finish"), act("finish", answer="  ok  ")])
    result = Agent(llm, registry).run("Q?")
    assert result.steps[0].error == "empty_answer"
    assert result.answer == "ok"


def test_repeated_call_is_warned_not_rerun(scripted, registry):
    calls = []
    registry._tools["search"].fn = lambda query: calls.append(query) or "result"
    llm = scripted([act("search", query="a"), act("search", query="a"), act("finish", answer="x")])
    result = Agent(llm, registry).run("Q?")
    assert calls == ["a"]  # executed once
    assert result.steps[1].error == "repeat" and "already made this exact call" in result.steps[1].observation
    assert result.status == "finished"


def test_same_args_in_a_different_order_still_count_as_a_repeat(scripted, registry):
    llm = scripted([act("echo", text="a", suffix="b"), act("echo", suffix="b", text="a"), act("finish", answer="x")])
    result = Agent(llm, registry).run("Q?")
    assert result.steps[1].error == "repeat"


def test_loop_is_aborted_after_too_many_repeats(scripted, registry):
    llm = scripted([act("search", query="a")] * 3)
    result = Agent(llm, registry, max_steps=10).run("Q?")
    assert result.status == "loop" and result.answer is None
    assert result.n_steps == 3 and result.n_tool_calls == 1


def test_max_repeats_is_configurable(scripted, registry):
    llm = scripted([act("search", query="a")] * 4)
    result = Agent(llm, registry, max_steps=10, max_repeats=2).run("Q?")
    assert result.status == "loop" and result.n_steps == 4


def test_step_budget_ends_the_run(scripted, registry):
    llm = scripted([act("search", query="a"), act("search", query="b"), act("search", query="c")])
    result = Agent(llm, registry, max_steps=3).run("Q?")
    assert result.status == "max_steps" and result.answer is None and result.n_steps == 3


def test_model_is_told_when_one_step_is_left(scripted, registry):
    llm = scripted([act("search", query="a"), act("search", query="b"), act("finish", answer="x")])
    result = Agent(llm, registry, max_steps=3).run("Q?")
    assert "[1 step left" not in result.steps[0].observation
    assert "[1 step left" in result.steps[1].observation
    assert "[1 step left" in llm.calls[2][-1]["content"]


def test_long_observations_are_truncated(scripted, registry):
    registry._tools["search"].fn = lambda query: "x" * 50
    llm = scripted([act("search", query="a"), act("finish", answer="x")])
    result = Agent(llm, registry, max_obs_chars=10).run("Q?")
    assert result.steps[0].observation == "x" * 10 + "... [truncated 40 chars]"


def test_latency_comes_from_the_injected_clock(scripted, registry):
    ticks = iter([0.0, 1.5, 10.0, 10.25])
    llm = scripted([act("search", query="a"), act("finish", answer="x")])
    result = Agent(llm, registry, clock=lambda: next(ticks)).run("Q?")
    assert [s.latency_s for s in result.steps] == [1.5, 0.25]


def test_raw_text_and_parsed_action_are_kept_for_the_trace(scripted, registry):
    raw = 'Sure.\n{"tool": "search", "args": {"query": "Gorbachev"}}'
    result = Agent(scripted([raw, act("finish", answer="x")]), registry).run("Q?")
    assert result.steps[0].raw == raw
    assert result.steps[0].action.tool == "search"


def test_max_steps_must_be_positive(scripted, registry):
    with pytest.raises(ValueError):
        Agent(scripted([]), registry, max_steps=0)


def test_multi_hop_run_uses_the_first_result_to_choose_the_second_call(scripted, registry):
    llm = scripted([
        act("search", query="Gorbachev"),
        act("search", query="Boris Yeltsin"),
        act("finish", answer="Boris Yeltsin"),
    ])
    result = Agent(llm, registry).run("Who led Russia after the last Soviet leader?")
    assert result.answer == "Boris Yeltsin" and result.n_tool_calls == 2

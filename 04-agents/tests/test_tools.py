import pytest

from agentlab import Tool, ToolError, ToolRegistry, build_system_prompt


def test_call_runs_the_tool(registry):
    assert "first president" in registry.call("search", {"query": "Boris Yeltsin"})


def test_results_are_stringified():
    reg = ToolRegistry([Tool("n", "Number.", {}, lambda: 42)])
    assert reg.call("n", {}) == "42"


def test_unknown_tool_lists_the_real_ones(registry):
    with pytest.raises(ToolError, match="unknown tool .*Available tools: search, boom, echo, finish"):
        registry.call("lookup", {})


def test_missing_required_arg(registry):
    with pytest.raises(ToolError, match="missing required args: query"):
        registry.call("search", {})


def test_unexpected_arg_names_the_valid_ones(registry):
    with pytest.raises(ToolError, match="unexpected args: q.*Valid args: query"):
        registry.call("search", {"query": "x", "q": "y"})


def test_optional_arg_may_be_omitted_or_given(registry):
    assert registry.call("echo", {"text": "a"}) == "a"
    assert registry.call("echo", {"text": "a", "suffix": "b"}) == "ab"


def test_tool_exceptions_become_tool_errors(registry):
    with pytest.raises(ToolError, match="RuntimeError: backend down"):
        registry.call("boom", {"x": 1})


def test_wrong_value_type_is_a_tool_error_not_a_crash():
    reg = ToolRegistry([Tool("add", "Add.", {"a": "int", "b": "int"}, lambda a, b: a + b)])
    with pytest.raises(ToolError, match="TypeError"):
        reg.call("add", {"a": 1, "b": "x"})


def test_finish_is_reserved_and_duplicates_rejected(registry):
    with pytest.raises(ValueError, match="reserved"):
        registry.register(Tool("finish", "x", {}, lambda: ""))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Tool("search", "x", {}, lambda: ""))


def test_system_prompt_lists_every_tool_and_finish(registry):
    prompt = build_system_prompt(registry)
    for name in ("search", "boom", "echo", "finish"):
        assert f"- {name}:" in prompt
    assert '"tool"' in prompt

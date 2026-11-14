import pytest

from agentlab.actions import Action, ActionParseError, extract_json_object, parse_action


def test_clean_json():
    assert parse_action('{"tool": "search", "args": {"query": "x"}}') == Action("search", {"query": "x"})


def test_code_fence_and_prose_around_json():
    raw = 'Let me look that up.\n```json\n{"tool": "search", "args": {"query": "Yeltsin"}}\n```\nThat should work.'
    assert parse_action(raw) == Action("search", {"query": "Yeltsin"})


def test_args_default_to_empty():
    assert parse_action('{"tool": "finish"}') == Action("finish", {})


def test_braces_inside_strings_do_not_confuse_the_scan():
    raw = '{"tool": "finish", "args": {"answer": "the set {a, b} and a \\"quote\\""}}'
    assert parse_action(raw).args["answer"] == 'the set {a, b} and a "quote"'


def test_nested_args_survive():
    raw = '{"tool": "t", "args": {"filter": {"year": 1991, "tags": ["a", "b"]}}}'
    assert parse_action(raw).args == {"filter": {"year": 1991, "tags": ["a", "b"]}}


def test_stray_brace_before_the_real_object():
    raw = 'I will use { as a placeholder. {"tool": "search", "args": {"query": "q"}}'
    assert parse_action(raw).tool == "search"


def test_first_valid_object_wins():
    raw = '{"tool": "a"} then {"tool": "b"}'
    assert parse_action(raw).tool == "a"


def test_unclosed_json_is_not_found():
    assert extract_json_object('{"tool": "search", "args": {"query": "x"') is None


@pytest.mark.parametrize("raw", [
    "I think the answer is Paris.",
    "",
    '{"args": {"query": "x"}}',           # no tool
    '{"tool": "", "args": {}}',           # empty tool
    '{"tool": 5, "args": {}}',            # tool is not a string
    '{"tool": "search", "args": "x"}',    # args is not an object
    "[1, 2, 3]",                          # JSON, but not an object
])
def test_unusable_replies_raise(raw):
    with pytest.raises(ActionParseError):
        parse_action(raw)

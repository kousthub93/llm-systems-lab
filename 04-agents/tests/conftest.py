import json

import pytest

from agentlab import Tool, ToolRegistry


class ScriptedLLM:
    """Replies with canned text in order, and remembers what it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # one messages-list snapshot per call

    def __call__(self, messages):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("model was called more times than the script allows")
        return self.replies.pop(0)


def act(tool, **args):
    """Build a model reply that calls a tool."""
    return json.dumps({"tool": tool, "args": args})


@pytest.fixture
def scripted():
    return ScriptedLLM


@pytest.fixture
def registry():
    docs = {
        "Boris Yeltsin": "Boris Yeltsin became the first president of Russia in 1991.",
        "Gorbachev": "Mikhail Gorbachev was the last leader of the Soviet Union.",
    }

    def search(query):
        for key, text in docs.items():
            if key.lower() in query.lower():
                return text
        return "No results."

    def boom(x):
        raise RuntimeError("backend down")

    return ToolRegistry([
        Tool("search", "Search the documents.", {"query": "what to look for"}, search),
        Tool("boom", "Always fails.", {"x": "anything"}, boom),
        Tool("echo", "Return text.", {"text": "text", "suffix": "optional"}, lambda text, suffix="": text + suffix,
             required=("text",)),
    ])

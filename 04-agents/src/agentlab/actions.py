"""Turn raw model text into one action: {"tool": name, "args": {...}}.

Small open models rarely emit clean JSON. They wrap it in code fences, add a
sentence before or after it, or put a stray brace in their reasoning. So we
scan for the first balanced {...} that actually parses as a JSON object,
instead of calling json.loads on the whole reply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


class ActionParseError(ValueError):
    """The model's reply did not contain a usable action."""


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict


def _balanced_end(text: str, start: int) -> int | None:
    """Index just past the brace that closes text[start] == '{', or None.

    Braces inside JSON strings are ignored.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_json_object(text: str) -> dict | None:
    """Return the first JSON object found in text, or None."""
    pos = text.find("{")
    while pos != -1:
        end = _balanced_end(text, pos)
        if end is not None:
            try:
                value = json.loads(text[pos:end])
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                return value
        pos = text.find("{", pos + 1)
    return None


def parse_action(text: str) -> Action:
    obj = extract_json_object(text)
    if obj is None:
        raise ActionParseError("no JSON object found in your reply")
    tool = obj.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise ActionParseError('the JSON object needs a non-empty string field "tool"')
    args = obj.get("args", {})
    if not isinstance(args, dict):
        raise ActionParseError('"args" must be a JSON object')
    return Action(tool=tool.strip(), args=args)

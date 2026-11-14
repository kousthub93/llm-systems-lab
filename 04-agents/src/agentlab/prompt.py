"""System prompt for the JSON action protocol."""

from __future__ import annotations

from .tools import ToolRegistry

_TEMPLATE = """You answer questions by calling tools, one call per reply.

Available tools:
{tools}
- finish: Give the final answer and stop. Args: answer: the short final answer

Reply with exactly one JSON object and nothing else:
{{"tool": "<tool name>", "args": {{...}}}}

Rules:
- Make one tool call per reply, then wait for the Observation.
- Questions may need several lookups. Use what you learned to decide the next one.
- Do not repeat a call you already made.
- When you know the answer, call finish with a short answer (a name, date or number, not a sentence).
- If you cannot find the answer, call finish with your best guess."""


def build_system_prompt(tools: ToolRegistry, extra: str = "") -> str:
    prompt = _TEMPLATE.format(tools=tools.describe())
    return f"{prompt}\n\n{extra.strip()}" if extra.strip() else prompt

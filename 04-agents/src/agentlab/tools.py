"""Tools the agent can call, and a registry that validates calls before running them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

RESERVED_NAMES = {"finish"}  # handled by the loop itself


class ToolError(Exception):
    """A tool call that cannot run or failed. The message is shown to the model.

    kind is recorded in the trace: unknown_tool | bad_args | tool_failed
    """

    def __init__(self, message: str, kind: str = "tool_failed"):
        super().__init__(message)
        self.kind = kind


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # parameter name -> short description
    fn: Callable[..., object]
    required: tuple[str, ...] | None = None  # default: every parameter is required

    def required_params(self) -> tuple[str, ...]:
        return tuple(self.params) if self.required is None else self.required


class ToolRegistry:
    def __init__(self, tools: tuple[Tool, ...] | list[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in RESERVED_NAMES:
            raise ValueError(f'"{tool.name}" is reserved for ending the run')
        if tool.name in self._tools:
            raise ValueError(f'tool "{tool.name}" is already registered')
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """Tool list for the system prompt."""
        lines = []
        for tool in self._tools.values():
            args = ", ".join(f"{k}: {v}" for k, v in tool.params.items()) or "none"
            lines.append(f"- {tool.name}: {tool.description} Args: {args}")
        return "\n".join(lines)

    def call(self, name: str, args: dict) -> str:
        """Validate and run one call. Raises ToolError with a message the model can act on."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(
                f'unknown tool "{name}". Available tools: {", ".join(self.names())}, finish', "unknown_tool"
            )
        missing = [p for p in tool.required_params() if p not in args]
        if missing:
            raise ToolError(f'tool "{name}" is missing required args: {", ".join(missing)}', "bad_args")
        unexpected = [a for a in args if a not in tool.params]
        if unexpected:
            raise ToolError(
                f'tool "{name}" got unexpected args: {", ".join(unexpected)}. '
                f'Valid args: {", ".join(tool.params) or "none"}',
                "bad_args",
            )
        try:
            return str(tool.fn(**args))
        except ToolError:
            raise
        except Exception as exc:  # tool bugs and bad values must not crash the run
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

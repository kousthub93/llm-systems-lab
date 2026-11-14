"""The agent loop: ask the model for an action, run it, show the result, repeat.

The model is any function from chat messages to reply text, so the loop can be
tested with a scripted fake and run with a real model without changing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .actions import Action, ActionParseError, parse_action
from .prompt import build_system_prompt
from .tools import ToolError, ToolRegistry

LLM = Callable[[list[dict]], str]


@dataclass
class Step:
    index: int
    raw: str                      # what the model said
    action: Action | None         # parsed action, None if the reply was unusable
    observation: str              # what the model was shown next ("" on the final step)
    error: str | None = None      # parse | unknown_tool | bad_args | tool_failed | repeat | empty_answer
    executed: bool = False        # a tool function actually ran
    latency_s: float = 0.0        # time spent in the model call


@dataclass
class AgentResult:
    status: str                   # finished | max_steps | loop
    answer: str | None
    steps: list[Step] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_tool_calls(self) -> int:
        return sum(s.executed for s in self.steps)

    @property
    def n_errors(self) -> int:
        return sum(s.error is not None for s in self.steps)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        max_steps: int = 6,
        max_obs_chars: int = 2000,
        max_repeats: int = 1,
        system_prompt: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        """
        max_steps      hard cap on model calls per question
        max_obs_chars  tool output longer than this is cut before the model sees it
        max_repeats    identical calls tolerated (with a warning, not executed) before
                       the run is aborted as a loop
        """
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm, self.tools = llm, tools
        self.max_steps, self.max_obs_chars, self.max_repeats = max_steps, max_obs_chars, max_repeats
        self.system_prompt = system_prompt or build_system_prompt(tools)
        self.clock = clock

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_obs_chars:
            return text
        return text[: self.max_obs_chars] + f"... [truncated {len(text) - self.max_obs_chars} chars]"

    def run(self, question: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        steps: list[Step] = []
        seen: dict[str, int] = {}

        for i in range(self.max_steps):
            t0 = self.clock()
            raw = self.llm(list(messages))
            latency = self.clock() - t0
            messages.append({"role": "assistant", "content": raw})
            step = Step(index=i, raw=raw, action=None, observation="", latency_s=latency)
            steps.append(step)

            try:
                action = parse_action(raw)
            except ActionParseError as exc:
                step.error, obs = "parse", (
                    f'Error: {exc}. Reply with one JSON object like {{"tool": "...", "args": {{...}}}}.'
                )
            else:
                step.action = action
                if action.tool == "finish":
                    answer = str(action.args.get("answer", "")).strip()
                    if answer:
                        return AgentResult("finished", answer, steps)
                    step.error, obs = "empty_answer", 'Error: finish needs a non-empty "answer" arg.'
                else:
                    key = f"{action.tool}:{json.dumps(action.args, sort_keys=True, default=str)}"
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] > 1 + self.max_repeats:
                        step.observation = "Aborted: the same call was repeated too many times."
                        return AgentResult("loop", None, steps)
                    if seen[key] > 1:
                        step.error, obs = "repeat", (
                            "You already made this exact call, so it was not run again. "
                            "Try a different query or call finish."
                        )
                    else:
                        try:
                            obs = self._clip(self.tools.call(action.tool, action.args))
                            step.executed = True
                        except ToolError as exc:
                            step.error, obs = exc.kind, f"Error: {exc}"

            if self.max_steps - i - 1 == 1:
                obs += "\n[1 step left: call finish with your best answer.]"
            step.observation = obs
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        return AgentResult("max_steps", None, steps)

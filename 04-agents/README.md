# 04 · Agents: a hand-rolled tool-calling loop

`agentlab` is a small agent loop written from scratch: the model picks an action, the code runs it, the result goes back to the model, and this repeats until the model answers or a budget runs out. It has no runtime dependencies. The model is just a function from chat messages to text, so the same loop runs against a scripted fake in tests and a real open model in Colab.

**Status:** step 1 of 4. The loop, tool registry, parser and tests are done. The retriever tool, the Qwen runner and the HotpotQA evaluation come next (see below).

## Why

Project 2's RAG pipeline does one retrieval and one answer. Questions that need two lookups break it, because the second query depends on the first answer. An agent handles that, and most of the engineering is the code around the model: what to do with malformed output, bad arguments, tool failures, repeated calls and runaway runs.

## Set up (WSL / Linux)

```bash
cd 04-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 44 tests
python examples/demo.py     # watch a scripted run
```

## Protocol

The model replies with one JSON object per turn, and `finish` ends the run:

```json
{"tool": "search", "args": {"query": "Boris Yeltsin"}}
{"tool": "finish", "args": {"answer": "1991"}}
```

I use a plain JSON protocol instead of a model's native function-calling format so the loop works with any model and every failure path can be tested with canned text.

## How the loop handles failure

| What goes wrong | What happens | Trace `error` |
|---|---|---|
| Reply has no usable JSON | Error shown to the model, run continues | `parse` |
| Tool name does not exist | Error lists the real tools | `unknown_tool` |
| Missing or unexpected args | Error names the valid args | `bad_args` |
| Tool raises an exception | Exception text shown to the model | `tool_failed` |
| Same call made again | Not re-run; model is told to vary or finish | `repeat` |
| Same call made too often | Run aborted, status `loop` | n/a |
| `finish` with no answer | Rejected, run continues | `empty_answer` |
| Step budget used up | Status `max_steps`, no answer | n/a |

Every failure costs a step, so a model that keeps failing hits the budget instead of running forever. On the second-to-last step the model is told it has one step left.

Parsing is deliberately forgiving: it finds the first balanced `{...}` that parses as a JSON object, so code fences, a sentence before the JSON, or a stray brace in the model's reasoning do not break a run.

## Layout

```
src/agentlab/
  actions.py   find and validate the JSON action in a reply
  tools.py     Tool, ToolRegistry (validates args before running), ToolError
  prompt.py    system prompt built from the tool list
  loop.py      Agent.run(): the loop, budgets, loop detection, per-step trace
tests/         44 tests, all using a scripted fake model
examples/      demo.py
```

## Next

1. A `search` tool backed by the Project 2 retriever (hybrid + re-ranker).
2. A Colab runner that plugs Qwen2.5-7B-Instruct (4-bit) in as the model.
3. HotpotQA evaluation: agent vs single-shot RAG on EM and F1, plus steps per question, error rates and cost, scored with `ragevals`.
4. A LangGraph port of the same loop, and a write-up comparing the two.

## Limitations

- Nothing here has run against a real model yet. The tests prove the loop handles each failure path. They say nothing about how often a real model triggers them.
- Tool arguments are checked by name only, not by type. A wrong type surfaces as a `tool_failed` error from the tool itself.

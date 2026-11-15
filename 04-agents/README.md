# 04 · Agents: a hand-rolled tool-calling loop

`agentlab` is a small agent loop written from scratch: the model picks an action, the code runs it, the result goes back to the model, and this repeats until the model answers or a budget runs out. It has no runtime dependencies. The model is just a function from chat messages to text, so the same loop runs against a scripted fake in tests and a real open model in Colab.

**Status:** step 3 of 4 is built but not yet run. The loop, tool registry, retriever, `search` tool, HotpotQA data prep and the three-mode runner are done, with 88 tests. The Colab notebook that runs them against Qwen has not been executed yet, so there are no results here yet.

## Why

Project 2's RAG pipeline does one retrieval and one answer. Questions that need two lookups break it, because the second query depends on the first answer. An agent handles that, and most of the engineering is the code around the model: what to do with malformed output, bad arguments, tool failures, repeated calls and runaway runs.

## Set up (WSL / Linux)

```bash
cd 04-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 88 tests
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
  retrieval.py HybridRetriever (dense + BM25 + RRF + optional re-rank) and make_search_tool
  hotpot.py    HotpotQA row parsing, corpus pooling, EM/F1, retrieval-ceiling check
  runner.py    closed-book / single-shot / agent runs, per-question records, summaries, trace printing
notebooks/     hotpotqa_agent.ipynb: the Colab runner (Qwen2.5-7B-Instruct, 4-bit)
tests/         88 tests, all using a scripted fake model and fake embedders
examples/      demo.py
```

## The search tool

`agentlab.retrieval` is the Project 2 retrieval recipe wrapped in a class: bge-small dense search and BM25 (with the stopword tokenizer) fused with weighted RRF, then an optional cross-encoder re-rank of a candidate pool. Differences from Project 2: no chunking (HotpotQA paragraphs are already short), and the title is part of the indexed text.

The models are injected as plain functions, so the logic is tested with fakes and no GPU. In Colab, `HybridRetriever.from_models(passages)` loads the real ones (`pip install -e ".[retrieval]"`). `make_search_tool(retriever, k=3)` turns a retriever into the agent's `search` tool; results come back as `[Title] passage text`.

It keeps the Project 2 invariant, which has a test: a re-ranker only reorders its pool, so it cannot promote a passage the hybrid step left out.

## The experiment

Same model (Qwen2.5-7B-Instruct, 4-bit), same 100 HotpotQA questions, three modes:

| mode | what it does |
|---|---|
| closed-book | answers from memory |
| single-shot RAG | one search with the raw question, answers from the top 5 passages |
| agent | the loop in this package: the model picks searches and decides when to stop |

The corpus pools the paragraphs of 1000 questions (the 100 evaluated plus 900 more), so the agent searches thousands of paragraphs, not just its own 10. That is easier than open-domain Wikipedia, and the write-up has to say so.

Each record also stores `gold_seen` and `both_seen`: which of the two supporting paragraphs the model was actually shown. That separates "retrieval never found the evidence" from "the model had it and still got it wrong".

**Run it:** push this folder, then open
`https://colab.research.google.com/github/kousthub93/llm-systems-lab/blob/main/04-agents/notebooks/hotpotqa_agent.ipynb`
on a T4 runtime. Read the smoke-test traces in P4-3 before starting the long run in P4-4.

## Next

1. ~~A `search` tool backed by the Project 2 retriever.~~ Done.
2. ~~A Colab runner that plugs Qwen in as the model.~~ Built, not yet run.
3. Run it, read the traces, fix what they show, then record the results here. Score the output with `ragevals` (needs a small extension for agent fields such as steps and errors).
4. A LangGraph port of the same loop, and a write-up comparing the two.

## Limitations

- Nothing here has run against a real LLM or the real embedding models yet. The retriever's tests use a crude hashed bag-of-words embedder, so real bge and MiniLM behavior is only checked once it runs in Colab. The tests prove the loop handles each failure path. They say nothing about how often a real model triggers them.
- Tool arguments are checked by name only, not by type. A wrong type surfaces as a `tool_failed` error from the tool itself.

"""Run a question through each of the three modes and record what happened.

closed-book    the model answers from memory
single-shot    one search with the raw question, then answer from the top-k passages (plain RAG)
agent          the tool-calling loop; the model decides what to search for, and when to stop

Each run_* function takes `llm(messages) -> str` and returns one flat, JSON-serializable record.
Timing and token counts depend on the model adapter, so the notebook adds them afterwards.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .hotpot import exact_match, f1_score
from .loop import Agent, AgentResult

SHORT_ANSWER_RULES = ("Answer with a short phrase (a few words), or just yes or no for a yes-or-no question. "
                      "Do not explain.")

HOTPOT_EXTRA = (
    "The documents are Wikipedia paragraphs. Many questions need two lookups: find one fact, then use it "
    "to search for the next. Search with specific names and keywords. For a yes-or-no question, finish "
    "with yes or no."
)


def closed_book_prompt(question: str) -> str:
    return f"Answer the question from your own knowledge. {SHORT_ANSWER_RULES}\n\nQuestion: {question}\nAnswer:"


def format_passages(passages) -> str:
    return "\n".join(f"[{p.title}] {p.text}" for p in passages)


def rag_prompt(question: str, passages) -> str:
    return (f"Answer the question using the passages below. {SHORT_ANSWER_RULES}\n\n"
            f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer:")


def clean_answer(text) -> str:
    """First non-empty line, minus an 'Answer:' prefix, quotes and a trailing period."""
    lines = [ln.strip() for ln in str(text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    ans = re.sub(r"^(final\s+)?answer\s*[:\-]\s*", "", lines[0], flags=re.I)
    return ans.strip().strip("\"'").rstrip(".").strip()


def seen_titles(texts, gold_titles) -> list[str]:
    """Which supporting paragraphs appeared, as '[Title]', in any of the texts the model was shown."""
    return [t for t in gold_titles if any(f"[{t}]" in s for s in texts)]


def _base(ex: dict, mode: str, answer: str) -> dict:
    return {
        "mode": mode, "id": ex["id"], "type": ex["type"], "level": ex["level"],
        "question": ex["question"], "gold_answer": ex["answer"], "gold_titles": ex["gold_titles"],
        "answer": answer, "em": exact_match(answer, ex["answer"]), "f1": f1_score(answer, ex["answer"]),
    }


def _with_seen(rec: dict, ex: dict, texts) -> dict:
    seen = seen_titles(texts, ex["gold_titles"])
    rec.update(seen_titles=seen, gold_seen=len(seen) / len(ex["gold_titles"]),
               both_seen=len(seen) == len(ex["gold_titles"]))
    return rec


def run_closed_book(llm, ex: dict) -> dict:
    raw = llm([{"role": "user", "content": closed_book_prompt(ex["question"])}])
    rec = _base(ex, "closed-book", clean_answer(raw))
    rec["raw"] = raw
    return rec


def run_single_shot(llm, retriever, ex: dict, k: int = 5) -> dict:
    passages = retriever.search(ex["question"], k)
    raw = llm([{"role": "user", "content": rag_prompt(ex["question"], passages)}])
    rec = _base(ex, "single-shot RAG", clean_answer(raw))
    rec["raw"] = raw
    return _with_seen(rec, ex, [format_passages(passages)])


def agent_record(ex: dict, result: AgentResult, mode: str = "agent") -> dict:
    rec = _base(ex, mode, clean_answer(result.answer))
    rec.update(
        status=result.status, n_steps=result.n_steps, n_tool_calls=result.n_tool_calls, n_errors=result.n_errors,
        steps=[{"tool": s.action.tool if s.action else None, "args": s.action.args if s.action else None,
                "error": s.error, "observation": s.observation} for s in result.steps],
    )
    return _with_seen(rec, ex, [s.observation for s in result.steps])


def run_agent(agent: Agent, ex: dict, mode: str = "agent") -> dict:
    return agent_record(ex, agent.run(ex["question"]), mode)


def format_trace(result: AgentResult, width: int = 300) -> str:
    """Readable multi-line trace of one agent run, for smoke tests."""
    def cut(s):
        s = " ".join(str(s).split())
        return s if len(s) <= width else s[:width] + "..."
    lines = []
    for s in result.steps:
        act = f"{s.action.tool} {s.action.args}" if s.action else "(no valid action)"
        lines.append(f"step {s.index}: model said {cut(s.raw)!r}")
        lines.append(f"         parsed as {act}" + (f"   [error: {s.error}]" if s.error else ""))
        if s.observation:
            lines.append(f"         observation: {cut(s.observation)}")
    lines.append(f"=> status={result.status} answer={result.answer!r} steps={result.n_steps} "
                 f"tool_calls={result.n_tool_calls} errors={result.n_errors}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- summaries

def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _summarize_group(recs: list[dict]) -> dict:
    out = {"n": len(recs), "EM": _mean(r["em"] for r in recs), "F1": _mean(r["f1"] for r in recs)}
    for key, name in (("gold_seen", "gold_seen"), ("both_seen", "both_seen"), ("n_steps", "steps"),
                      ("n_tool_calls", "tool_calls"), ("n_errors", "errors"), ("wall_s", "wall_s"),
                      ("prompt_tokens", "prompt_tokens")):
        m = _mean(float(r[key]) if key in r else None for r in recs)
        if m is not None:
            out[name] = m
    if any("status" in r for r in recs):
        for status in ("finished", "max_steps", "loop"):
            out[f"status_{status}"] = sum(r.get("status") == status for r in recs) / len(recs)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def summarize(records: list[dict]) -> dict:
    """Per-mode averages. Modes appear in the order first seen."""
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r["mode"]].append(r)
    return {mode: _summarize_group(recs) for mode, recs in groups.items()}


def summarize_by_type(records: list[dict]) -> dict:
    """EM and F1 per (mode, question type), e.g. bridge vs comparison."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["mode"], r["type"])].append(r)
    return {f"{mode} / {typ}": {"n": len(g), "EM": round(_mean(x["em"] for x in g), 3),
                                "F1": round(_mean(x["f1"] for x in g), 3)} for (mode, typ), g in groups.items()}

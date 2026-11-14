"""Watch the loop run with a scripted 'model'. No GPU, no downloads.

    python examples/demo.py
"""
import json

from agentlab import Agent, Tool, ToolRegistry

DOCS = {
    "gorbachev": "Mikhail Gorbachev was the last leader of the Soviet Union. He resigned in 1991.",
    "yeltsin": "Boris Yeltsin became the first president of Russia in 1991.",
}


def search(query: str) -> str:
    for key, text in DOCS.items():
        if key in query.lower():
            return text
    return "No results."


def call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


# Stand-in for a real model. Note the sloppy second reply: prose plus a code fence.
script = iter([
    call("search", query="last Soviet leader Gorbachev"),
    'Now I need the next person.\n```json\n' + call("search", query="Boris Yeltsin") + '\n```',
    call("finish", answer="Boris Yeltsin"),
])

tools = ToolRegistry([Tool("search", "Search the documents.", {"query": "what to look for"}, search)])
agent = Agent(lambda messages: next(script), tools, max_steps=5)
result = agent.run("Who led Russia after the last Soviet leader?")

for s in result.steps:
    print(f"step {s.index}: {s.action.tool if s.action else 'UNPARSED'} {s.action.args if s.action else ''}")
    if s.observation:
        print(f"   observation: {s.observation}")
print(f"\nstatus={result.status} answer={result.answer!r} steps={result.n_steps} "
      f"tool_calls={result.n_tool_calls} errors={result.n_errors}")

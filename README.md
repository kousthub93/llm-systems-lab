# llm-systems-lab

Hands-on projects on the engineering side of LLM applications: retrieval, RAG, evaluation, agents, LLMOps, fine-tuning and guardrails. Each project is built around measurement, so every claim comes with a number, the setup that produced it and the caveats.

All projects use public datasets and open models, and run on a free Google Colab GPU.

## Projects

| # | Project | Focus | Status |
| --- | --- | --- | --- |
| 01 | [Retrieval benchmark](01-retrieval-benchmark) | Dense vs BM25 vs hybrid retrieval, embedding model size, evaluation harness | Done |
| 02 | RAG pipeline | Chunking, re-ranking, grounded generation with citations, abstain gate | In progress |
| 03 | Evals | Regression suite for the RAG system, run on every change | Planned |
| 04 | Agents | Multi-agent orchestration with failure isolation and tracing | Planned |
| 05 | LLMOps | Prompt release pipeline, canary rollout, cost and latency dashboards | Planned |
| 06 | Fine-tuning | LoRA fine-tune targeted at eval failures, compared against prompting | Planned |
| 07 | Guardrails | Prompt-injection and PII defenses, red-team report | Planned |

## Highlights so far

- **Project 01:** fusing dense and BM25 retrieval raised recall@5 from 0.884 to 0.938 on 500 labeled SQuAD v2 queries, and beat a larger dense model at lower latency. A tokenizer fix alone moved BM25 recall@5 from 0.704 to 0.886.

## Approach

- Build the evaluation harness first, then change one variable at a time.
- Report measured numbers only, and state what the data does not show.
- Keep each project runnable end to end in a single Colab notebook.

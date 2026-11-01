# llm-systems-lab

Hands-on projects on the engineering side of LLM applications: retrieval, RAG, evaluation, agents, LLMOps, fine-tuning and guardrails. Each project is built around measurement, so every claim comes with a number, the setup that produced it and the caveats.

All projects use public datasets and open models, and run on a free Google Colab GPU.

## Projects

| # | Project | Focus | Status |
| --- | --- | --- | --- |
| 01 | [Retrieval benchmark](01-retrieval-benchmark) | Dense vs BM25 vs hybrid retrieval, embedding model size, evaluation harness | Done |
| 02 | [RAG pipeline](02-rag-pipeline) | Chunking, hybrid retrieval, cross-encoder re-ranking, grounded generation with citations | Done (abstain gate deferred) |
| 03 | Evals | Regression suite for the RAG system, run on every change | Next |
| 04 | Agents | Multi-agent orchestration with failure isolation and tracing | Planned |
| 05 | LLMOps | Prompt release pipeline, canary rollout, cost and latency dashboards | Planned |
| 06 | Fine-tuning | LoRA fine-tune targeted at eval failures, compared against prompting | Planned |
| 07 | Guardrails | Prompt-injection and PII defenses, red-team report | Planned |

## Highlights so far

- **Project 01:** fusing dense and BM25 retrieval raised recall@5 from 0.884 to 0.938 on 500 labeled SQuAD v2 queries, and beat a larger dense model at lower latency. A tokenizer fix alone moved BM25 recall@5 from 0.704 to 0.886.
- **Project 02:** a MiniLM cross-encoder re-ranker raised recall@1 from 0.616 to 0.782 on 500 questions over long documents, and retrieved context raised answer F1 from 0.24 to 0.79. The LLM never abstained when the right chunk was missing, and a larger re-ranker (bge-reranker-base) scored below the no-re-rank baseline.

## Approach

- Build the evaluation harness first, then change one variable at a time.
- Keep the no-change baseline in every results table.
- Report measured numbers only, and state what the data does not show.
- Keep each project runnable end to end in a single Colab notebook.


# 01 · Retrieval Benchmark: Dense, BM25 and Hybrid

How much do retrieval choices matter before an LLM ever sees a prompt? This project builds a small evaluation harness and uses it to compare dense embeddings, BM25 and hybrid fusion on labeled questions.

**Headline:** fusing bge-small embeddings with a properly tokenized BM25 through reciprocal rank fusion raised recall@5 from 0.884 to 0.938. The fused small-model retriever also beat dense bge-large at recall@1 and recall@5, with lower latency.

## Setup

- **Corpus:** the first 3,000 unique passages from SQuAD v2 (train). One passage is one vector; no chunking.
- **Queries:** 500 answerable questions sampled with seed 42. Ground truth is the passage each question was written from.
- **Metrics:** recall@1/3/5/10, MRR, and p50/p95 per-query latency.
- **Harness contract:** a retriever is any function `retrieve_fn(question, top_k)` that returns ranked passage IDs. Every config is scored by the same code, so rows are directly comparable.
- **Retrievers:**
  - Dense: `BAAI/bge-small-en-v1.5` and `bge-large-en-v1.5`, normalized embeddings, exact dot-product search over all passages.
  - Sparse: BM25 (`rank_bm25`) with two tokenizers.
  - Hybrid: weighted reciprocal rank fusion (k=60, top 50 from each list).
- **Environment:** Google Colab, T4 GPU.

## Results

Sorted by recall@5. Latency comes from one warmed-up pass over all configs, so it is comparable within this table only.

| Config | recall@1 | recall@5 | recall@10 | MRR | p50 latency (ms) |
| --- | --- | --- | --- | --- | --- |
| Hybrid RRF, bge-large + BM25 v2 (1:1) | 0.742 | 0.950 | 0.976 | 0.834 | 35.7 |
| Hybrid RRF, bge-small + BM25 v2 (1:1) | 0.734 | 0.938 | 0.966 | 0.819 | 21.6 |
| Hybrid RRF, bge-small + BM25 v2 (dense weight 2:1) | 0.738 | 0.930 | 0.960 | 0.818 | 21.6 |
| Hybrid RRF, bge-small + BM25 v2 (BM25 weight 2:1) | 0.714 | 0.928 | 0.964 | 0.809 | 21.4 |
| Dense bge-large + query prefix | 0.698 | 0.924 | 0.962 | 0.788 | 26.5 |
| Dense bge-large | 0.704 | 0.912 | 0.972 | 0.795 | 27.4 |
| BM25 v2 (regex tokens + stopwords) | 0.682 | 0.886 | 0.930 | 0.767 | 4.2 |
| Dense bge-small | 0.652 | 0.884 | 0.928 | 0.749 | 14.1 |
| BM25 v1 (lowercase + whitespace split) | 0.506 | 0.704 | 0.766 | 0.587 | 8.1 |

## Findings

1. **Fusion beat either retriever alone.** With bge-small, recall@5 rose from 0.884 (dense) and 0.886 (BM25 v2) to 0.938, and recall@1 from 0.652 and 0.682 to 0.734.
2. **Each retriever finds queries the other misses.** At top 5, both found 398 of 500, dense only 44, BM25 v2 only 45, neither 13. The union covers 487 queries (97.4%), which is the ceiling for any fusion; the best fusion reached 93.8% with bge-small.
3. **Fusion beat a larger model, at lower latency.** Dense bge-large improved on bge-small but roughly doubled p50 latency (27.4 ms vs 14.1 ms). Hybrid with the small model scored higher at recall@1 (0.734 vs 0.704) and recall@5 (0.938 vs 0.912) at 21.6 ms. Dense bge-large still led at recall@10 (0.972 vs 0.966). Adding bge-large to the hybrid gave a smaller gain, 0.938 to 0.950 recall@5, for about 1.65x the latency.
4. **Tokenization drove BM25's quality.** Switching from lowercase-and-split to regex tokens plus stopword removal lifted recall@5 from 0.704 to 0.886. Both changes were made together, so their individual effects are not separated. A hybrid built on the weak BM25 v1 reached only 0.858 recall@5, below dense.
5. **Two knobs did not matter.** Three fusion weightings landed within 1.0 point of each other at recall@5. BGE's query prefix gave no consistent gain on bge-large.
6. **The ranking gap remains.** The best hybrid has recall@10 of 0.976 but recall@1 of 0.742: the right passage is usually retrieved but not first. This is the motivation for re-ranking in the next project.

## Caveats

- SQuAD questions were written by people reading the passage, so they likely share vocabulary with it. That favors lexical matching and may not carry over to real user queries.
- 500 queries and one seed. One standard error is about 1.5 points at these recall levels. The fusion gains of 4 to 8 points are well above that, but no significance test was run, and differences of 1 to 3 points between similar configs are within noise.
- Fusion weights and models were compared on the same 500 queries used for reporting, with no held-out set.
- 3,000 passages is small enough that exact search is fast. An approximate index (FAISS HNSW) was not benchmarked, so this project makes no index-speed claims.
- Latency was measured on shared Colab hardware and varies between runs.

## Run it

Open `notebook.ipynb` in Google Colab, set the runtime to a T4 GPU, and choose Runtime → Run all. Embedding the passages takes a few minutes. The notebook writes `project1_results.csv`.

## Next

[02-rag-pipeline](../02-rag-pipeline): chunking on long documents, a cross-encoder re-ranker, grounded generation with citations, and an abstain gate.

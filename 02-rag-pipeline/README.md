
# 02 · RAG Pipeline: Chunking, Re-ranking and Grounded Generation

A retrieval-augmented generation pipeline over long documents, built and measured one stage at a time: chunking, hybrid retrieval, cross-encoder re-ranking and answer generation with citations. Each stage is scored on its own so that gains and failures can be attributed to the stage that caused them.

**Headline:** adding a cross-encoder re-ranker to hybrid retrieval raised recall@1 from 0.616 to 0.782 and recall@5 from 0.844 to 0.924 on 500 questions. Feeding the re-ranked chunks to an open LLM raised answer F1 from 0.24 (no context) to 0.79 on 100 questions. The model never answered "I don't know", even when the right chunk was missing, and for 27% to 30% of answers with citations no cited chunk contained the answer.

## Pipeline

```
long documents
  → chunks (100 words, 20-word overlap)
  → hybrid retrieval: bge-small embeddings + BM25, fused with reciprocal rank fusion
  → top 20 candidates
  → cross-encoder re-ranker (MiniLM) → top 5
  → Phi-3.5-mini answers using only those chunks and cites chunk ids
```

## Setup

- **Corpus:** 150 SQuAD v2 articles (random sample, seed 42). Each article's paragraphs are joined into one long document, then chunked. At 100 words with 20 overlap this gives 9,652 chunks.
- **Questions:** 500 answerable questions for retrieval metrics. The first 100 of a seeded sample are used for generation.
- **Ground truth:** SQuAD gives each answer's character position within its paragraph. I shift that into the long document and mark every chunk whose character range fully contains the answer as correct. Each label is checked (`text[start:end] == answer`) and inconsistent ones are skipped. A question whose answer is cut by a chunk boundary has no correct chunk and counts as a miss.
- **Models:** `BAAI/bge-small-en-v1.5` (embeddings), BM25 via `rank_bm25`, `cross-encoder/ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base` (re-rankers), `microsoft/Phi-3.5-mini-instruct` in fp16 with greedy decoding and at most 48 new tokens (generator).
- **Environment:** Google Colab, T4 GPU.

## Stage 1: chunking

Only part of the planned sweep finished: 3 of 8 chunk setups (6 of 16 rows). Sizes are in words; recall is recall@5.

| Chunk size | Overlap | Answers cut by a boundary | Dense | Hybrid |
| --- | --- | --- | --- | --- |
| 50 | 0 | 3.0% | 0.724 | 0.754 |
| 50 | 10 | 0.0% | 0.782 | 0.828 |
| 100 | 0 | 2.0% | 0.740 | 0.830 |

- Overlap removed the cut-answer problem at 50 words (3.0% to 0.0%). The recall gain from overlap (5.8 points dense, 7.4 hybrid) is larger than that alone can explain, because overlap also puts the answer in two chunks, which makes a hit easier.
- Hybrid beat dense in all three completed pairs.
- Different chunk sizes put different amounts of text in the top 5, so these rows are not a like-for-like size comparison.
- The setup used for the rest of the project, 100 words with 20 overlap, was not one of the completed sweep rows. It was chosen from these partial results. It had 0.0% cut answers and a hybrid recall@5 of 0.844.

## Stage 2: re-ranking (500 questions)

Candidates come from hybrid retrieval, and the re-ranker reorders them. `pool` is how many candidates the re-ranker sees.

| Config | Pool | recall@1 | recall@5 | recall@10 | MRR | p50 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid, no re-rank | - | 0.616 | 0.844 | 0.916 | 0.717 | 39.5 |
| + MiniLM | 10 | 0.770 | 0.902 | 0.916 | 0.828 | 77.9 |
| + MiniLM | 20 | 0.782 | 0.924 | 0.948 | 0.844 | 99.0 |
| + MiniLM | 50 | 0.790 | 0.932 | 0.958 | 0.853 | 140.8 |
| + bge-reranker-base | 10 | 0.488 | 0.820 | 0.916 | 0.629 | 149.2 |
| + bge-reranker-base | 20 | 0.416 | 0.750 | 0.884 | 0.550 | 263.1 |
| + bge-reranker-base | 50 | 0.340 | 0.600 | 0.748 | 0.455 | 613.2 |

The best any re-ranker can do is bounded by how often the correct chunk is in its pool. For hybrid retrieval that ceiling is recall@10 = 0.916, recall@20 = 0.956 and recall@50 = 0.982.

- **MiniLM at pool 20** improved recall@1 by 16.6 points, recall@5 by 8.0, recall@10 by 3.2 and MRR by 12.7, for about 59 ms more per question (2.5x).
- **Pool 20 was the chosen setting.** Going from 10 to 20 added 2.2 points of recall@5 for about 21 ms. Going from 20 to 50 added 0.8 points for about 42 ms, which is within noise.
- **Sanity check:** at pool 10, recall@10 equals the baseline (0.916) for both re-rankers. A re-ranker only reorders its pool, so it cannot change which questions have the answer in it.
- **`bge-reranker-base` scored below the no-re-rank baseline at every pool size and got worse as the pool grew.** I did not find out why.

## Stage 3: generation (100 questions)

| Mode | EM | F1 | Correct chunk in the 5 given | F1 if correct chunk given | F1 if not |
| --- | --- | --- | --- | --- | --- |
| Closed-book (no context) | 0.10 | 0.242 | - | - | - |
| Hybrid top 5 | 0.58 | 0.766 | 0.91 | 0.821 | 0.201 |
| Hybrid + MiniLM top 5 | 0.60 | 0.792 | 0.95 | 0.833 | 0.000 |

| Mode | Never says "I don't know" | Answers with citations | Citations all among the 5 given | A cited chunk holds the answer | Prompt tokens | p50 generation (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Closed-book | yes | - | - | - | 46 | 0.31 |
| Hybrid top 5 | yes (0 of 100) | 100% | 97% | 73% | 961 | 2.31 |
| Hybrid + MiniLM top 5 | yes (0 of 100) | 100% | 93% | 70% | 968 | 2.34 |

Citation columns are shares of answers that included citations.

## Findings

1. **Retrieval is the main lever.** Giving the model retrieved context lifted F1 from 0.24 to 0.77 to 0.79, far outside noise.
2. **Re-ranking clearly helped retrieval and barely moved end-to-end answers at this sample size.** Re-ranked context scored 2.6 F1 points above plain hybrid, with 4 more questions having a correct chunk in the top 5. With 100 questions the standard error is about 4 points of F1, so that difference is not confirmed. The retrieval gain from Stage 2 (500 questions) is the stronger evidence.
3. **The prompt did not produce abstention.** The model never said "I don't know" in any mode, including the 9 (hybrid) and 5 (re-ranked) questions where the right chunk was not provided. On those it answered anyway and still cited chunks. These subsets are tiny, so they show the pattern, not a precise rate.
4. **Citations are less reliable than answers.** Between 27% and 30% of answers with citations cited no chunk that contains the answer, even though a correct chunk was available for 91% to 95% of questions and answers scored about 0.83 F1 in that case. The model also sometimes wrote ids that were not among the chunks it was given.
5. **A bigger re-ranker was not a better one.** `bge-reranker-base` was slower and worse than the small MiniLM model on this task.
6. **Context is the cost.** Each prompt carried about 960 tokens, and generation took 2.3 s at the median against 0.3 s without context. That excludes retrieval and re-ranking time.

## What I learned

- **Compute the ceiling before tuning.** Recall at the pool size bounds what a re-ranker can achieve, and the pool-10 invariant doubled as a cheap correctness check.
- **Label automatically, and verify the labels.** Mapping SQuAD answer positions to chunks gave thousands of labeled examples. Checking each label against the answer text catches silent offset bugs.
- **Always keep the no-change baseline in the table.** Without the plain-hybrid row, the `bge-reranker-base` result would have looked like a normal number instead of a regression.
- **Run a small smoke test on real outputs before the full run.** It exposed a parser bug: a malformed chunk id was being silently ignored instead of counted as an invalid citation, which would have flattered the citation metrics.
- **Measure retrieval and generation separately.** Splitting F1 by whether the correct chunk was provided shows where failures come from.
- **Instructions in a prompt are not a safety mechanism.** Telling the model to say "I don't know" did nothing here, so abstention needs a mechanism outside the model.
- **Validity is not correctness for citations.** A cited id can exist and still not support the answer.

## Caveats

- The chunking sweep is incomplete, and the chosen setup was not part of it.
- Generation was scored on 100 questions, one run, greedy decoding, one LLM.
- The pool size, models and chunking were chosen on the same questions used to report results. There is no held-out set.
- The `bge-reranker-base` result is unexplained and may reflect a setup problem rather than the model's quality, so it applies only to this configuration.
- Ground truth marks the location of the answer span. A cited chunk can contain the answer text without being marked correct, so `a cited chunk holds the answer` is a strict measure.
- Only answerable questions were used, so abstention on questions with no answer was not tested.
- Latency was measured on shared Colab hardware, and generation latency excludes retrieval.

## Not done

- **Abstain gate:** refuse to answer when the best re-ranker score is below a threshold, tested against SQuAD v2's unanswerable questions. Deferred.
- Finish the chunking sweep, and compare sizes at a fixed context budget.
- Find out why `bge-reranker-base` underperformed.
- Citation fixes: short numeric ids mapped back in code, and a check that the cited chunk contains the answer text.
- Re-run generation on more questions.

## Run it

Open `notebook.ipynb` in Google Colab with a T4 runtime and run the cells in order. The first generation run downloads about 8 GB of model weights, and the generation stage took about 7 minutes for 3 modes of 100 questions. Outputs: `project2_reranker.csv`, `project2_generation.csv` and `project2_generations.jsonl` (per-question outputs, reused as input for the evals project).

## Next

03 · Evals: turn these metrics into a regression suite that runs on every change.

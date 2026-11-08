# 03 · Evals: a regression gate for a RAG pipeline

`ragevals` scores the output of a RAG run, stores the scores as a baseline, and fails the build when a later run gets worse. It does not run retrieval or generation itself. It takes the recorded outputs of [Project 2](../02-rag-pipeline) and answers one question: *did this change make the pipeline worse?*

It is a small Python package with no runtime dependencies, 42 unit tests, and a GitHub Actions workflow.

## Why

Project 2 produced numbers (EM 0.60, F1 0.792 for the best pipeline), but numbers in a write-up can't stop a bad change. Any edit to the prompt, chunking, retriever or model can quietly lower quality, and nobody notices until much later. This project turns those numbers into a check that runs on every pull request.

## Quick start (WSL / Linux)

```bash
cd 03-evals
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 42 tests

ragevals summarize data/project2_generations.jsonl
ragevals baseline  data/project2_generations.jsonl --out baselines/new.json
ragevals check     data/project2_generations.jsonl --baseline baselines/project2.json --tolerance 0.002
```

`check` exits 0 when nothing regressed, 1 on a regression or a missing mode, and 2 on a bad input file.

## How it works

Each line of the input `.jsonl` is one question for one pipeline ("mode"): the model's raw output, the chunk ids it was shown (`provided`), the correct chunk ids (`gold`), and the accepted answers (`golds`), both taken from SQuAD's labels.

1. **Parse** (`generation.py`). Split raw output into the answer, the cited chunk ids and an abstention flag. Anything shaped like a chunk id counts as a citation attempt, so a garbled id such as `d125#36#3` is counted and later judged invalid, not silently dropped.
2. **Score** (`generation.py`, `metrics.py`). Per answer:
   - `em`, `f1`: SQuAD-style exact match and token-overlap F1, best over all accepted answers.
   - `hit`: a correct chunk was among the chunks the model was shown.
   - `cite_valid`: every cited id was among the chunks shown (no invented ids).
   - `cite_correct`: a valid cited chunk contains the answer.
3. **Summarize** (`report.py`). Average per mode. `F1_if_hit` averages only questions where retrieval found a correct chunk, which separates generation quality from retrieval quality.
4. **Compare** (`regression.py`). Check each gated metric against the baseline:
   - Quality metrics (EM, F1, retrieval_hit, F1_if_hit, cite_valid, cite_correct) fail if they **drop** by more than the absolute tolerance (default 0.05).
   - Cost metrics (prompt_tokens, p50_gen_s) fail if they **rise** by more than the relative tolerance (default 25%).
   - Improvements never fail. Metrics with no single right direction, such as `idk_rate`, are reported but not gated.

## Results

**Reproducibility.** I regenerated the Project 2 outputs from scratch in a fresh Colab session and ran `check` against the baseline from the write-up. Every quality and token metric matched to three decimals (for example, hybrid + MiniLM: EM 0.600, F1 0.792, cite_correct 0.700), so greedy decoding is deterministic here and the saved numbers are real.

**Latency is noisy.** `p50_gen_s` differed between the two runs by +1.3% (hybrid top-5), +6.7% (hybrid + MiniLM) and -7% (closed-book), with identical code and outputs. A 1% relative tolerance produced false failures, so latency uses the 25% default while quality stays at 0.002. CI uses the same settings.

**The gate fires.** To verify it, I made a synthetic bad run by breaking every third answered record in the retrieval modes (wrong answer, broken citation). The check reported 10 regressions and exited 1:

```
hybrid + MiniLM top-5 EM            0.600 -> 0.420   regression
hybrid + MiniLM top-5 F1            0.792 -> 0.514   regression
hybrid + MiniLM top-5 retrieval_hit 0.950 -> 0.950   ok
hybrid + MiniLM top-5 cite_correct  0.700 -> 0.450   regression
FAILED: 10 regression(s)
```

`retrieval_hit` and the token counts stayed `ok`, so the failure points at generation and citations, not retrieval. This degradation is simulated. It shows the gate works but says nothing about what a real change would do.

## CI

`.github/workflows/03-evals.yml` runs on pull requests and pushes to `main` that touch `03-evals/`. It installs the package, runs `pytest`, and, if `data/project2_generations.jsonl` is present, runs `ragevals check` against the baseline. A non-zero exit fails the run.

## Limitations

- **It scores recorded outputs.** The records hold precomputed `em`, `f1`, `hit` and citation flags, and `check` averages them. So the CI step verifies the data file against the baseline, and the unit tests are what guard the metric code. Re-scoring from `raw`, `golds` and `gold` (all saved now) would let CI catch metric-code changes too.
- **Fixed tolerances, small sample.** With 100 questions the standard error on F1 is about 4 points, so a fixed threshold is a starting point. Because the questions and greedy decoding are fixed, differences come from the change itself, but a few flipped answers can still move F1 by several points.
- **Synthetic failure demo.** See above. A real regression, such as a weaker prompt or smaller top-k, has not been run through the gate yet.
- **Pipeline runs happen in Colab.** This tool checks outputs. It does not produce fresh ones.

## Next

- Re-score from raw output so CI covers the metric code.
- Paired bootstrap, so tolerances come from measured noise instead of a fixed number.
- Run a real regression (smaller top-k or weaker prompt) through the gate.

## Layout

```
src/ragevals/
  metrics.py      EM, F1, recall@k, reciprocal rank
  generation.py   parse a cited answer, score it
  report.py       load JSONL, summarize per mode
  regression.py   compare to a baseline, produce findings
  cli.py          summarize | baseline | check
tests/            42 tests
baselines/        stored baselines
data/             recorded outputs
```
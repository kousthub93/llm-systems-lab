# 03 · Evals: regression checks for a RAG pipeline

`ragevals` turns the metrics from Project 2 into a small tool that scores a run, stores a baseline, and fails when a later run gets worse. It has no runtime dependencies, so it runs anywhere, including CI.

## Set up (WSL / Linux)

```bash
cd /home/kb/code/llm-systems-lab/03-evals
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

If `python3 -m venv` fails on Ubuntu: `sudo apt install python3-venv python3-pip`.

## Use it

```bash
ragevals summarize data/project2_generations.jsonl                      # per-mode metrics
ragevals baseline  data/project2_generations.jsonl --out baselines/new.json
ragevals check     data/project2_generations.jsonl --baseline baselines/project2.json
```

`check` exits 0 when nothing regressed, 1 on a regression or a missing mode, and 2 on a bad file.

**First real check.** Copy `project2_generations.jsonl` from Project 2 into `data/`, then run:

```bash
ragevals check data/project2_generations.jsonl --baseline baselines/project2.json --tolerance 0.002 --rel-tolerance 0.01
```

Every row should say `ok`. `baselines/project2.json` holds the numbers recorded in the Project 2 write-up, so a mismatch means the saved outputs and the write-up disagree (for example, the citation parser version). That is worth finding out.

## How the check works

- Quality metrics (EM, F1, retrieval_hit, F1_if_hit, cite_valid, cite_correct) fail if they **drop** by more than the absolute tolerance (default 0.05).
- Cost metrics (prompt_tokens, p50_gen_s) fail if they **rise** by more than the relative tolerance (default 25%).
- Improvements never fail. Metrics with no single right direction (such as idk_rate) are reported but not gated.
- Per-metric overrides go in the baseline file under a `"tolerances"` key.

## Layout

```
src/ragevals/
  metrics.py      EM, F1, recall@k, reciprocal rank
  generation.py   parse a cited answer, score it (validity, correctness, abstention)
  report.py       load JSONL records, summarize per mode
  regression.py   compare to a baseline, produce findings
  cli.py          summarize | baseline | check
tests/            pytest suite
baselines/        stored baselines
data/             run outputs (not committed by default)
```

## Limitations

- **Noise.** With 100 questions the standard error on F1 is about 4 points. Because the questions are the same and decoding is greedy, differences come from the change itself, but a few flipped answers can still move F1 by several points. Treat the default tolerances as a starting point.
- **Gold labels are not in the saved records.** The records hold precomputed scores (`hit`, `cite_correct`), not the gold chunk ids, so outputs cannot yet be re-scored from raw text. Fix: have the Project 2 notebook also write `gold` per record.
- **It scores recorded outputs.** Running the pipeline to produce fresh outputs still happens in the Colab notebook.

## Next

- Store gold ids so a run can be re-scored from raw output.
- A paired bootstrap, so tolerances reflect noise instead of a fixed number.
- A fresh-run path: generate outputs in Colab, commit them, and let CI gate the change.

import json

from ragevals.cli import main


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def good_records(make_record):
    return [
        make_record(em=1, f1=1.0, hit=True, has_cite=True, cite_valid=True, cite_correct=True, gen_s=1.0),
        make_record(em=1, f1=1.0, hit=True, has_cite=True, cite_valid=True, cite_correct=True, gen_s=1.0),
        make_record(em=0, f1=0.5, hit=True, has_cite=True, cite_valid=True, cite_correct=True, gen_s=2.0),
        make_record(em=0, f1=0.0, hit=False, has_cite=True, cite_valid=False, cite_correct=False, gen_s=3.0),
    ]


def test_summarize_prints_each_mode(tmp_path, make_record, capsys):
    run = tmp_path / "run.jsonl"
    write_jsonl(run, good_records(make_record))
    assert main(["summarize", str(run)]) == 0
    out = capsys.readouterr().out
    assert "hybrid + MiniLM top-5" in out and "0.625" in out  # F1 = (1 + 1 + 0.5 + 0) / 4


def test_baseline_then_check_passes_on_the_same_run(tmp_path, make_record, capsys):
    run, base = tmp_path / "run.jsonl", tmp_path / "out" / "base.json"
    write_jsonl(run, good_records(make_record))
    assert main(["baseline", str(run), "--out", str(base)]) == 0
    assert base.exists()  # parent directory was created
    assert main(["check", str(run), "--baseline", str(base), "--tolerance", "0.001"]) == 0
    assert "OK: no regressions" in capsys.readouterr().out


def test_check_exits_1_when_quality_drops(tmp_path, make_record, capsys):
    run, base = tmp_path / "run.jsonl", tmp_path / "base.json"
    write_jsonl(run, good_records(make_record))
    main(["baseline", str(run), "--out", str(base)])

    degraded = good_records(make_record)
    for r in degraded:
        r["f1"], r["em"] = 0.0, 0.0
    write_jsonl(run, degraded)

    assert main(["check", str(run), "--baseline", str(base)]) == 1
    out = capsys.readouterr().out
    assert "regression" in out and "FAILED" in out


def test_check_exits_1_when_a_baseline_mode_is_missing(tmp_path, make_record):
    run, base = tmp_path / "run.jsonl", tmp_path / "base.json"
    write_jsonl(run, good_records(make_record))
    main(["baseline", str(run), "--out", str(base)])
    write_jsonl(run, [make_record(mode="some other mode", f1=1.0, em=1)])
    assert main(["check", str(run), "--baseline", str(base)]) == 1


def test_bad_input_exits_2(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nope.jsonl"), "--baseline", str(tmp_path / "nope.json")]) == 2
    assert "error:" in capsys.readouterr().err

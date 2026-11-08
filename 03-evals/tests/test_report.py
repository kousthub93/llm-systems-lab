import json

import pytest

from ragevals.report import load_records, summarize, summarize_by_mode


def sample_records(make_record):
    return [
        make_record(em=1, f1=1.0, hit=True, has_cite=True, cite_valid=True, cite_correct=True, gen_s=1.0),  # good
        make_record(em=0, f1=0.5, hit=True, has_cite=True, cite_valid=True, cite_correct=True, gen_s=2.0),  # partial
        make_record(idk=True, hit=False, gen_s=3.0),                                                        # miss, abstained
        make_record(em=0, f1=0.0, hit=False, has_cite=True, cite_valid=False, cite_correct=False, gen_s=4.0),  # miss, bad cite
    ]


def test_summarize_hand_checked_values(make_record):
    s = summarize(sample_records(make_record))
    assert s["n"] == 4
    assert s["EM"] == pytest.approx(0.25)
    assert s["F1"] == pytest.approx(0.375)
    assert s["retrieval_hit"] == pytest.approx(0.5)
    assert s["F1_if_hit"] == pytest.approx(0.75)
    assert s["F1_if_miss"] == pytest.approx(0.0)
    assert s["idk_rate"] == pytest.approx(0.25)
    assert s["idk_if_miss"] == pytest.approx(0.5)
    assert s["cite_present"] == pytest.approx(1.0)       # among the 3 answered
    assert s["cite_valid"] == pytest.approx(2 / 3)       # among the 3 with citations
    assert s["cite_correct"] == pytest.approx(2 / 3)
    assert s["p50_gen_s"] == 3.0
    assert s["prompt_tokens"] == 100


def test_summarize_closed_book_has_no_retrieval_or_citation_fields(make_record):
    s = summarize(sample_records(make_record), closed_book=True)
    assert "retrieval_hit" not in s and "cite_valid" not in s
    assert s["n"] == 4


def test_empty_denominators_are_none(make_record):
    # every question has the correct chunk, so F1_if_miss has no rows
    s = summarize([make_record(em=1, f1=1.0, hit=True, has_cite=True, cite_valid=True, cite_correct=True)])
    assert s["F1_if_miss"] is None and s["idk_if_miss"] is None


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_by_mode_groups_and_detects_closed_book(make_record):
    recs = [
        make_record(mode="closed-book", f1=0.2),
        make_record(mode="closed-book", f1=0.4),
        make_record(mode="hybrid top-5", f1=1.0, em=1, hit=True, has_cite=True, cite_valid=True, cite_correct=True),
    ]
    out = summarize_by_mode(recs)
    assert list(out) == ["closed-book", "hybrid top-5"]
    assert out["closed-book"]["n"] == 2 and "retrieval_hit" not in out["closed-book"]
    assert out["hybrid top-5"]["retrieval_hit"] == 1.0


def test_summarize_by_mode_requires_mode_field(make_record):
    rec = make_record()
    del rec["mode"]
    with pytest.raises(ValueError, match="mode"):
        summarize_by_mode([rec])


def test_load_records_skips_blank_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert load_records(p) == [{"a": 1}, {"a": 2}]


def test_load_records_reports_the_bad_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"a": 1}\n{not json}\n')
    with pytest.raises(ValueError, match=r":2:"):
        load_records(p)

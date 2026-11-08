from ragevals.regression import compare, has_regression

MODE = "hybrid + MiniLM top-5"


def status_of(findings, metric):
    return next(f.status for f in findings if f.metric == metric)


def run(base, cur, **kw):
    return compare({MODE: cur}, {MODE: base}, **kw)


def test_small_drop_within_tolerance_is_ok():
    f = run({"F1": 0.80}, {"F1": 0.77})
    assert status_of(f, "F1") == "ok" and not has_regression(f)


def test_drop_beyond_tolerance_is_a_regression():
    f = run({"F1": 0.80}, {"F1": 0.74})
    assert status_of(f, "F1") == "regression" and has_regression(f)


def test_drop_exactly_at_tolerance_passes():
    assert status_of(run({"F1": 0.80}, {"F1": 0.75}), "F1") == "ok"


def test_improvement_never_fails():
    assert status_of(run({"F1": 0.60}, {"F1": 0.95}), "F1") == "ok"


def test_cost_metrics_fail_when_they_rise_too_much():
    assert status_of(run({"p50_gen_s": 2.0}, {"p50_gen_s": 2.6}), "p50_gen_s") == "regression"  # +30% > 25%
    assert status_of(run({"p50_gen_s": 2.0}, {"p50_gen_s": 2.4}), "p50_gen_s") == "ok"          # +20%
    assert status_of(run({"prompt_tokens": 900}, {"prompt_tokens": 600}), "prompt_tokens") == "ok"  # cheaper is fine


def test_none_values_are_skipped_not_failed():
    f = run({"F1_if_hit": 0.8}, {"F1_if_hit": None})
    assert status_of(f, "F1_if_hit") == "skipped" and not has_regression(f)


def test_missing_mode_fails():
    f = compare({}, {MODE: {"F1": 0.8}})
    assert f[0].status == "missing" and has_regression(f)


def test_per_metric_override_and_cli_tolerance():
    base, cur = {"F1": 0.80}, {"F1": 0.74}
    assert status_of(run(base, cur, overrides={"F1": 0.10}), "F1") == "ok"
    assert status_of(run(base, cur, abs_tol=0.10), "F1") == "ok"


def test_ungated_metrics_are_ignored():
    # idk_rate is informational: there is no single right direction for it
    f = run({"idk_rate": 0.0}, {"idk_rate": 0.5})
    assert f == [] and not has_regression(f)

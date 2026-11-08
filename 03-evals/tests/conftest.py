import pytest


@pytest.fixture
def make_record():
    """Build a per-question record with sensible defaults; override any field."""

    def _make(**overrides):
        rec = dict(
            mode="hybrid + MiniLM top-5",
            em=0.0, f1=0.0, idk=False, hit=False,
            has_cite=False, cite_valid=False, cite_correct=False,
            prompt_tokens=100, gen_s=1.0,
        )
        rec.update(overrides)
        return rec

    return _make

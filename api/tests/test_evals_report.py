import pytest

from evals.agreement import AgreementMetrics
from evals.judge import JudgedMetrics
from evals.metrics import DeterministicMetrics
from evals.report import _fmt, render_report


def _det(**kw):
    base = {
        "topics_evaluated": 5, "schema_valid_rate": 1.0, "grounding_rate": 0.93,
        "ungrounded_claim_rate": 0.07, "mean_claims_per_topic": 3.2,
        "mean_entities_per_topic": 4.1,
    }
    base.update(kw)
    return DeterministicMetrics(**base)


def _agree(**kw):
    base = {
        "labelled_coverage": 0, "total_labels": 0, "entity_precision": None,
        "entity_recall": None, "entity_f1": None, "stance_accuracy": None,
        "stance_kappa": None, "unavailable_reason": "no reviewed labels",
    }
    base.update(kw)
    return AgreementMetrics(**base)


def _meta():
    return {"model": "openai/gpt-oss-20b", "prompt_version": "1", "generated": "2026-07-31"}


def test_report_contains_the_deterministic_numbers():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None), _meta())
    assert "0.93" in out or "93" in out
    assert "Deterministic" in out


def test_unavailable_metrics_say_so_rather_than_printing_zero():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None), _meta())
    assert "unavailable" in out.lower()
    assert "no reviewed labels" in out
    assert "0.00" not in out.split("Agreement")[1].split("Judged")[0]


def test_report_records_provenance_metadata():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None), _meta())
    assert "gpt-oss-20b" in out
    assert "prompt_version" in out.lower() or "prompt version" in out.lower()


def test_agreement_section_states_label_coverage():
    out = render_report(_det(), _agree(labelled_coverage=3, total_labels=5,
                                        entity_f1=0.8, entity_precision=0.75,
                                        entity_recall=0.86, stance_accuracy=0.66,
                                        stance_kappa=0.4, unavailable_reason=None),
                        JudgedMetrics(0, None, None, None), _meta())
    assert (
        "Backed by 3 human-reviewed labels out of 5 in the golden set." in out
    ), "how many labels actually back the numbers is the whole caveat"


def test_judged_section_is_separated_from_deterministic():
    out = render_report(_det(), _agree(), JudgedMetrics(4, 4.2, 4.8, 4.0), _meta())
    assert out.index("Deterministic") < out.index("Judged")
    assert "4.2" in out


def test_skipped_corpus_records_are_stated_in_the_header():
    """`topics evaluated: 4` from a five-record corpus is not something a
    reader should have to reconstruct from the stderr of a run they missed."""
    meta = _meta() | {"skipped_records": 2}
    out = render_report(_det(topics_evaluated=3), _agree(),
                        JudgedMetrics(0, None, None, None), meta)
    assert "Corpus records skipped for having no articles: 2" in out
    assert "Topics evaluated: 3" in out


def test_a_run_that_skipped_nothing_says_nothing():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None),
                        _meta() | {"skipped_records": 0})
    assert "skipped" not in out.lower(), "'0 skipped' is noise, not disclosure"


def test_a_caller_that_omits_the_skipped_count_still_renders():
    out = render_report(_det(), _agree(), JudgedMetrics(0, None, None, None), _meta())
    assert "skipped" not in out.lower()
    assert "Topics evaluated" in out


def _row(out: str, label: str) -> str:
    """The value cell of the table row named ``label``."""
    matches = [line for line in out.splitlines() if line.startswith(f"| {label} |")]
    assert len(matches) == 1, f"expected exactly one row labelled {label!r}, got {matches}"
    return matches[0].split("|")[2].strip()


def _distinct_render() -> str:
    """Every metric given a value no other metric shares, so a row reading the
    wrong field cannot pass by coincidence."""
    return render_report(
        _det(schema_valid_rate=0.11, grounding_rate=0.22, ungrounded_claim_rate=0.33,
             mean_claims_per_topic=4.4, mean_entities_per_topic=5.5),
        _agree(labelled_coverage=3, total_labels=5, entity_precision=0.61,
               entity_recall=0.72, entity_f1=0.83, stance_accuracy=0.94,
               stance_kappa=0.15, unavailable_reason=None),
        JudgedMetrics(6, 1.1, 2.2, 3.3),
        _meta(),
    )


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("Schema validity rate", "0.11"),
        ("Quote grounding rate", "0.22"),
        ("Ungrounded claim rate", "0.33"),
        ("Mean claims per topic", "4.4"),
        ("Mean entities per topic", "5.5"),
        ("Entity precision", "0.61"),
        ("Entity recall", "0.72"),
        ("Entity F1", "0.83"),
        ("Stance accuracy", "0.94"),
        ("Stance Cohen's kappa", "0.15"),
        ("Coverage", "1.1"),
        ("Neutrality", "2.2"),
        ("Coherence", "3.3"),
    ],
)
def test_each_row_renders_its_own_metric(label, value):
    """A row wired to the neighbouring field — kappa printing stance accuracy,
    neutrality printing coherence — is exactly the misreading the surrounding
    prose warns readers about."""
    assert _row(_distinct_render(), label) == value


def test_fmt_reports_a_missing_value_as_the_word_unavailable():
    assert _fmt(None) == "unavailable"
    assert _fmt(None, 1) == "unavailable"


def test_fmt_still_prints_a_measured_zero():
    assert _fmt(0.0) == "0.00"
    assert _fmt(0.0, 1) == "0.0"


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("Schema validity rate", {"det": {"schema_valid_rate": None}}),
        ("Quote grounding rate", {"det": {"grounding_rate": None}}),
        ("Mean claims per topic", {"det": {"mean_claims_per_topic": None}}),
        ("Entity F1", {"agree": {"entity_f1": None}}),
        ("Stance Cohen's kappa", {"agree": {"stance_kappa": None}}),
    ],
)
def test_a_metric_that_cannot_be_computed_renders_unavailable_not_zero(label, kwargs):
    """The harness's headline guarantee: no number is invented for missing
    data, and 0.00 never stands in for "we could not say"."""
    agree = {"labelled_coverage": 1, "total_labels": 1, "entity_precision": 0.61,
             "entity_recall": 0.72, "entity_f1": 0.83, "stance_accuracy": 0.94,
             "stance_kappa": 0.15, "unavailable_reason": None}
    agree.update(kwargs.get("agree", {}))
    out = render_report(_det(**kwargs.get("det", {})), _agree(**agree),
                        JudgedMetrics(6, 1.1, 2.2, 3.3), _meta())
    assert _row(out, label) == "unavailable"
    assert "0.00" not in _row(out, label)


def test_a_missing_judged_mean_renders_unavailable_not_zero():
    out = render_report(_det(), _agree(), JudgedMetrics(6, 1.1, None, 3.3), _meta())
    assert _row(out, "Neutrality") == "unavailable"
    assert _row(out, "Coverage") == "1.1", "one missing axis must not blank the others"

from evals.agreement import AgreementMetrics
from evals.judge import JudgedMetrics
from evals.metrics import DeterministicMetrics
from evals.report import render_report


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
    assert "3" in out and "5" in out


def test_judged_section_is_separated_from_deterministic():
    out = render_report(_det(), _agree(), JudgedMetrics(4, 4.2, 4.8, 4.0), _meta())
    assert out.index("Deterministic") < out.index("Judged")
    assert "4.2" in out

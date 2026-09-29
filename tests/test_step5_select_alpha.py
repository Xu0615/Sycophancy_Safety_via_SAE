import pytest

from src.step5_select_alpha import (
    Candidate,
    _parse_beta_from_run_name,
    select_candidates,
)


def _candidate(beta, syco, *, split=None, anomaly=0.0, eligible=True):
    return Candidate(
        run_name=f"run_{beta}",
        beta=float(beta),
        sycophantic_pct=float(syco),
        objective_pct=100.0 - float(syco),
        repetitive_pct=float(anomaly),
        uncertain_pct=0.0,
        anomaly_pct=float(anomaly),
        selected_split_sycophantic_pct=float(split if split is not None else syco),
        selected_split_objective_pct=100.0 - float(split if split is not None else syco),
        selected_split_repetitive_pct=float(anomaly),
        selected_split_uncertain_pct=0.0,
        selected_split_anomaly_pct=float(anomaly),
        selected_split_n=200,
        selected_split_ids_sha256="abc",
        train_dir="/tmp/train",
        eval_dir="/tmp/eval",
        syco_summary="/tmp/summary.md",
        eligible=eligible,
        exclusion_reasons=[] if eligible else ["quality_gate"],
    )


def test_alpha_parser_accepts_historical_positive_and_negative_names():
    assert _parse_beta_from_run_name("x_alpha30_full") == 30.0
    assert _parse_beta_from_run_name("x_alpha_neg1_full") == -1.0
    assert _parse_beta_from_run_name("x_alpha_neg0p5_full") == -0.5


def test_endpoint_selection_is_independent_across_signs():
    selected = select_candidates(
        [
            _candidate(1, 70),
            _candidate(30, 15),
            _candidate(-1, 95),
            _candidate(-30, 25),
        ],
        min_syco_separation_pp=20,
    )

    assert selected["treatment"].beta == 30
    assert selected["reverse"].beta == -1


def test_endpoint_selection_uses_selection_split_and_quality_gate():
    selected = select_candidates(
        [
            _candidate(5, 30, split=40),
            _candidate(30, 16, split=20),
            _candidate(-1, 94, split=90),
            _candidate(-5, 98, split=99, eligible=False),
        ],
        min_syco_separation_pp=20,
    )
    assert selected["treatment"].beta == 30
    assert selected["reverse"].beta == -1


def test_endpoint_selection_rejects_too_little_syco_separation():
    with pytest.raises(ValueError, match="enough held-out sycophancy separation"):
        select_candidates(
            [_candidate(1, 70), _candidate(-1, 75)],
            min_syco_separation_pp=10,
        )

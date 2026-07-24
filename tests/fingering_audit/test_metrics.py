from __future__ import annotations

import pandas as pd
import pytest

from fingering_audit.evaluation.metrics import compute_metrics, per_finger_metrics
from fingering_audit.evaluation.bootstrap import clustered_intervals


def _labels() -> pd.DataFrame:
    fingers = ["L1", "L1", "L2", "L2", "R1", "R1", "R2", "R2", "R2", "R2"]
    errors = [True, False, True, False, True, False, True, False, False, False]
    return pd.DataFrame(
        {
            "note_id": [f"n{i}" for i in range(10)],
            "recording_id": ["a"] * 5 + ["b"] * 5,
            "gt_finger_id": fingers,
            "exact_error": errors,
            "hand_error": [True, False, False, False, True, False, False, False, False, False],
            "within_hand_finger_error": [
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
        }
    )


def test_compute_metrics_matches_hand_calculation():
    labels = _labels()
    selected = pd.Series([True, True, True, False, False, False, True, False, False, False])
    metric = compute_metrics(selected, labels, set_id="fixture")
    assert metric.values["hard_count"] == 4
    assert metric.values["error_count"] == 4
    assert metric.values["selected_errors"] == 3
    assert metric.values["error_recall"] == pytest.approx(0.75)
    assert metric.values["precision"] == pytest.approx(0.75)
    assert metric.values["correct_sieve_rate"] == pytest.approx(1 / 6)
    assert metric.values["enrichment"] == pytest.approx(1.875)


def test_per_finger_materializes_all_ten_fingers():
    labels = _labels()
    selected = pd.Series([True, True, True, False, False, False, True, False, False, False])
    result = per_finger_metrics(selected, labels, set_id="fixture")
    assert result["finger_id"].tolist() == [
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ]
    l1 = result.set_index("finger_id").loc["L1"]
    assert l1["gt_notes"] == 2
    assert l1["errors"] == 1
    assert l1["selected_errors"] == 1
    assert l1["error_recall"] == 1
    assert pd.isna(result.set_index("finger_id").loc["L3", "error_recall"])


def test_clustered_intervals_are_reproducible_and_recording_based():
    labels = _labels()
    selected = pd.Series([True, True, True, False, False, False, True, False, False, False])
    first = clustered_intervals(selected, labels, seed=7, replicates=500)
    second = clustered_intervals(selected, labels, seed=7, replicates=500)
    assert first == second
    assert first["cluster_count"] == 2
    assert first["replicates"] == 500
    assert 0 <= first["error_recall_ci_low"] <= first["error_recall_ci_high"] <= 1

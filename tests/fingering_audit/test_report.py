from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fingering_audit.study import (
    NOINFO_VARIANTS,
    StudyData,
    _combined_sets,
    _oof_noinfo_tail,
    summarize_study,
)


@pytest.fixture
def study():
    notes = pd.DataFrame(
        {
            "note_id": [f"n{index}" for index in range(6)],
            "recording_id": ["a", "a", "a", "b", "b", "b"],
            "pred_hand": ["R", "R", "R", "L", "L", pd.NA],
            "pred_finger": pd.array([1, 2, 3, 1, 2, pd.NA], dtype="Int64"),
            "pred_finger_id": ["R1", "R2", "R3", "L1", "L2", pd.NA],
        }
    )
    labels = notes.assign(
        gt_finger_id=["R1", "R2", "R3", "L1", "L2", "L3"],
        exact_error=[False, True, False, True, False, True],
        hand_error=[False, False, False, False, False, True],
        within_hand_finger_error=[False, True, False, True, False, False],
    )
    physical = pd.Series([True, False, False, False, False, False])
    integrity = pd.Series([False, False, False, False, False, True])
    queue_full = {
        "physical_candidate_diagnostic": physical.copy(),
        "physical_must_alert": physical,
        "data_integrity_must_resolve": integrity,
    }
    for index, variant in enumerate(NOINFO_VARIANTS):
        mask = pd.Series(False, index=notes.index)
        mask.loc[1 + index % 4] = True
        queue_full[variant] = mask
    queue_gt = {name: mask.copy() for name, mask in queue_full.items()}
    base_full = {"base": pd.Series([False, False, True, False, False, False])}
    base_gt = {name: mask.copy() for name, mask in base_full.items()}
    selections_full = {**base_full, **_combined_sets(base_full, queue_full)}
    selections_gt = {**base_gt, **_combined_sets(base_gt, queue_gt)}
    metadata = pd.DataFrame(
        [
            {
                "set_id": set_id,
                "strategy": "fixture",
                "evidence_grade": "fixture",
                "threshold_summary": "fixture",
            }
            for set_id in selections_full
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "calibration": "fixed",
                "variant": variant,
                "min_run": min_run,
                "radius": radius,
            }
            for variant, (min_run, radius) in NOINFO_VARIANTS.items()
        ]
    )
    return StudyData(
        notes=notes,
        labels=labels,
        features=pd.DataFrame(index=notes.index),
        selections_full=selections_full,
        selections_gt=selections_gt,
        set_metadata=metadata,
        fold_thresholds=pd.DataFrame(),
        queue_masks_full=queue_full,
        queue_masks_gt=queue_gt,
        noinfo_sensitivity=sensitivity,
    )


def test_each_combined_set_contains_its_mandatory_masks(study):
    physical = study.queue_masks_full["physical_must_alert"]
    integrity = study.queue_masks_full["data_integrity_must_resolve"]
    assert study.selections_full["base"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    for set_id, selected in study.selections_full.items():
        if "__ni_" not in set_id:
            continue
        variant = set_id.split("__", 1)[1]
        assert (physical <= selected).all()
        assert (study.queue_masks_full[variant] <= selected).all()
        assert not (selected & integrity).any()


def test_noinfo_table_has_nine_fixed_rows_and_finger_outputs(study):
    tables = summarize_study(study, "fixture", seed=7)
    fixed = tables["noinfo_sensitivity"].query("calibration == 'fixed'")
    assert len(fixed) == 9
    assert set(fixed["min_run"]) == {2, 3, 5}
    assert set(fixed["radius"]) == {1, 2, 4}
    assert {
        "hard_count",
        "hard_percentage_all_notes",
        "gt_error_recall",
        "assigned_gt_error_recall",
        "gt_precision",
        "error_enrichment",
        "incremental_count_beyond_physical",
        "incremental_errors_beyond_physical",
    } <= set(fixed)
    combined = {set_id for set_id in study.selections_full if "__ni_" in set_id}
    assert combined <= set(tables["per_finger"]["set_id"])


def test_oof_noinfo_tail_uses_other_recordings_and_nonzero_scores_only():
    notes = pd.DataFrame(
        {
            "note_id": ["a0", "a1", "b0", "c0"],
            "recording_id": ["a", "a", "b", "c"],
            "pred_hand": ["R"] * 4,
            "pred_finger": [1] * 4,
        }
    )
    labels = notes[["note_id", "recording_id"]].copy()
    full, gt, thresholds = _oof_noinfo_tail(
        notes,
        labels,
        pd.Series([0.0, 0.9, 0.2, 0.8]),
        quantile=0.5,
    )
    by_fold = thresholds.set_index("held_out_recording")
    assert by_fold.loc["a", "threshold"] == pytest.approx(0.5)
    assert by_fold.loc["b", "threshold"] == pytest.approx(0.85)
    assert by_fold.loc["c", "threshold"] == pytest.approx(0.55)
    assert gt.tolist() == [False, True, False, True]
    assert full.tolist() == [False, True, False, True]


def test_oof_noinfo_tail_fails_closed_without_nonzero_training_scores():
    notes = pd.DataFrame(
        {
            "note_id": ["a0", "b0"],
            "recording_id": ["a", "b"],
            "pred_hand": ["R", "L"],
            "pred_finger": [1, 1],
        }
    )
    labels = notes[["note_id", "recording_id"]].copy()
    full, gt, thresholds = _oof_noinfo_tail(
        notes, labels, pd.Series([0.0, np.nan]), quantile=0.995
    )
    assert not full.any()
    assert not gt.any()
    assert np.isinf(thresholds["threshold"]).all()

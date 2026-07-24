from pathlib import Path

import pandas as pd

from fingering_audit.canonical import (
    attach_ground_truth,
    load_ground_truth,
    load_pianovam_notes,
)
from fingering_audit.evaluation.labels import label_errors


FIXTURES = Path(__file__).parent / "fixtures"


def test_canonical_loader_normalizes_labels_and_ids():
    notes = load_pianovam_notes(FIXTURES / "pianovam")
    assert list(notes["note_id"]) == [
        "trial_a#0",
        "trial_a#1",
        "trial_a#2",
        "trial_a#3",
    ]
    assert notes.loc[0, "pred_finger_id"] == "L1"
    assert notes.loc[2, "pred_finger_id"] == "R1"
    assert pd.isna(notes.loc[3, "pred_finger_id"])
    assert pd.isna(notes.loc[0, "offset_sec"])


def test_canonical_loader_does_not_iterate_rows(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("row-wise DataFrame iteration is forbidden")

    monkeypatch.setattr(pd.DataFrame, "iterrows", forbidden)
    notes = load_pianovam_notes(FIXTURES / "pianovam")
    assert len(notes) == 4


def test_gt_attachment_and_error_taxonomy():
    notes = load_pianovam_notes(FIXTURES / "pianovam")
    gt = load_ground_truth(FIXTURES / "gt_fixture.py")
    labeled = label_errors(attach_ground_truth(notes, gt))

    assert len(labeled) == 4
    assert not bool(labeled.loc[0, "exact_error"])

    assert bool(labeled.loc[1, "exact_error"])
    assert not bool(labeled.loc[1, "hand_error"])
    assert bool(labeled.loc[1, "within_hand_finger_error"])

    assert bool(labeled.loc[2, "hand_error"])
    assert not bool(labeled.loc[2, "within_hand_finger_error"])

    assert bool(labeled.loc[3, "exact_error"])
    assert bool(labeled.loc[3, "hand_error"])
    assert not bool(labeled.loc[3, "within_hand_finger_error"])
    assert labeled.loc[3, "gt_finger_id"] == "R2"

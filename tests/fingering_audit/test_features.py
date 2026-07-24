from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from fingering_audit.features.context import context_features
from fingering_audit.features.ergonomic import ergonomic_features
from fingering_audit.features.model import disagreement_features, hmm_features
from fingering_audit.features.model import unconstrained_viterbi_extended_fast
from FingeringInterpolation.hmm import constrained_viterbi_extended, load_model


def _notes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": ["piece"] * 8,
            "note_idx": range(8),
            "note_id": [f"piece#{i}" for i in range(8)],
            "onset_sec": [0.0, 0.1, 0.2, 0.2, 0.4, 0.7, 0.7, 1.0],
            "offset_sec": [0.15, 0.3, 0.5, 0.5, 0.6, 0.9, 0.9, np.nan],
            "pitch": [60, 62, 67, 72, 62, 55, 48, 60],
            "pred_hand": ["R", "R", "R", "R", "R", "L", "L", "L"],
            "pred_finger": pd.array([1, 2, 3, 5, 2, 1, 5, 3], dtype="Int64"),
        }
    )


def test_context_features_preserve_rows_and_boundaries():
    original = _notes()
    before = original.copy(deep=True)
    features = context_features(original)
    pd.testing.assert_frame_equal(original, before)
    assert features["note_id"].tolist() == original["note_id"].tolist()
    assert np.isnan(features.loc[0, "prev_ioi_ms"])
    assert features.loc[1, "prev_ioi_ms"] == 100
    assert features.loc[2, "chord_size"] == 2
    assert features.loc[3, "chord_size"] == 2
    assert bool(features.loc[0, "is_black_key"]) is False
    assert bool(features.loc[1, "is_black_key"]) is False


def test_ergonomic_features_are_time_aware_and_hand_aware():
    features = ergonomic_features(_notes())
    assert features.loc[1, "finger_pair"] == "1-2"
    assert features.loc[1, "directed_pair_span"] == 2
    assert bool(features.loc[1, "non_thumb_crossing"]) is False
    assert features.loc[6, "finger_pair"] == "1-5"
    assert features.loc[6, "directed_pair_span"] == 7
    assert bool(features.loc[6, "non_thumb_crossing"]) is False
    assert features.loc[1, "position_change_rate"] == 20


def test_pitch_distance_alone_does_not_create_time_conditioned_boolean():
    notes = _notes()
    features = ergonomic_features(notes)
    assert "time_conditioned_large_position_change" not in features
    assert features.loc[7, "absolute_pitch_change"] == 12
    assert features.loc[7, "prev_ioi_ms"] == pytest.approx(300)


def test_context_scaling_is_linearithmic():
    size = 100_000
    notes = pd.DataFrame(
        {
            "recording_id": np.repeat("scale", size),
            "note_idx": np.arange(size),
            "note_id": [f"scale#{i}" for i in range(size)],
            "onset_sec": np.arange(size) * 0.05,
            "offset_sec": np.arange(size) * 0.05 + 0.04,
            "pitch": 48 + np.arange(size) % 36,
            "pred_hand": np.where(np.arange(size) % 2, "R", "L"),
            "pred_finger": pd.array(1 + np.arange(size) % 5, dtype="Int64"),
        }
    )
    started = time.monotonic()
    result = context_features(notes)
    assert len(result) == size
    assert time.monotonic() - started < 8.0


def test_hmm_and_disagreement_features_are_explicitly_available():
    notes = _notes()
    models = {
        "L": "FingeringInterpolation/models/hmm_L.npz",
        "R": "FingeringInterpolation/models/hmm_R.npz",
    }
    hmm = hmm_features(notes, models)
    assert len(hmm) == len(notes)
    assert hmm["hmm_feature_available"].all()
    assert hmm["hmm_finger"].between(1, 5).all()
    combined = notes.join(hmm.drop(columns=["note_id"]))
    disagreement = disagreement_features(combined)
    assert set(disagreement["agreement_category"]) <= {
        "exact_agreement",
        "within_hand_finger_disagreement",
        "hand_disagreement",
        "unavailable",
    }


def test_fast_extended_viterbi_matches_reference():
    model = load_model("FingeringInterpolation/models/hmm_R.npz")
    pitches = [60, 64, 67, 72, 71, 69, 67]
    onsets = [0.0, 0.15, 0.3, 0.45, 0.55, 0.65, 0.8]
    reference = constrained_viterbi_extended(
        pitches, onsets, [None] * len(pitches), model
    )
    assert unconstrained_viterbi_extended_fast(pitches, onsets, model) == reference

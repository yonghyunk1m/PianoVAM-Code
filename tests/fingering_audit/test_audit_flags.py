import pandas as pd
import pytest

from fingering_audit.features.audit_flags import (
    NOINFO_CONTEXT_RADII,
    NOINFO_RUN_LENGTHS,
    compute_audit_flags,
    local_missingness_features,
    noinfo_context_mask,
)


def fixture_notes(rows):
    records = []
    for index, values in enumerate(rows):
        records.append(
            {
                "recording_id": "r",
                "note_id": f"r#{index}",
                "note_idx": index,
                "compound_fingering": False,
                **values,
            }
        )
    return pd.DataFrame.from_records(records)


def noinfo_fixture(run_length):
    assert run_length in {3, 5}
    total = run_length + 4
    return pd.DataFrame(
        {
            "recording_id": ["r"] * total,
            "note_id": [f"r#{i}" for i in range(total)],
            "note_idx": range(total),
            "onset_sec": [float(i) for i in range(total)],
            "offset_sec": [i + 0.5 for i in range(total)],
            "pitch": [60] * total,
            "pred_hand": ["R"] + [None] * run_length + ["L", "R", "L"],
            "pred_finger": pd.array(
                [1] + [None] * run_length + [2, 3, 4], dtype="Int64"
            ),
            "compound_fingering": False,
        }
    )


def test_noinfo_context_uses_recording_order_and_only_selects_assigned():
    frame = noinfo_fixture(run_length=3)
    selected = noinfo_context_mask(frame, min_run=3, radius=2)
    assert selected.tolist() == [
        True,
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_noinfo_context_radius_skips_assigned_integrity_rows():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": 0.4,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 1,
            },
            {
                "onset_sec": 0.1,
                "offset_sec": float("inf"),
                "pitch": 62,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 0.2,
                "offset_sec": 0.3,
                "pitch": 64,
                "pred_hand": None,
                "pred_finger": None,
            },
            {
                "onset_sec": 0.3,
                "offset_sec": 0.4,
                "pitch": 65,
                "pred_hand": None,
                "pred_finger": None,
            },
            {
                "onset_sec": 0.4,
                "offset_sec": 0.8,
                "pitch": 67,
                "pred_hand": "R",
                "pred_finger": 3,
            },
        ]
    )

    selected = noinfo_context_mask(frame, min_run=2, radius=1)

    assert selected.tolist() == [True, False, False, False, True]


def test_noinfo_grid_is_monotone():
    frame = noinfo_fixture(run_length=5)
    broad = noinfo_context_mask(frame, min_run=2, radius=4)
    strict = noinfo_context_mask(frame, min_run=5, radius=1)
    assert (strict <= broad).all()


def test_noinfo_context_has_exact_grid_and_rejects_unknown_variants():
    assert NOINFO_RUN_LENGTHS == (2, 3, 5)
    assert NOINFO_CONTEXT_RADII == (1, 2, 4)
    frame = noinfo_fixture(run_length=3)
    with pytest.raises(ValueError, match="unsupported min_run: 4"):
        noinfo_context_mask(frame, min_run=4, radius=1)
    with pytest.raises(ValueError, match="unsupported radius: 3"):
        noinfo_context_mask(frame, min_run=3, radius=3)
    with pytest.raises(ValueError, match="unsupported sequence: hand"):
        noinfo_context_mask(
            frame, min_run=3, radius=1, sequence="hand"
        )


def test_noinfo_context_preserves_input_alignment_and_does_not_mutate():
    frame = noinfo_fixture(run_length=3).iloc[[4, 0, 2, 1, 3, 6, 5]]
    frame.index = pd.Index(
        ["n4", "n0", "n2", "n1", "n3", "n6", "n5"], name="source"
    )
    original = frame.copy(deep=True)

    selected = noinfo_context_mask(frame, min_run=3, radius=2)

    assert selected.index.equals(frame.index)
    assert selected.tolist() == [True, True, False, False, False, False, True]
    pd.testing.assert_frame_equal(frame, original)


def test_available_hand_context_ignores_rows_without_a_valid_hand():
    frame = noinfo_fixture(run_length=3).iloc[[0, 1, 2, 4]].copy()
    frame["note_idx"] = range(4)
    frame["onset_sec"] = [0.0, 1.0, 2.0, 3.0]
    frame.loc[frame.index[-1], ["pred_hand", "pred_finger"]] = ["R", 2]

    recording = noinfo_context_mask(frame, min_run=2, radius=1)
    available_hand = noinfo_context_mask(
        frame, min_run=2, radius=1, sequence="available_hand"
    )

    assert recording.tolist() == [True, False, False, True]
    assert available_hand.tolist() == [False, False, False, False]


def test_local_missingness_features_have_fixed_windows_and_distances():
    frame = noinfo_fixture(run_length=3)
    frame.index = pd.Index(range(10, 17), name="source")
    original = frame.copy(deep=True)

    result = local_missingness_features(frame)

    assert list(result) == [
        "noinfo_fraction_w5",
        "noinfo_fraction_w9",
        "noinfo_fraction_w17",
        "nearest_noinfo_note_distance",
        "nearest_noinfo_time_distance_sec",
    ]
    assert result.index.equals(frame.index)
    assert result["noinfo_fraction_w5"].tolist() == pytest.approx(
        [2 / 3, 3 / 4, 3 / 5, 3 / 5, 2 / 5, 1 / 4, 0]
    )
    assert result["noinfo_fraction_w9"].between(0, 1).all()
    assert result["noinfo_fraction_w17"].between(0, 1).all()
    assert result["nearest_noinfo_note_distance"].tolist() == [
        1,
        0,
        0,
        0,
        1,
        2,
        3,
    ]
    assert result["nearest_noinfo_time_distance_sec"].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    pd.testing.assert_frame_equal(frame, original)


def test_local_missingness_note_and_time_distances_are_independent():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.9,
                "offset_sec": 1.0,
                "pitch": 60,
                "pred_hand": None,
                "pred_finger": None,
            },
            {
                "onset_sec": 0.95,
                "offset_sec": 1.05,
                "pitch": 61,
                "pred_hand": "R",
                "pred_finger": 1,
            },
            {
                "onset_sec": 1.0,
                "offset_sec": 1.1,
                "pitch": 62,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 10.0,
                "offset_sec": 10.1,
                "pitch": 63,
                "pred_hand": None,
                "pred_finger": None,
            },
        ]
    )

    result = local_missingness_features(frame)

    assert result.loc[2, "nearest_noinfo_note_distance"] == 1
    assert result.loc[2, "nearest_noinfo_time_distance_sec"] == pytest.approx(
        0.1
    )


def test_local_missingness_uses_infinite_distances_when_recording_is_complete():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": 0.5,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 1,
            },
            {
                "onset_sec": 1.0,
                "offset_sec": 1.5,
                "pitch": 62,
                "pred_hand": "L",
                "pred_finger": 2,
            },
        ]
    )

    result = local_missingness_features(frame)

    assert result["nearest_noinfo_note_distance"].tolist() == [
        float("inf"),
        float("inf"),
    ]
    assert result["nearest_noinfo_time_distance_sec"].tolist() == [
        float("inf"),
        float("inf"),
    ]


def test_invalid_offset_is_integrity_not_physical():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": None,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 1,
            }
        ]
    )
    flags = compute_audit_flags(frame, {"1-5": 16})
    assert flags.integrity.tolist() == [True]
    assert "missing_offset" in flags.integrity_reasons.iloc[0]
    assert flags.physical_candidate.tolist() == [False]


def test_non_finite_timing_is_integrity_not_physical():
    frame = fixture_notes(
        [
            {
                "onset_sec": float("-inf"),
                "offset_sec": 1.0,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 0.2,
                "offset_sec": 0.8,
                "pitch": 64,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 0.0,
                "offset_sec": float("inf"),
                "pitch": 67,
                "pred_hand": "R",
                "pred_finger": 3,
            },
            {
                "onset_sec": 0.2,
                "offset_sec": 0.8,
                "pitch": 71,
                "pred_hand": "R",
                "pred_finger": 3,
            },
            {
                "onset_sec": float("inf"),
                "offset_sec": float("inf"),
                "pitch": 72,
                "pred_hand": "R",
                "pred_finger": 4,
            },
        ]
    )

    flags = compute_audit_flags(frame)

    assert flags.integrity.tolist() == [True, False, True, False, True]
    assert flags.integrity_reasons.tolist() == [
        ("non_finite_onset",),
        (),
        ("non_finite_offset",),
        (),
        ("non_finite_offset", "non_finite_onset"),
    ]
    assert flags.physical_candidate.tolist() == [False] * 5


def test_fractional_finger_and_pitch_are_integrity_not_physical():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": 1.0,
                "pitch": 60.5,
                "pred_hand": "R",
                "pred_finger": 2.5,
            },
            {
                "onset_sec": 0.2,
                "offset_sec": 0.8,
                "pitch": 64,
                "pred_hand": "R",
                "pred_finger": 2,
            },
        ]
    )

    flags = compute_audit_flags(frame)

    assert flags.integrity.tolist() == [True, False]
    assert flags.integrity_reasons.tolist() == [
        ("non_integral_finger", "non_integral_pitch"),
        (),
    ]
    assert flags.physical_candidate.tolist() == [False, False]


def test_same_finger_overlap_flags_both_notes():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": 1.0,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 0.2,
                "offset_sec": 0.8,
                "pitch": 64,
                "pred_hand": "R",
                "pred_finger": 2,
            },
        ]
    )
    flags = compute_audit_flags(frame)
    assert flags.same_finger_candidate.tolist() == [True, True]


def test_overlap_is_strict_at_one_millisecond():
    frame = fixture_notes(
        [
            {
                "recording_id": "exact",
                "onset_sec": 0.0,
                "offset_sec": 0.501,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "recording_id": "exact",
                "onset_sec": 0.5,
                "offset_sec": 1.0,
                "pitch": 64,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "recording_id": "beyond",
                "onset_sec": 0.0,
                "offset_sec": 0.501001,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "recording_id": "beyond",
                "onset_sec": 0.5,
                "offset_sec": 1.0,
                "pitch": 64,
                "pred_hand": "R",
                "pred_finger": 2,
            },
        ]
    )

    flags = compute_audit_flags(frame)

    assert flags.same_finger_candidate.tolist() == [False, False, True, True]


def test_touching_intervals_and_repeated_pitch_do_not_flag():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": 0.5,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 0.5,
                "offset_sec": 1.0,
                "pitch": 64,
                "pred_hand": "R",
                "pred_finger": 2,
            },
            {
                "onset_sec": 0.1,
                "offset_sec": 0.4,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 2,
            },
        ]
    )
    assert not compute_audit_flags(frame).physical_candidate.any()


def test_span_is_strict_and_checks_non_adjacent_active_notes():
    frame = fixture_notes(
        [
            {
                "onset_sec": 0.0,
                "offset_sec": 1.0,
                "pitch": 48,
                "pred_hand": "R",
                "pred_finger": 1,
            },
            {
                "onset_sec": 0.1,
                "offset_sec": 0.9,
                "pitch": 60,
                "pred_hand": "R",
                "pred_finger": 3,
            },
            {
                "onset_sec": 0.2,
                "offset_sec": 0.8,
                "pitch": 65,
                "pred_hand": "R",
                "pred_finger": 5,
            },
        ]
    )
    flags = compute_audit_flags(
        frame,
        {"1-5": 16, "1-3": 12, "3-5": 7},
    )
    assert flags.span_candidate.tolist() == [True, False, True]

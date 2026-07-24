import pandas as pd

from fingering_audit.features.audit_flags import compute_audit_flags


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

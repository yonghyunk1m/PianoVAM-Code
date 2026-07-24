from __future__ import annotations

from annotate.prepare_review_data import apply_hard_rules_to_notes


AUDIT_FIELDS = {
    "physical_must_alert",
    "physical_reasons",
    "data_integrity_must_resolve",
    "data_integrity_reasons",
    "noinfo_context_alert",
    "noinfo_context_reasons",
}


def test_review_json_separates_audit_categories():
    notes = [
        {
            "global_idx": 0,
            "onset_sec": 0.0,
            "offset_sec": 1.0,
            "pitch": 60,
            "algorithm_hand": "Right",
            "algorithm_finger": 2,
            "algorithm_int": 7,
        },
        {
            "global_idx": 1,
            "onset_sec": 0.1,
            "offset_sec": 0.9,
            "pitch": 64,
            "algorithm_hand": "Right",
            "algorithm_finger": 2,
            "algorithm_int": 7,
        },
    ]

    result = apply_hard_rules_to_notes(
        notes, ["physical_candidate_diagnostic"]
    )

    assert result[0]["physical_reasons"] == [
        "same_finger_simultaneous_keys"
    ]
    assert result[0]["data_integrity_reasons"] == []
    assert {"is_hard", "hard_reasons"} <= set(result[0])


def test_review_json_uses_within_hand_finger_for_physical_validation():
    notes = [
        {
            "global_idx": 0,
            "onset_sec": 0.0,
            "offset_sec": 1.0,
            "pitch": 60,
            "algorithm_hand": "Right",
            "algorithm_finger": 2,
            "algorithm_int": 7,
        },
        {
            "global_idx": 1,
            "onset_sec": 0.1,
            "offset_sec": 0.9,
            "pitch": 64,
            "algorithm_hand": "Right",
            "algorithm_finger": 2,
            "algorithm_int": 7,
        },
    ]

    result = apply_hard_rules_to_notes(
        notes, ["physical_candidate_diagnostic"]
    )

    assert all(
        note["data_integrity_reasons"] == []
        for note in result
    )


def test_missing_offset_is_integrity_not_physical_overlap():
    notes = [
        {
            "global_idx": 0,
            "onset_sec": 0.0,
            "offset_sec": None,
            "pitch": 60,
            "algorithm_hand": "Right",
            "algorithm_finger": 2,
            "algorithm_int": 7,
        },
        {
            "global_idx": 1,
            "onset_sec": 0.1,
            "offset_sec": 0.9,
            "pitch": 64,
            "algorithm_hand": "Right",
            "algorithm_finger": 2,
            "algorithm_int": 7,
        },
    ]

    result = apply_hard_rules_to_notes(
        notes, ["physical_candidate_diagnostic"]
    )

    assert result[0]["data_integrity_must_resolve"] is True
    assert result[0]["data_integrity_reasons"] == ["missing_offset"]
    assert result[0]["physical_must_alert"] is False
    assert result[0]["physical_reasons"] == []
    assert result[1]["physical_reasons"] == []


def test_noinfo_records_are_integrity_and_neighbors_are_context_alerts():
    notes = [
        {
            "global_idx": index,
            "onset_sec": float(index),
            "offset_sec": float(index) + 0.5,
            "pitch": 60 + index,
            "algorithm_hand": "Right" if index not in {2, 3, 4} else None,
            "algorithm_finger": 2 if index not in {2, 3, 4} else None,
            "algorithm_int": 7 if index not in {2, 3, 4} else None,
        }
        for index in range(7)
    ]

    result = apply_hard_rules_to_notes(
        notes, ["noinfo_context_k3_r2"]
    )

    assert [
        note["data_integrity_must_resolve"] for note in result
    ] == [False, False, True, True, True, False, False]
    assert [
        note["noinfo_context_alert"] for note in result
    ] == [True, True, False, False, False, True, True]
    assert all(AUDIT_FIELDS <= set(note) for note in result)

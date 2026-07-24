from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class AuditFlags:
    integrity: pd.Series
    integrity_reasons: pd.Series
    same_finger_candidate: pd.Series
    span_candidate: pd.Series
    physical_candidate: pd.Series
    physical_reasons: pd.Series


def _pair_key(first, second):
    return f"{min(first, second)}-{max(first, second)}"


def compute_audit_flags(
    notes: pd.DataFrame,
    span_boundaries: Mapping[str, int] | None = None,
    *,
    timing_epsilon_sec: float = 0.001,
) -> AuditFlags:
    work = notes.reset_index(drop=True)
    hand = work["pred_hand"].astype("string")
    finger = pd.to_numeric(work["pred_finger"], errors="coerce")
    pitch = pd.to_numeric(work["pitch"], errors="coerce")
    onset = pd.to_numeric(work["onset_sec"], errors="coerce")
    offset = pd.to_numeric(work["offset_sec"], errors="coerce")
    reason_sets = [set() for _ in range(len(work))]
    for index in work.index:
        if hand.loc[index] not in {"L", "R"}:
            reason_sets[index].add("missing_or_invalid_hand")
        if pd.isna(finger.loc[index]) or not 1 <= finger.loc[index] <= 5:
            reason_sets[index].add("missing_or_invalid_finger")
        elif not float(finger.loc[index]).is_integer():
            reason_sets[index].add("non_integral_finger")
        if pd.isna(pitch.loc[index]) or not 0 <= pitch.loc[index] <= 127:
            reason_sets[index].add("missing_or_invalid_pitch")
        elif not float(pitch.loc[index]).is_integer():
            reason_sets[index].add("non_integral_pitch")
        if pd.isna(onset.loc[index]):
            reason_sets[index].add("missing_onset")
        elif not isfinite(onset.loc[index]):
            reason_sets[index].add("non_finite_onset")
        if pd.isna(offset.loc[index]):
            reason_sets[index].add("missing_offset")
        elif not isfinite(offset.loc[index]):
            reason_sets[index].add("non_finite_offset")
        elif (
            not pd.isna(onset.loc[index])
            and isfinite(onset.loc[index])
            and offset.loc[index] < onset.loc[index]
        ):
            reason_sets[index].add("offset_before_onset")
    integrity = pd.Series(
        [bool(value) for value in reason_sets], index=work.index
    )
    same_finger = pd.Series(False, index=work.index)
    span = pd.Series(False, index=work.index)
    physical_reason_sets = [set() for _ in range(len(work))]
    valid = work.loc[~integrity].copy()
    valid["_hand"] = hand.loc[valid.index]
    valid["_finger"] = finger.loc[valid.index].astype(int)
    valid["_pitch"] = pitch.loc[valid.index].astype(int)
    valid["_onset"] = onset.loc[valid.index]
    valid["_offset"] = offset.loc[valid.index]
    for _, group in valid.groupby(["recording_id", "_hand"], sort=False):
        active = []
        ordered = group.sort_values(["_onset", "note_idx"], kind="stable")
        for current in ordered.index:
            current_onset = float(valid.at[current, "_onset"])
            active = [
                earlier
                for earlier in active
                if float(valid.at[earlier, "_offset"])
                > current_onset + timing_epsilon_sec
            ]
            for earlier in active:
                if valid.at[earlier, "_pitch"] == valid.at[current, "_pitch"]:
                    continue
                first = int(valid.at[earlier, "_finger"])
                second = int(valid.at[current, "_finger"])
                simple = not bool(work.at[earlier, "compound_fingering"])
                simple = simple and not bool(
                    work.at[current, "compound_fingering"]
                )
                if simple and first == second:
                    same_finger.loc[[earlier, current]] = True
                    physical_reason_sets[earlier].add(
                        "same_finger_simultaneous_keys"
                    )
                    physical_reason_sets[current].add(
                        "same_finger_simultaneous_keys"
                    )
                boundary = (span_boundaries or {}).get(
                    _pair_key(first, second)
                )
                distance = abs(
                    int(valid.at[earlier, "_pitch"])
                    - int(valid.at[current, "_pitch"])
                )
                if simple and boundary is not None and distance > boundary:
                    span.loc[[earlier, current]] = True
                    physical_reason_sets[earlier].add(
                        "simultaneous_span_beyond_policy"
                    )
                    physical_reason_sets[current].add(
                        "simultaneous_span_beyond_policy"
                    )
            active.append(current)
    integrity_reasons = pd.Series(
        [tuple(sorted(value)) for value in reason_sets], index=work.index
    )
    physical_reasons = pd.Series(
        [tuple(sorted(value)) for value in physical_reason_sets],
        index=work.index,
    )
    return AuditFlags(
        integrity=integrity,
        integrity_reasons=integrity_reasons,
        same_finger_candidate=same_finger,
        span_candidate=span,
        physical_candidate=same_finger | span,
        physical_reasons=physical_reasons,
    )

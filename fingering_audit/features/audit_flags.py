from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd


NOINFO_RUN_LENGTHS = (2, 3, 5)
NOINFO_CONTEXT_RADII = (1, 2, 4)
NOINFO_LOCAL_WINDOWS = (5, 9, 17)


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


def _assigned_fingering(notes: pd.DataFrame) -> pd.Series:
    hand = notes["pred_hand"].astype("string")
    finger = pd.to_numeric(notes["pred_finger"], errors="coerce")
    return hand.isin(["L", "R"]) & finger.between(1, 5).fillna(False)


def noinfo_context_mask(
    notes: pd.DataFrame,
    *,
    min_run: int,
    radius: int,
    sequence: str = "recording",
) -> pd.Series:
    if min_run not in NOINFO_RUN_LENGTHS:
        raise ValueError(f"unsupported min_run: {min_run}")
    if radius not in NOINFO_CONTEXT_RADII:
        raise ValueError(f"unsupported radius: {radius}")
    if sequence not in {"recording", "available_hand"}:
        raise ValueError(f"unsupported sequence: {sequence}")

    work = notes.reset_index(drop=True)
    assigned = _assigned_fingering(work).to_numpy(dtype=bool)
    selected = np.zeros(len(work), dtype=bool)
    if sequence == "available_hand":
        valid_hand = work["pred_hand"].astype("string").isin(["L", "R"])
        groups = work.loc[valid_hand].groupby(
            ["recording_id", "pred_hand"], sort=False, dropna=False
        )
    else:
        groups = work.groupby("recording_id", sort=False, dropna=False)

    for _, group in groups:
        ordered = group.sort_values(
            ["onset_sec", "note_idx"], kind="stable"
        )
        positions = ordered.index.to_numpy(dtype=int)
        missing = ~assigned[positions]
        boundaries = np.diff(
            np.concatenate(([False], missing, [False])).astype(np.int8)
        )
        starts = np.flatnonzero(boundaries == 1)
        ends = np.flatnonzero(boundaries == -1)
        for start, end in zip(starts, ends):
            if end - start < min_run:
                continue
            context = np.concatenate(
                (
                    positions[max(0, start - radius) : start],
                    positions[end : min(len(positions), end + radius)],
                )
            )
            selected[context] = assigned[context]

    return pd.Series(selected, index=notes.index)


def local_missingness_features(notes: pd.DataFrame) -> pd.DataFrame:
    work = notes.reset_index(drop=True)
    missing = ~_assigned_fingering(work).to_numpy(dtype=bool)
    fractions = {
        width: np.empty(len(work), dtype=float)
        for width in NOINFO_LOCAL_WINDOWS
    }
    note_distance = np.full(len(work), np.inf)
    time_distance = np.full(len(work), np.inf)

    for _, group in work.groupby(
        "recording_id", sort=False, dropna=False
    ):
        ordered = group.sort_values(
            ["onset_sec", "note_idx"], kind="stable"
        )
        positions = ordered.index.to_numpy(dtype=int)
        local_missing = missing[positions]
        for width in NOINFO_LOCAL_WINDOWS:
            fractions[width][positions] = (
                pd.Series(local_missing, dtype=float)
                .rolling(width, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

        onset = pd.to_numeric(
            ordered["onset_sec"], errors="coerce"
        ).to_numpy(dtype=float)
        local_note_distance = np.full(len(positions), np.inf)
        local_time_distance = np.full(len(positions), np.inf)
        last_missing = -1
        for index in range(len(positions)):
            if local_missing[index]:
                last_missing = index
                local_note_distance[index] = 0
                local_time_distance[index] = 0
            elif last_missing >= 0:
                local_note_distance[index] = index - last_missing
                delta = abs(onset[index] - onset[last_missing])
                if not np.isnan(delta):
                    local_time_distance[index] = delta
        next_missing = -1
        for index in range(len(positions) - 1, -1, -1):
            if local_missing[index]:
                next_missing = index
            elif next_missing >= 0:
                local_note_distance[index] = min(
                    local_note_distance[index], next_missing - index
                )
                delta = abs(onset[next_missing] - onset[index])
                if not np.isnan(delta):
                    local_time_distance[index] = min(
                        local_time_distance[index], delta
                    )
        note_distance[positions] = local_note_distance
        time_distance[positions] = local_time_distance

    return pd.DataFrame(
        {
            **{
                f"noinfo_fraction_w{width}": fractions[width]
                for width in NOINFO_LOCAL_WINDOWS
            },
            "nearest_noinfo_note_distance": note_distance,
            "nearest_noinfo_time_distance_sec": time_distance,
        },
        index=notes.index,
    )


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

from __future__ import annotations

import numpy as np
import pandas as pd


_BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})


def _window_counts(
    recording: pd.Series, onset: pd.Series, radius_sec: float = 0.5
) -> np.ndarray:
    result = np.zeros(len(onset), dtype=np.int32)
    work = pd.DataFrame(
        {"recording": recording.to_numpy(), "onset": onset.to_numpy()}
    )
    for _, indices in work.groupby("recording", sort=False).indices.items():
        indices = np.asarray(indices)
        values = onset.iloc[indices].to_numpy(dtype=float)
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        left = np.searchsorted(sorted_values, sorted_values - radius_sec, side="left")
        right = np.searchsorted(sorted_values, sorted_values + radius_sec, side="right")
        result[indices[order]] = right - left
    return result


def _active_counts(notes: pd.DataFrame) -> np.ndarray:
    result = np.ones(len(notes), dtype=np.int32)
    for _, indices in notes.groupby("recording_id", sort=False).indices.items():
        indices = np.asarray(indices)
        onsets = notes["onset_sec"].iloc[indices].to_numpy(dtype=float)
        offsets = notes["offset_sec"].iloc[indices].to_numpy(dtype=float)
        offsets = np.where(np.isfinite(offsets), offsets, onsets)
        sorted_onsets = np.sort(onsets)
        sorted_offsets = np.sort(offsets)
        started = np.searchsorted(sorted_onsets, onsets, side="right")
        ended = np.searchsorted(sorted_offsets, onsets, side="right")
        result[indices] = np.maximum(1, started - ended)
    return result


def context_features(notes: pd.DataFrame) -> pd.DataFrame:
    """Return vectorized musical context, one row per canonical note."""
    required = {
        "recording_id",
        "note_id",
        "onset_sec",
        "offset_sec",
        "pitch",
        "pred_hand",
        "pred_finger",
    }
    missing = required - set(notes)
    if missing:
        raise ValueError(f"context feature input missing columns: {sorted(missing)}")

    work = notes.reset_index(drop=True).copy()
    group_columns = ["recording_id", "pred_hand"]
    valid_hand = work["pred_hand"].isin(["L", "R"])
    sequence = work.loc[valid_hand]
    grouped = sequence.groupby(group_columns, sort=False, dropna=False)
    prev_onset = grouped["onset_sec"].shift(1)
    next_onset = grouped["onset_sec"].shift(-1)
    prev_pitch = grouped["pitch"].shift(1)
    next_pitch = grouped["pitch"].shift(-1)
    prev_offset = grouped["offset_sec"].shift(1)

    result = pd.DataFrame(
        {
            "note_id": work["note_id"],
            "recording_id": work["recording_id"],
            "prev_note_id": pd.Series(pd.NA, index=work.index, dtype="string"),
            "next_note_id": pd.Series(pd.NA, index=work.index, dtype="string"),
            "prev_ioi_ms": np.nan,
            "next_ioi_ms": np.nan,
            "pitch_change": np.nan,
            "next_pitch_change": np.nan,
            "absolute_pitch_change": np.nan,
            "duration_ms": (
                (work["offset_sec"] - work["onset_sec"]) * 1000.0
            ).where(work["offset_sec"].notna()),
            "overlaps_previous_same_hand": False,
        }
    )
    result.loc[sequence.index, "prev_note_id"] = grouped["note_id"].shift(1)
    result.loc[sequence.index, "next_note_id"] = grouped["note_id"].shift(-1)
    result.loc[sequence.index, "prev_ioi_ms"] = (
        sequence["onset_sec"] - prev_onset
    ) * 1000.0
    result.loc[sequence.index, "next_ioi_ms"] = (
        next_onset - sequence["onset_sec"]
    ) * 1000.0
    result.loc[sequence.index, "pitch_change"] = sequence["pitch"] - prev_pitch
    result.loc[sequence.index, "next_pitch_change"] = next_pitch - sequence["pitch"]
    result["absolute_pitch_change"] = result["pitch_change"].abs()
    result.loc[sequence.index, "overlaps_previous_same_hand"] = (
        sequence["onset_sec"] < prev_offset
    ).fillna(False)

    chord_key = [work["recording_id"], work["pred_hand"], work["onset_sec"]]
    result["chord_size"] = (
        work.groupby(chord_key, dropna=False)["note_id"].transform("size").astype("int32")
    )
    result["local_note_count_1s"] = _window_counts(
        work["recording_id"], work["onset_sec"]
    )
    result["local_polyphony"] = _active_counts(work)
    result["is_black_key"] = (work["pitch"].astype(int) % 12).isin(
        _BLACK_PITCH_CLASSES
    )
    result["register_octave"] = (work["pitch"].astype(int) // 12) - 1

    repeat = pd.Series(False, index=work.index)
    repeat.loc[sequence.index] = sequence["pitch"].eq(prev_pitch).fillna(False)
    run_boundary = (~repeat).groupby(
        [work["recording_id"], work["pred_hand"]], dropna=False
    ).cumsum()
    result["repeated_pitch_run_length"] = (
        repeat.groupby(
            [work["recording_id"], work["pred_hand"], run_boundary], dropna=False
        )
        .transform("size")
        .where(repeat, 1)
        .astype("int32")
    )
    return result

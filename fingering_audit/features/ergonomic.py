from __future__ import annotations

import numpy as np
import pandas as pd

from .context import context_features


def ergonomic_features(notes: pd.DataFrame) -> pd.DataFrame:
    """Compute raw ergonomic relations; thresholding belongs to the ledger."""
    context = context_features(notes)
    work = notes.reset_index(drop=True)
    valid = work["pred_hand"].isin(["L", "R"]) & work["pred_finger"].notna()
    sequence = work.loc[valid]
    grouped = sequence.groupby(["recording_id", "pred_hand"], sort=False)
    prev_finger = grouped["pred_finger"].shift(1).astype("Float64")
    prev_pitch = grouped["pitch"].shift(1)

    current_finger = sequence["pred_finger"].astype("Float64")
    finger_delta = current_finger - prev_finger
    pitch_delta = sequence["pitch"] - prev_pitch
    hand_axis = sequence["pred_hand"].map({"R": 1.0, "L": -1.0})
    lower_finger = pd.concat([prev_finger, current_finger], axis=1).min(axis=1)
    upper_finger = pd.concat([prev_finger, current_finger], axis=1).max(axis=1)
    finger_pair = (
        lower_finger.astype("Int64").astype("string")
        + "-"
        + upper_finger.astype("Int64").astype("string")
    )
    directed_span = pitch_delta * np.sign(finger_delta.astype(float)) * hand_axis
    non_thumb = (prev_finger.ne(1) & current_finger.ne(1)).fillna(False)
    crossing = (
        (pitch_delta * finger_delta.astype(float) * hand_axis < 0)
        & pitch_delta.ne(0)
        & finger_delta.ne(0)
        & non_thumb
    )

    result = context.copy()
    result["previous_finger"] = pd.array([pd.NA] * len(work), dtype="Int64")
    result["finger_pair"] = pd.Series(pd.NA, index=work.index, dtype="string")
    result["finger_delta"] = np.nan
    result["directed_pair_span"] = np.nan
    result["non_thumb_crossing"] = False
    result.loc[sequence.index, "previous_finger"] = prev_finger.astype("Int64")
    result.loc[sequence.index, "finger_pair"] = finger_pair.where(
        finger_delta.ne(0)
    )
    result.loc[sequence.index, "finger_delta"] = finger_delta.astype(float)
    result.loc[sequence.index, "directed_pair_span"] = directed_span.where(
        finger_delta.ne(0)
    )
    result.loc[sequence.index, "non_thumb_crossing"] = crossing.fillna(False)
    ioi_seconds = result["prev_ioi_ms"] / 1000.0
    result["position_change_rate"] = (
        result["absolute_pitch_change"] / ioi_seconds
    ).where(ioi_seconds > 0)

    previous_same_finger = (
        prev_finger.eq(current_finger) & prev_finger.notna()
    ).reindex(work.index, fill_value=False)
    result["same_finger_overlap_different_pitch"] = (
        result["overlaps_previous_same_hand"]
        & previous_same_finger
        & result["pitch_change"].ne(0)
    )
    return result

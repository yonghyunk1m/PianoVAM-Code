from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from FingeringInterpolation import hmm as hmm_reference
from FingeringInterpolation.hmm import (
    constrained_viterbi,
    constrained_viterbi_extended,
    load_model,
)


def _short_time_matrix(hand: str, ioi: float, pitch_delta: int) -> np.ndarray:
    if ioi >= hmm_reference._SHORT_TIME_S:
        return np.zeros((5, 5), dtype=float)
    previous = np.arange(5)[:, None]
    current = np.arange(5)[None, :]
    direction = (current - previous) * pitch_delta
    wrong = direction < 0 if hand == "R" else direction > 0
    return np.where(wrong, hmm_reference._SHORT_TIME_COST, 0.0)


def unconstrained_viterbi_extended_fast(
    pitches: list[int], onsets: list[float], model: dict
) -> list[int]:
    """Vectorized equivalent of the repository's unconstrained 2nd-order HMM."""
    count = len(pitches)
    if count == 0:
        return []
    if count == 1:
        return [1]
    eps = hmm_reference.EPS
    log_trans = np.log(model["trans"] + eps)
    log_out1 = np.log(model["outProb"] + eps)
    log_out2 = np.log(model["outProb2"] + eps)
    hand = model.get("hand", "R")
    dx, dy = hmm_reference._keypos_interval(pitches[1], pitches[0])
    key = hmm_reference._keypos_idx(dx, dy)
    dp = (
        np.log(model["init"] + eps)
        + hmm_reference._W1 * log_out1[:, :, key]
        + _short_time_matrix(hand, onsets[1] - onsets[0], pitches[1] - pitches[0])
    )
    if count == 2:
        best = np.unravel_index(np.argmax(dp), dp.shape)
        return [int(finger) + 1 for finger in best]
    history: list[np.ndarray] = []
    for note in range(2, count):
        dx1, dy1 = hmm_reference._keypos_interval(
            pitches[note], pitches[note - 1]
        )
        dx2, dy2 = hmm_reference._keypos_interval(
            pitches[note], pitches[note - 2]
        )
        emit1 = hmm_reference._W1 * log_out1[
            :, :, hmm_reference._keypos_idx(dx1, dy1)
        ]
        emit2 = hmm_reference._W2 * log_out2[
            :, :, hmm_reference._keypos_idx(dx2, dy2)
        ]
        short1 = _short_time_matrix(
            hand,
            onsets[note] - onsets[note - 1],
            pitches[note] - pitches[note - 1],
        )
        short2 = _short_time_matrix(
            hand,
            onsets[note] - onsets[note - 2],
            pitches[note] - pitches[note - 2],
        )
        scores = (
            dp[:, :, None]
            + log_trans
            + emit1[None, :, :]
            + emit2[:, None, :]
            + short1[None, :, :]
            + short2[:, None, :]
        )
        backpointer = np.argmax(scores, axis=0).astype(np.int8)
        dp = np.max(scores, axis=0)
        history.append(backpointer)
    best_last = np.unravel_index(np.argmax(dp), dp.shape)
    sequence = [int(best_last[0]), int(best_last[1])]
    for backpointer in reversed(history):
        sequence.insert(0, int(backpointer[sequence[0], sequence[1]]))
    return [finger + 1 for finger in sequence]


def _infer_missing_hands(recording: pd.DataFrame) -> pd.Series:
    hands = recording["pred_hand"].astype("string").copy()
    known_l = recording.loc[hands.eq("L"), "pitch"].to_numpy(dtype=float)
    known_r = recording.loc[hands.eq("R"), "pitch"].to_numpy(dtype=float)
    if len(known_l) and len(known_r):
        left_center = float(np.median(known_l))
        right_center = float(np.median(known_r))
        missing = hands.isna()
        pitch = recording.loc[missing, "pitch"].to_numpy(dtype=float)
        hands.loc[missing] = np.where(
            np.abs(pitch - left_center) <= np.abs(pitch - right_center), "L", "R"
        )
    elif len(known_l):
        hands = hands.fillna("L")
    elif len(known_r):
        hands = hands.fillna("R")
    else:
        hands = hands.fillna(
            pd.Series(
                np.where(recording["pitch"].to_numpy() < 60, "L", "R"),
                index=recording.index,
                dtype="string",
            )
        )
    return hands


def hmm_features(
    notes: pd.DataFrame, models: Mapping[str, str | Path]
) -> pd.DataFrame:
    """Run the existing PIG-trained HMM independently of detector labels."""
    loaded = {hand: load_model(str(Path(path))) for hand, path in models.items()}
    output = pd.DataFrame(
        {
            "note_id": notes["note_id"].to_numpy(),
            "hmm_hand": pd.Series(pd.NA, index=notes.index, dtype="string"),
            "hmm_finger": pd.array([pd.NA] * len(notes), dtype="Int64"),
            "hmm_feature_available": False,
        }
    )
    for _, recording_indices in notes.groupby("recording_id", sort=False).indices.items():
        recording_indices = np.asarray(recording_indices)
        recording = notes.iloc[recording_indices]
        inferred = _infer_missing_hands(recording)
        for hand in ("L", "R"):
            local_mask = inferred.eq(hand).to_numpy()
            local_positions = np.flatnonzero(local_mask)
            if not len(local_positions):
                continue
            sequence = recording.iloc[local_positions].sort_values(
                ["onset_sec", "note_idx"], kind="stable"
            )
            model = loaded[hand]
            pitches = sequence["pitch"].astype(int).tolist()
            labels = [None] * len(sequence)
            if model.get("type") == "extended":
                predicted = unconstrained_viterbi_extended_fast(
                    pitches, sequence["onset_sec"].astype(float).tolist(), model
                )
            else:
                predicted = constrained_viterbi(pitches, labels, model)
            target_indices = sequence.index.to_numpy()
            output.loc[target_indices, "hmm_hand"] = hand
            output.loc[target_indices, "hmm_finger"] = pd.array(
                predicted, dtype="Int64"
            )
            output.loc[target_indices, "hmm_feature_available"] = True
    return output.reset_index(drop=True)


def disagreement_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "note_id",
        "pred_hand",
        "pred_finger",
        "hmm_hand",
        "hmm_finger",
        "hmm_feature_available",
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"disagreement input missing columns: {sorted(missing)}")
    available = (
        features["hmm_feature_available"].fillna(False)
        & features["pred_hand"].isin(["L", "R"])
        & features["pred_finger"].notna()
    )
    hand_disagreement = available & features["pred_hand"].ne(features["hmm_hand"])
    finger_disagreement = (
        available
        & ~hand_disagreement
        & features["pred_finger"].ne(features["hmm_finger"])
    )
    categories = np.select(
        [hand_disagreement, finger_disagreement, available],
        [
            "hand_disagreement",
            "within_hand_finger_disagreement",
            "exact_agreement",
        ],
        default="unavailable",
    )
    return pd.DataFrame(
        {
            "note_id": features["note_id"].to_numpy(),
            "hmm_disagreement_available": available.to_numpy(dtype=bool),
            "hmm_hand_disagreement": hand_disagreement.to_numpy(dtype=bool),
            "hmm_exact_disagreement": (
                hand_disagreement | finger_disagreement
            ).to_numpy(dtype=bool),
            "agreement_category": categories,
        }
    )

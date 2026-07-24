from __future__ import annotations

import pandas as pd


def label_errors(notes_with_gt: pd.DataFrame) -> pd.DataFrame:
    result = notes_with_gt.copy()
    missing_hand = result["pred_hand"].isna()
    missing_finger = result["pred_finger"].isna()
    hand_error = missing_hand | result["pred_hand"].ne(result["gt_hand"])
    finger_mismatch = result["pred_finger"].ne(result["gt_finger"])
    within_hand = ~hand_error & (missing_finger | finger_mismatch)
    result["hand_error"] = hand_error.astype(bool)
    result["within_hand_finger_error"] = within_hand.astype(bool)
    result["exact_error"] = (hand_error | within_hand).astype(bool)
    return result

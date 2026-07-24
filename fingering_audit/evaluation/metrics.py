from __future__ import annotations

import math

import pandas as pd

from ..contracts import MetricRecord


FINGER_IDS = tuple(f"{hand}{finger}" for hand in ("L", "R") for finger in range(1, 6))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def compute_metrics(
    selection: pd.Series,
    labels: pd.DataFrame,
    *,
    set_id: str,
    error_column: str = "exact_error",
) -> MetricRecord:
    selected = pd.Series(selection, index=labels.index).fillna(False).astype(bool)
    errors = labels[error_column].fillna(False).astype(bool)
    hard_count = int(selected.sum())
    error_count = int(errors.sum())
    selected_errors = int((selected & errors).sum())
    selected_correct = int((selected & ~errors).sum())
    correct_count = int((~errors).sum())
    precision = _ratio(selected_errors, hard_count)
    prevalence = _ratio(error_count, len(labels))
    enrichment = precision / prevalence if prevalence and not math.isnan(precision) else math.nan
    return MetricRecord(
        set_id=set_id,
        values={
            "eligible_notes": int(len(labels)),
            "hard_count": hard_count,
            "hard_percentage": _ratio(hard_count, len(labels)),
            "error_count": error_count,
            "selected_errors": selected_errors,
            "error_recall": _ratio(selected_errors, error_count),
            "precision": precision,
            "correct_sieve_rate": _ratio(selected_correct, correct_count),
            "enrichment": enrichment,
        },
    )


def per_finger_metrics(
    selection: pd.Series,
    labels: pd.DataFrame,
    *,
    set_id: str,
    error_column: str = "exact_error",
) -> pd.DataFrame:
    selected = pd.Series(selection, index=labels.index).fillna(False).astype(bool)
    rows = []
    for finger_id in FINGER_IDS:
        mask = labels["gt_finger_id"].eq(finger_id)
        errors = labels[error_column].fillna(False).astype(bool) & mask
        gt_notes = int(mask.sum())
        error_count = int(errors.sum())
        selected_notes = int((selected & mask).sum())
        selected_errors = int((selected & errors).sum())
        rows.append(
            {
                "set_id": set_id,
                "finger_id": finger_id,
                "gt_notes": gt_notes,
                "errors": error_count,
                "selected_notes": selected_notes,
                "selected_errors": selected_errors,
                "error_recall": _ratio(selected_errors, error_count),
                "precision": _ratio(selected_errors, selected_notes),
            }
        )
    return pd.DataFrame.from_records(rows)


def workload_per_predicted_finger(
    selection: pd.Series, notes: pd.DataFrame, *, set_id: str
) -> pd.DataFrame:
    selected = pd.Series(selection, index=notes.index).fillna(False).astype(bool)
    finger = notes["pred_finger_id"].fillna("NA")
    rows = []
    for finger_id in (*FINGER_IDS, "NA"):
        mask = finger.eq(finger_id)
        rows.append(
            {
                "set_id": set_id,
                "predicted_finger_id": finger_id,
                "eligible_notes": int(mask.sum()),
                "hard_count": int((selected & mask).sum()),
                "hard_percentage": _ratio(int((selected & mask).sum()), int(mask.sum())),
            }
        )
    return pd.DataFrame.from_records(rows)

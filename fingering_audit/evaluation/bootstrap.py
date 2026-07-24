from __future__ import annotations

import numpy as np
import pandas as pd


def clustered_intervals(
    selection: pd.Series,
    labels: pd.DataFrame,
    *,
    seed: int,
    replicates: int = 2000,
    error_column: str = "exact_error",
) -> dict[str, float | int]:
    """Percentile intervals from resampling intact recordings."""
    selected = pd.Series(selection, index=labels.index).fillna(False).astype(bool)
    errors = labels[error_column].fillna(False).astype(bool)
    records = []
    for _, indices in labels.groupby("recording_id", sort=True).indices.items():
        indices = np.asarray(indices)
        local_selected = selected.iloc[indices]
        local_errors = errors.iloc[indices]
        records.append(
            (
                int(local_errors.sum()),
                int((local_selected & local_errors).sum()),
                int(local_selected.sum()),
            )
        )
    cluster = np.asarray(records, dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cluster), size=(replicates, len(cluster)))
    totals = cluster[draws].sum(axis=1)
    recall = np.divide(
        totals[:, 1],
        totals[:, 0],
        out=np.full(replicates, np.nan),
        where=totals[:, 0] > 0,
    )
    precision = np.divide(
        totals[:, 1],
        totals[:, 2],
        out=np.full(replicates, np.nan),
        where=totals[:, 2] > 0,
    )
    return {
        "cluster_count": int(len(cluster)),
        "replicates": int(replicates),
        "error_recall_ci_low": float(np.nanquantile(recall, 0.025)),
        "error_recall_ci_high": float(np.nanquantile(recall, 0.975)),
        "precision_ci_low": float(np.nanquantile(precision, 0.025)),
        "precision_ci_high": float(np.nanquantile(precision, 0.975)),
    }

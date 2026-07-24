from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ManualCheck.hard_part_selector import load_fingering_tsv, select_hard_parts

from .canonical import attach_ground_truth, load_ground_truth, load_pianovam_notes
from .contracts import AuditConfig
from .evaluation.labels import label_errors
from .evaluation.bootstrap import clustered_intervals
from .evaluation.metrics import (
    FINGER_IDS,
    compute_metrics,
    per_finger_metrics,
    workload_per_predicted_finger,
)
from .features.ergonomic import ergonomic_features
from .features.model import disagreement_features, hmm_features


SPAN_BOUNDS = {
    "practical": {
        "1-2": (-5, 10), "1-3": (-4, 12), "1-4": (-3, 14),
        "1-5": (-1, 15), "2-3": (1, 5), "2-4": (1, 7),
        "2-5": (2, 10), "3-4": (1, 4), "3-5": (1, 7),
        "4-5": (1, 5),
    },
    "comfortable": {
        "1-2": (-3, 8), "1-3": (-2, 10), "1-4": (-1, 12),
        "1-5": (1, 13), "2-3": (1, 3), "2-4": (1, 5),
        "2-5": (2, 8), "3-4": (1, 2), "3-5": (1, 5),
        "4-5": (1, 3),
    },
    "relative": {
        "1-2": (1, 5), "1-3": (3, 7), "1-4": (5, 9),
        "1-5": (7, 10), "2-3": (1, 2), "2-4": (3, 4),
        "2-5": (5, 6), "3-4": (1, 2), "3-5": (3, 4),
        "4-5": (1, 2),
    },
}


@dataclass
class StudyData:
    notes: pd.DataFrame
    labels: pd.DataFrame
    features: pd.DataFrame
    selections_full: Mapping[str, pd.Series]
    selections_gt: Mapping[str, pd.Series]
    set_metadata: pd.DataFrame
    fold_thresholds: pd.DataFrame


def _span_mask(features: pd.DataFrame, variant: str) -> pd.Series:
    bounds = SPAN_BOUNDS[variant]
    lower = features["finger_pair"].map({key: value[0] for key, value in bounds.items()})
    upper = features["finger_pair"].map({key: value[1] for key, value in bounds.items()})
    return (
        features["directed_pair_span"].lt(lower)
        | features["directed_pair_span"].gt(upper)
    ).fillna(False)


def _legacy_default_mask(notes: pd.DataFrame) -> pd.Series:
    selected = pd.Series(False, index=notes.index)
    for source_path, indices in notes.groupby("source_path", sort=True).indices.items():
        frame = load_fingering_tsv(source_path)
        result = select_hard_parts(
            frame,
            enabled_rules=["impossible_fingering", "fast_jump", "noinfo_cluster"],
        )
        positions = np.asarray(indices)
        if len(result) != len(positions):
            raise ValueError(f"legacy selector length mismatch: {source_path}")
        selected.iloc[positions] = result["is_hard"].to_numpy(dtype=bool)
    return selected


def _map_to_gt(mask: pd.Series, notes: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    lookup = pd.Series(mask.to_numpy(dtype=bool), index=notes["note_id"])
    return labels["note_id"].map(lookup).fillna(False).astype(bool)


def _oof_upper_tail(
    features: pd.DataFrame,
    notes: pd.DataFrame,
    labels: pd.DataFrame,
    quantile: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    score = features["position_change_rate"].replace([np.inf, -np.inf], np.nan)
    labeled_score = labels["note_id"].map(pd.Series(score.to_numpy(), index=notes["note_id"]))
    gt_mask = pd.Series(False, index=labels.index)
    rows = []
    thresholds = []
    for held_out in sorted(labels["recording_id"].unique()):
        train = labels["recording_id"].ne(held_out)
        train_scores = labeled_score.loc[train].dropna()
        threshold = float(train_scores.quantile(quantile))
        test = labels["recording_id"].eq(held_out)
        gt_mask.loc[test] = labeled_score.loc[test].ge(threshold).fillna(False)
        thresholds.append(threshold)
        rows.append(
            {
                "rule_id": "time_conditioned_large_position_change",
                "variant": f"q{quantile:.3f}",
                "held_out_recording": held_out,
                "threshold_semitones_per_second": threshold,
                "train_notes": int(len(train_scores)),
            }
        )
    deployment_threshold = float(np.median(thresholds))
    full_mask = score.ge(deployment_threshold).fillna(False)
    return full_mask, gt_mask, pd.DataFrame.from_records(rows)


def _rule_masks(
    notes: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.DataFrame]:
    full: dict[str, pd.Series] = {}
    gt: dict[str, pd.Series] = {}
    assigned = notes["pred_finger_id"].notna()
    full["missing"] = ~assigned
    full["span_practical"] = _span_mask(features, "practical") & assigned
    full["span_comfortable"] = _span_mask(features, "comfortable") & assigned
    full["span_relative"] = _span_mask(features, "relative") & assigned
    full["crossing"] = features["non_thumb_crossing"].fillna(False) & assigned
    full["step_crossing"] = (
        full["crossing"] & features["absolute_pitch_change"].le(2).fillna(False)
    )
    full["hmm_disagreement"] = (
        features["hmm_exact_disagreement"].fillna(False) & assigned
    )
    full["hmm_agreement"] = (
        features["hmm_disagreement_available"].fillna(False)
        & ~features["hmm_exact_disagreement"].fillna(False)
        & assigned
    )
    full["legacy_default"] = _legacy_default_mask(notes) & assigned
    full["legacy_fast_jump"] = (
        features["absolute_pitch_change"].ge(15)
        & features["prev_ioi_ms"].le(180)
        & assigned
    ).fillna(False)

    threshold_rows = []
    for name, quantile in (("rate_q995", 0.995), ("rate_q990", 0.990), ("rate_q975", 0.975)):
        full[name], gt[name], rows = _oof_upper_tail(
            features, notes, labels, quantile
        )
        full[name] &= assigned
        threshold_rows.append(rows.assign(rule_mask=name))

    for name, mask in full.items():
        if name not in gt:
            gt[name] = _map_to_gt(mask, notes, labels)
    return full, gt, pd.concat(threshold_rows, ignore_index=True)


def _combine(masks: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
    assigned = ~masks["missing"]
    ergonomic_central = masks["span_comfortable"] | masks["crossing"]
    ergonomic_sensitive = masks["span_relative"] | masks["crossing"]
    return {
        "mandatory_missing": masks["missing"],
        "legacy_current_default": masks["legacy_default"],
        "bl_span_practical": masks["span_practical"],
        "bl_span_comfortable": masks["span_comfortable"],
        "bl_span_relative": masks["span_relative"],
        "bl_crossing": masks["crossing"],
        "bl_step_crossing": masks["step_crossing"],
        "bl_rate_q995": masks["rate_q995"],
        "bl_rate_q990": masks["rate_q990"],
        "bl_rate_q975": masks["rate_q975"],
        "bl_hmm_disagreement": masks["hmm_disagreement"],
        "bl_practical_or_rate995": masks["span_practical"] | masks["rate_q995"],
        "bl_practical_or_crossing": masks["span_practical"] | masks["crossing"],
        "bl_two_signal_strict": (
            masks["span_practical"].astype(int)
            + masks["crossing"].astype(int)
            + masks["rate_q990"].astype(int)
            + masks["hmm_disagreement"].astype(int)
        ).ge(2),
        "wl_model_agreement": assigned & ~masks["hmm_agreement"],
        "wl_strict_obvious": assigned & ~(
            masks["hmm_agreement"]
            & ~ergonomic_sensitive
            & ~masks["rate_q975"]
        ),
        "hy_direct_plus_corroborated": (
            masks["span_practical"]
            | (masks["crossing"] & masks["hmm_disagreement"])
            | (masks["rate_q990"] & masks["hmm_disagreement"])
        ),
        "hy_two_of_three_families": (
            ergonomic_central.astype(int)
            + masks["hmm_disagreement"].astype(int)
            + masks["rate_q990"].astype(int)
        ).ge(2),
        "hy_hierarchical": (
            masks["span_practical"]
            | (
                (
                    masks["span_comfortable"]
                    | masks["crossing"]
                    | masks["rate_q990"]
                )
                & masks["hmm_disagreement"]
            )
        ),
    }


def _metadata() -> pd.DataFrame:
    rows = [
        ("mandatory_missing", "integrity", "physical_invariant", "schema completeness"),
        ("legacy_current_default", "baseline", "exploratory", "legacy 500ms/15st-180ms/3-note cluster"),
        ("bl_span_practical", "blacklist", "research_supported", "Parncutt MaxPrac"),
        ("bl_span_comfortable", "blacklist", "research_supported", "Parncutt MaxComf"),
        ("bl_span_relative", "blacklist", "research_supported", "Parncutt MaxRel"),
        ("bl_crossing", "blacklist", "research_supported_corroboration_required", "non-thumb crossing"),
        ("bl_step_crossing", "blacklist", "exploratory", "crossing and <=2 semitones"),
        ("bl_rate_q995", "blacklist", "empirically_calibrated", "LOPO 99.5th percentile"),
        ("bl_rate_q990", "blacklist", "empirically_calibrated", "LOPO 99th percentile"),
        ("bl_rate_q975", "blacklist", "empirically_calibrated", "LOPO 97.5th percentile"),
        ("bl_hmm_disagreement", "blacklist", "research_supported", "PIG-trained HMM disagreement"),
        ("bl_practical_or_rate995", "blacklist", "mixed", "MaxPrac OR LOPO q99.5"),
        ("bl_practical_or_crossing", "blacklist", "mixed", "MaxPrac OR crossing"),
        ("bl_two_signal_strict", "blacklist", "mixed", "at least 2 of 4 families"),
        ("wl_model_agreement", "whitelist", "research_supported", "only HMM agreement is safe"),
        ("wl_strict_obvious", "whitelist", "mixed", "agreement + relative span + low rate"),
        ("hy_direct_plus_corroborated", "hybrid", "mixed", "MaxPrac direct; weak signals need HMM"),
        ("hy_two_of_three_families", "hybrid", "mixed", "at least 2 of ergonomic/model/context"),
        ("hy_hierarchical", "hybrid", "mixed", "MaxPrac direct; central risks need HMM"),
    ]
    return pd.DataFrame(rows, columns=["set_id", "strategy", "evidence_grade", "threshold_summary"])


def build_study(config: AuditConfig) -> StudyData:
    notes = load_pianovam_notes(config.pianovam_fingering_dir)
    gt = load_ground_truth(config.ground_truth_module)
    labels = label_errors(attach_ground_truth(notes, gt))
    features = ergonomic_features(notes)
    hmm = hmm_features(
        notes,
        {
            hand: config.repository_root / f"FingeringInterpolation/models/hmm_{hand}.npz"
            for hand in ("L", "R")
        },
    )
    disagreement_input = pd.concat(
        [
            notes[["note_id", "pred_hand", "pred_finger"]].reset_index(drop=True),
            hmm.drop(columns=["note_id"]).reset_index(drop=True),
        ],
        axis=1,
    )
    disagreement = disagreement_features(disagreement_input)
    features = pd.concat(
        [
            features.reset_index(drop=True),
            hmm.drop(columns=["note_id"]).reset_index(drop=True),
            disagreement.drop(columns=["note_id"]).reset_index(drop=True),
        ],
        axis=1,
    )
    rule_full, rule_gt, thresholds = _rule_masks(notes, labels, features)
    return StudyData(
        notes=notes,
        labels=labels,
        features=features,
        selections_full=_combine(rule_full),
        selections_gt=_combine(rule_gt),
        set_metadata=_metadata(),
        fold_thresholds=thresholds,
    )


def summarize_study(
    study: StudyData, pig_status: str, *, seed: int = 20260723
) -> dict[str, pd.DataFrame]:
    metadata = study.set_metadata.set_index("set_id")
    assigned_full = study.notes["pred_finger_id"].notna()
    assigned_gt = study.labels["pred_finger_id"].notna()
    rows = []
    per_finger = []
    workload = []
    per_recording = []
    error_types = []
    for set_id, full_mask in study.selections_full.items():
        gt_mask = study.selections_gt[set_id]
        metric = compute_metrics(gt_mask, study.labels, set_id=set_id)
        assigned_metric = compute_metrics(
            gt_mask.loc[assigned_gt].reset_index(drop=True),
            study.labels.loc[assigned_gt].reset_index(drop=True),
            set_id=set_id,
        )
        finger = per_finger_metrics(gt_mask, study.labels, set_id=set_id)
        per_finger.append(finger.assign(scope="all_gt"))
        per_finger.append(
            per_finger_metrics(
                gt_mask.loc[assigned_gt].reset_index(drop=True),
                study.labels.loc[assigned_gt].reset_index(drop=True),
                set_id=set_id,
            ).assign(scope="assigned_gt")
        )
        workload.append(
            workload_per_predicted_finger(full_mask, study.notes, set_id=set_id)
        )
        finger_recalls = finger["error_recall"].dropna()
        worst = (
            finger.loc[finger["error_recall"].idxmin(), "finger_id"]
            if len(finger_recalls)
            else "NA"
        )
        meta = metadata.loc[set_id]
        intervals = clustered_intervals(
            gt_mask, study.labels, seed=seed, replicates=2000
        )
        row = {
            "strategy": meta["strategy"],
            "set_id": set_id,
            "evidence_grade": meta["evidence_grade"],
            "threshold_summary": meta["threshold_summary"],
            "pig_status": pig_status,
            "recommendable": False,
            "hard_count": int(full_mask.sum()),
            "hard_percentage_all_notes": float(full_mask.mean()),
            "hard_percentage_assigned_notes": float(
                (full_mask & assigned_full).sum() / assigned_full.sum()
            ),
            **{f"gt_{key}": value for key, value in metric.values.items()},
            "assigned_gt_error_recall": assigned_metric.values["error_recall"],
            "assigned_gt_precision": assigned_metric.values["precision"],
            "macro_finger_recall": float(finger_recalls.mean()),
            "worst_finger": worst,
            **intervals,
        }
        rows.append(row)
        for recording_id, indices in study.labels.groupby("recording_id").indices.items():
            local = pd.Series(gt_mask.iloc[np.asarray(indices)].to_numpy())
            local_labels = study.labels.iloc[np.asarray(indices)].reset_index(drop=True)
            values = compute_metrics(local, local_labels, set_id=set_id).values
            per_recording.append(
                {"set_id": set_id, "recording_id": recording_id, **values}
            )
        for error_column in ("exact_error", "hand_error", "within_hand_finger_error"):
            values = compute_metrics(
                gt_mask, study.labels, set_id=set_id, error_column=error_column
            ).values
            error_types.append(
                {"set_id": set_id, "error_type": error_column, **values}
            )
    filter_sets = pd.DataFrame.from_records(rows)
    filter_sets = filter_sets.sort_values(
        ["strategy", "hard_count", "set_id"], kind="stable"
    ).reset_index(drop=True)
    overlap_ids = list(study.selections_full)
    overlap = pd.DataFrame(index=overlap_ids, columns=overlap_ids, dtype=int)
    for left in overlap_ids:
        for right in overlap_ids:
            overlap.loc[left, right] = int(
                (study.selections_full[left] & study.selections_full[right]).sum()
            )
    return {
        "filter_sets": filter_sets,
        "per_finger": pd.concat(per_finger, ignore_index=True),
        "workload_per_finger": pd.concat(workload, ignore_index=True),
        "per_recording": pd.DataFrame.from_records(per_recording),
        "error_types": pd.DataFrame.from_records(error_types),
        "overlap_matrix": overlap,
        "threshold_sensitivity": study.fold_thresholds,
    }

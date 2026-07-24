from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .acquire import PigUnavailableError, ensure_pig
from .canonical import load_pig_canonical
from .contracts import AuditConfig
from .evidence import (
    RecommendationGateError,
    enforce_recommendation_gate,
    load_evidence_ledger,
)
from .evaluation.metrics import FINGER_IDS
from .manifest import RunManifest, stage_key
from .physical_policy import derive_physical_policy, write_physical_policy
from .report import EXPECTED_QUEUE_SELECTION_IDS, write_reports
from .study import build_study, summarize_study


def _run_id(config: AuditConfig, label: str | None) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str).encode()
    suffix = hashlib.sha256(payload).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_label = "".join(c for c in (label or "") if c.isalnum() or c in "-_")
    return "-".join(part for part in (timestamp, clean_label, suffix) if part)


def _queue_selection_ids(selection_ids) -> set[str]:
    return {
        str(set_id)
        for set_id in selection_ids
        if str(set_id).startswith("ni_") or "__ni_" in str(set_id)
    }


def reconcile_exact_queue_selection_universe(selection_ids) -> bool:
    return _queue_selection_ids(selection_ids) == EXPECTED_QUEUE_SELECTION_IDS


def reconcile_exact_queue_selection_universes(
    full_selection_ids,
    gt_selection_ids,
) -> dict[str, bool]:
    full_ids = _queue_selection_ids(full_selection_ids)
    gt_ids = _queue_selection_ids(gt_selection_ids)
    return {
        "exact_full_queue_selection_universe": (
            full_ids == EXPECTED_QUEUE_SELECTION_IDS
        ),
        "exact_gt_queue_selection_universe": (
            gt_ids == EXPECTED_QUEUE_SELECTION_IDS
        ),
        "exact_queue_selection_key_parity": full_ids == gt_ids,
    }


def run_research(
    config: AuditConfig,
    *,
    limit_recordings: int | None = None,
    run_label: str | None = None,
) -> Path:
    if limit_recordings is not None:
        raise ValueError(
            "limit-recordings is disabled for authoritative research runs; "
            "use the test fixtures for smoke tests"
        )
    manifest = RunManifest.start(config, _run_id(config, run_label))
    stage = "evidence"
    try:
        ledger_path = config.repository_root / "fingering_audit/evidence/thresholds.yaml"
        ledger = load_evidence_ledger(ledger_path)
        manifest.complete_stage(
            "evidence",
            stage_key("evidence", {str(ledger_path): "validated"}, {"rules": len(ledger.rules)}),
            ledger_path,
        )

        stage = "pig_gate"
        physical_policy = None
        pig_gate_open = False
        try:
            pig_root = ensure_pig(config)
            pig_notes = load_pig_canonical(pig_root)
            physical_policy = derive_physical_policy(pig_notes, pig_root)
            write_physical_policy(
                physical_policy,
                manifest.run_dir / "data/physical_policy.yaml",
            )
            try:
                enforce_recommendation_gate(
                    physical_policy.validations,
                    physical_policy.validations.keys(),
                )
                pig_gate_open = True
                pig_status = (
                    f"passed: {len(pig_notes)} annotations across "
                    f"{pig_notes['piece_id'].nunique()} pieces and "
                    f"{pig_notes[['piece_id', 'performer_id']].drop_duplicates().shape[0]} "
                    f"performances; {len(physical_policy.validations)} "
                    "physical-policy validations passed"
                )
            except RecommendationGateError as exc:
                pig_status = f"failed: {exc}"
        except PigUnavailableError as exc:
            pig_root = None
            pig_notes = None
            pig_status = f"unavailable: {exc}"
        (manifest.run_dir / "pig_status.json").write_text(
            json.dumps({"status": pig_status}, indent=2) + "\n", encoding="utf-8"
        )
        manifest.complete_stage(
            "pig_gate",
            stage_key("pig_gate", {}, {"status": pig_status}),
            manifest.run_dir / "pig_status.json",
        )

        stage = "study"
        study = build_study(config, physical_policy=physical_policy)
        queue_universe_reconciliations = (
            reconcile_exact_queue_selection_universes(
                study.selections_full,
                study.selections_gt,
            )
        )
        exact_queue_selection_universe = all(
            queue_universe_reconciliations.values()
        )
        if not exact_queue_selection_universe:
            failed = sorted(
                name
                for name, passed in queue_universe_reconciliations.items()
                if not passed
            )
            raise ValueError(f"failed reconciliation checks: {failed}")
        data_dir = manifest.run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if pig_notes is not None:
            pig_notes.to_parquet(data_dir / "pig_canonical.parquet", index=False)
        study.notes.to_parquet(data_dir / "canonical_notes.parquet", index=False)
        study.labels.to_parquet(data_dir / "ground_truth_labels.parquet", index=False)
        study.features.to_parquet(data_dir / "features.parquet", index=False)
        pd.DataFrame(
            {key: value.to_numpy(dtype=bool) for key, value in study.selections_full.items()}
        ).assign(note_id=study.notes["note_id"].to_numpy()).to_parquet(
            data_dir / "selection_masks.parquet", index=False
        )
        pd.DataFrame(
            {
                key: value.to_numpy(dtype=bool)
                for key, value in study.queue_masks_full.items()
            }
        ).assign(note_id=study.notes["note_id"].to_numpy()).to_parquet(
            data_dir / "queue_masks.parquet", index=False
        )
        manifest.complete_stage(
            "study",
            stage_key(
                "study",
                {"notes": str(len(study.notes)), "gt": str(len(study.labels))},
                {"seed": config.random_seed},
            ),
            data_dir / "features.parquet",
        )

        stage = "reports"
        tables = summarize_study(study, pig_status, seed=config.random_seed)
        files = write_reports(
            manifest.run_dir,
            tables,
            corpus_notes=len(study.notes),
            assigned_notes=int(study.notes["pred_finger_id"].notna().sum()),
            missing_notes=int(study.notes["pred_finger_id"].isna().sum()),
            pig_status=pig_status,
        )
        manifest.complete_stage(
            "reports",
            stage_key("reports", {"sets": str(len(tables["filter_sets"]))}, {}),
            manifest.run_dir / "report/research_report.md",
        )

        queue_variants = {
            set_id: (
                set_id.split("__", 1)[1] if "__" in set_id else set_id
            )
            for set_id in EXPECTED_QUEUE_SELECTION_IDS
        }
        mandatory_masks_contained = (
            exact_queue_selection_universe
            and all(
                (
                    (
                        study.queue_masks_full["physical_must_alert"]
                        <= study.selections_full[set_id]
                    ).all()
                    and (
                        study.queue_masks_full[variant]
                        <= study.selections_full[set_id]
                    ).all()
                    and (
                        study.queue_masks_gt["physical_must_alert"]
                        <= study.selections_gt[set_id]
                    ).all()
                    and (
                        study.queue_masks_gt[variant]
                        <= study.selections_gt[set_id]
                    ).all()
                )
                for set_id, variant in queue_variants.items()
            )
        )
        integrity_disjoint_from_assigned = (
            exact_queue_selection_universe
            and not (
                study.queue_masks_full["physical_must_alert"]
                & study.queue_masks_full[
                    "data_integrity_must_resolve"
                ]
            ).any()
            and not (
                study.queue_masks_gt["physical_must_alert"]
                & study.queue_masks_gt[
                    "data_integrity_must_resolve"
                ]
            ).any()
            and all(
                (
                    not (
                        study.queue_masks_full[
                            "data_integrity_must_resolve"
                        ]
                        & study.selections_full[set_id]
                    ).any()
                    and not (
                        study.queue_masks_gt[
                            "data_integrity_must_resolve"
                        ]
                        & study.selections_gt[set_id]
                    ).any()
                )
                for set_id in queue_variants
            )
        )
        queue_summary = pd.read_csv(
            manifest.run_dir / "results/queue_summary.csv"
        )
        queue_workload = pd.read_csv(
            manifest.run_dir / "results/queue_workload_per_finger.csv"
        )
        filter_queue = tables["filter_sets"].loc[
            tables["filter_sets"]["set_id"].isin(
                _queue_selection_ids(tables["filter_sets"]["set_id"])
            )
        ]
        filter_counts = (
            filter_queue
            .set_index("set_id")
            ["hard_count"]
            .astype(int)
            .to_dict()
        )
        queue_counts = (
            queue_summary.set_index("set_id")["hard_count"]
            .astype(int)
            .to_dict()
        )
        workload_counts = (
            queue_workload.groupby("set_id")["hard_count"]
            .sum()
            .astype(int)
            .to_dict()
        )
        all_gt_finger_counts = (
            tables["per_finger"]
            .loc[lambda frame: frame["scope"].eq("all_gt")]
            .loc[
                lambda frame: frame["set_id"].isin(
                    _queue_selection_ids(frame["set_id"])
                )
            ]
            .groupby("set_id")["selected_notes"]
            .sum()
            .astype(int)
            .to_dict()
        )
        queue_gt_counts = (
            queue_summary.set_index("set_id")["gt_hard_count"]
            .astype(int)
            .to_dict()
        )
        reconciliations = {
            "all_1800_gt_labels_present": len(study.labels) == 1800,
            "all_508621_notes_present": len(study.notes) == 508621,
            "all_ten_gt_fingers_reported": set(FINGER_IDS).issubset(
                set(tables["per_finger"]["finger_id"])
            ),
            "all_filter_counts_reconcile": all(
                int(row.hard_count) == int(study.selections_full[row.set_id].sum())
                for row in tables["filter_sets"].itertuples()
            ),
            "report_files_present": all(path.is_file() for path in files),
            **queue_universe_reconciliations,
            "exact_queue_selection_universe": exact_queue_selection_universe,
            "mandatory_masks_contained": mandatory_masks_contained,
            "integrity_disjoint_from_assigned": (
                integrity_disjoint_from_assigned
            ),
            "queue_summary_reconciles": (
                EXPECTED_QUEUE_SELECTION_IDS == set(filter_counts)
                and EXPECTED_QUEUE_SELECTION_IDS == set(queue_counts)
                and queue_counts == filter_counts
            ),
            "queue_workload_reconciles": (
                EXPECTED_QUEUE_SELECTION_IDS == set(workload_counts)
                and workload_counts == queue_counts
            ),
            "queue_per_finger_reconciles": (
                EXPECTED_QUEUE_SELECTION_IDS == set(queue_gt_counts)
                and EXPECTED_QUEUE_SELECTION_IDS
                == set(all_gt_finger_counts)
                and queue_gt_counts == all_gt_finger_counts
            ),
        }
        if not pig_gate_open:
            manifest.close_recommendation_gate(pig_status, reconciliations)
        else:
            manifest.finalize(reconciliations)
        return manifest.run_dir
    except Exception as exc:
        manifest.fail(stage, f"{type(exc).__name__}: {exc}")
        raise

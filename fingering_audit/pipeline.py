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
from .contracts import RuleKind
from .evidence import (
    enforce_recommendation_gate,
    load_evidence_ledger,
    validate_pig,
)
from .evaluation.metrics import FINGER_IDS
from .manifest import RunManifest, stage_key
from .report import write_reports
from .study import build_study, summarize_study


def _run_id(config: AuditConfig, label: str | None) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str).encode()
    suffix = hashlib.sha256(payload).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_label = "".join(c for c in (label or "") if c.isalnum() or c in "-_")
    return "-".join(part for part in (timestamp, clean_label, suffix) if part)


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
        try:
            pig_root = ensure_pig(config)
            pig_notes = load_pig_canonical(pig_root)
            invalidity_rules = [
                rule for rule in ledger.rules.values() if rule.kind is RuleKind.INVALIDITY
            ]
            validations = {
                rule.rule_id: validate_pig(rule, pig_notes)
                for rule in invalidity_rules
            }
            enforce_recommendation_gate(validations, validations)
            pig_status = (
                f"passed: {len(pig_notes)} annotations across "
                f"{pig_notes['piece_id'].nunique()} pieces and "
                f"{pig_notes[['piece_id', 'performer_id']].drop_duplicates().shape[0]} "
                f"performances; {len(validations)} invalidity rules passed"
            )
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
        study = build_study(config)
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
        }
        if pig_root is None:
            manifest.close_recommendation_gate(pig_status, reconciliations)
        else:
            manifest.finalize(reconciliations)
        return manifest.run_dir
    except Exception as exc:
        manifest.fail(stage, f"{type(exc).__name__}: {exc}")
        raise

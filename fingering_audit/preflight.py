from __future__ import annotations

from .canonical import load_ground_truth, load_pianovam_notes
from .contracts import AuditConfig


def preflight_summary(config: AuditConfig) -> dict[str, int]:
    notes = load_pianovam_notes(config.pianovam_fingering_dir)
    gt = load_ground_truth(config.ground_truth_module)
    return {
        "tsv_files": int(notes["recording_id"].nunique()),
        "notes": int(len(notes)),
        "gt_recordings": int(gt["recording_id"].nunique()),
        "gt_labels": int(len(gt)),
    }

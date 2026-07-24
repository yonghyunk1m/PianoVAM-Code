import json
from dataclasses import replace
from pathlib import Path

import pytest

from fingering_audit import pipeline as pipeline_module
from fingering_audit.config import load_config
from fingering_audit.manifest import RunManifest, sha256_file, stage_key
from fingering_audit.report import EXPECTED_QUEUE_SELECTION_IDS


FIXTURES = Path(__file__).parent / "fixtures"


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("one", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("two", encoding="utf-8")
    assert sha256_file(path) != first


def test_stage_key_is_stable_and_input_sensitive():
    first = stage_key("canonical", {"a": "hash-a"}, {"seed": 7})
    assert first == stage_key("canonical", {"a": "hash-a"}, {"seed": 7})
    assert first != stage_key("canonical", {"a": "hash-b"}, {"seed": 7})


def test_success_marker_requires_all_reconciliations(tmp_path):
    cfg = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    manifest = RunManifest.start(cfg, run_id="test-run")
    assert not (manifest.run_dir / "SUCCESS.json").exists()

    with pytest.raises(ValueError, match="reconciliation"):
        manifest.finalize({"counts_match": False})

    assert not (manifest.run_dir / "SUCCESS.json").exists()
    manifest.finalize({"counts_match": True, "pig_gate": True})
    payload = json.loads((manifest.run_dir / "SUCCESS.json").read_text())
    assert payload["status"] == "success"


def test_failed_mandatory_reconciliation_cannot_finalize(tmp_path):
    cfg = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    manifest = RunManifest.start(cfg, run_id="mandatory-failure")

    with pytest.raises(ValueError, match="mandatory"):
        manifest.finalize(
            {
                "counts_match": True,
                "pig_gate": True,
                "mandatory_masks_contained": False,
            }
        )

    assert not (manifest.run_dir / "SUCCESS.json").exists()


def test_failed_exact_queue_universe_cannot_close_terminal_gate(tmp_path):
    cfg = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    manifest = RunManifest.start(cfg, run_id="queue-universe-failure")

    with pytest.raises(ValueError, match="exact_queue_selection_universe"):
        manifest.close_recommendation_gate(
            "PIG unavailable",
            {
                "mandatory_masks_contained": True,
                "exact_queue_selection_universe": False,
            },
        )

    assert not (
        manifest.run_dir / "RECOMMENDATION_GATE_CLOSED.json"
    ).exists()


def _assert_gt_only_universe_failure_blocks_both_markers(
    tmp_path, gt_selection_ids
):
    reconcile = getattr(
        pipeline_module,
        "reconcile_exact_queue_selection_universes",
        lambda full_ids, gt_ids: {
            "exact_full_queue_selection_universe": None,
            "exact_gt_queue_selection_universe": None,
            "exact_queue_selection_key_parity": None,
        },
    )
    reconciliations = reconcile(
        EXPECTED_QUEUE_SELECTION_IDS,
        gt_selection_ids,
    )
    assert reconciliations[
        "exact_full_queue_selection_universe"
    ] is True
    assert reconciliations[
        "exact_gt_queue_selection_universe"
    ] is False
    assert reconciliations["exact_queue_selection_key_parity"] is False

    cfg = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    manifest = RunManifest.start(cfg, run_id="gt-universe-failure")
    with pytest.raises(ValueError, match="exact_gt"):
        manifest.finalize(reconciliations)
    with pytest.raises(ValueError, match="exact_gt"):
        manifest.close_recommendation_gate(
            "PIG unavailable",
            reconciliations,
        )
    assert not (manifest.run_dir / "SUCCESS.json").exists()
    assert not (
        manifest.run_dir / "RECOMMENDATION_GATE_CLOSED.json"
    ).exists()


def test_gt_only_missing_selection_blocks_both_terminal_markers(tmp_path):
    missing = next(iter(EXPECTED_QUEUE_SELECTION_IDS))
    gt_selection_ids = set(EXPECTED_QUEUE_SELECTION_IDS)
    gt_selection_ids.remove(missing)

    _assert_gt_only_universe_failure_blocks_both_markers(
        tmp_path,
        gt_selection_ids,
    )


def test_gt_only_extra_selection_blocks_both_terminal_markers(tmp_path):
    gt_selection_ids = set(EXPECTED_QUEUE_SELECTION_IDS)
    gt_selection_ids.add("unexpected_gt_method__ni_k2_r1")

    _assert_gt_only_universe_failure_blocks_both_markers(
        tmp_path,
        gt_selection_ids,
    )

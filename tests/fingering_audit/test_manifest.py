import json
from dataclasses import replace
from pathlib import Path

import pytest

from fingering_audit.config import load_config
from fingering_audit.manifest import RunManifest, sha256_file, stage_key


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

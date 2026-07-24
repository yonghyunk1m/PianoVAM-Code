from dataclasses import replace
from pathlib import Path

import pytest

from fingering_audit.config import discover_paths, load_config
from fingering_audit.preflight import preflight_summary


FIXTURES = Path(__file__).parent / "fixtures"


def test_default_config_is_noninteractive_and_safe():
    cfg = load_config(FIXTURES / "research-minimal.yaml")
    assert cfg.noninteractive is True
    assert cfg.overwrite_sources is False
    assert cfg.random_seed == 20260723
    assert cfg.target_budgets == (10_000, 20_000, 30_000, 50_000)


def test_path_discovery_never_selects_output_as_input(tmp_path):
    cfg = load_config(FIXTURES / "research-minimal.yaml")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    resolved = discover_paths(
        replace(
            cfg,
            repository_root=tmp_path,
            pianovam_fingering_dir=input_dir,
            artifact_root=tmp_path / "artifacts",
        )
    )
    assert resolved.artifact_root not in resolved.input_roots


def test_unknown_configuration_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nunexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_source_output_overlap_is_rejected(tmp_path):
    cfg = load_config(FIXTURES / "research-minimal.yaml")
    source = tmp_path / "data"
    source.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        discover_paths(
            replace(
                cfg,
                repository_root=tmp_path,
                pianovam_fingering_dir=source,
                artifact_root=source / "results",
            )
        )


def test_preflight_counts_fixture_inputs():
    cfg = load_config(FIXTURES / "research-minimal.yaml")
    summary = preflight_summary(cfg)
    assert summary["tsv_files"] == 1
    assert summary["notes"] == 4
    assert summary["gt_recordings"] == 1
    assert summary["gt_labels"] == 4

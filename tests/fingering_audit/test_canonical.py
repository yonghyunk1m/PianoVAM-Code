from pathlib import Path
from dataclasses import replace
import hashlib

import pandas as pd
import pytest

from fingering_audit.canonical import (
    attach_ground_truth,
    load_ground_truth,
    load_pianovam_notes,
)
from fingering_audit.contracts import AuditConfig
from fingering_audit.acquire import ensure_authoritative_timing
from fingering_audit.config import load_config
from fingering_audit.evaluation.labels import label_errors
from fingering_audit.features.audit_flags import compute_audit_flags
from fingering_audit.study import build_study


FIXTURES = Path(__file__).parent / "fixtures"


def test_canonical_loader_normalizes_labels_and_ids():
    notes = load_pianovam_notes(FIXTURES / "pianovam")
    assert list(notes["note_id"]) == [
        "trial_a#0",
        "trial_a#1",
        "trial_a#2",
        "trial_a#3",
    ]
    assert notes.loc[0, "pred_finger_id"] == "L1"
    assert notes.loc[2, "pred_finger_id"] == "R1"
    assert pd.isna(notes.loc[3, "pred_finger_id"])
    assert pd.isna(notes.loc[0, "offset_sec"])


def test_canonical_loader_does_not_iterate_rows(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("row-wise DataFrame iteration is forbidden")

    monkeypatch.setattr(pd.DataFrame, "iterrows", forbidden)
    notes = load_pianovam_notes(FIXTURES / "pianovam")
    assert len(notes) == 4


def test_authoritative_loader_enriches_offsets_and_retains_source_hash(tmp_path):
    fingering_dir = tmp_path / "fingering"
    fingering_dir.mkdir()
    source_path = fingering_dir / "take.tsv"
    source_path.write_text(
        "onset\tnote\thand\tfinger\tvelocity\n"
        "1.000000\t60\tR\t1\t80\n",
        encoding="utf-8",
    )
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        timing_cache_dir=tmp_path / "cache",
    )

    def downloader(_url, destination):
        destination.write_text(
            "# onset\tkey_offset\tframe_offset\tnote\tvelocity\n"
            "1.000000\t1.125000\t1.200000\t60\t80\n",
            encoding="utf-8",
        )

    timing_source = ensure_authoritative_timing(
        config, ["take"], downloader=downloader
    )
    notes = load_pianovam_notes(fingering_dir, timing_source=timing_source)

    assert notes["offset_sec"].tolist() == [1.125]
    assert notes["source_sha256"].tolist() == [before]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before


def test_authoritative_loader_rejects_incomplete_timing_source(tmp_path):
    from fingering_audit.contracts import TimingSource

    incomplete = TimingSource(
        cache_dir=tmp_path,
        repository_id="PianoVAM/PianoVAM_v1",
        revision="7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8",
        recording_ids=("trial_a",),
        provenance=pd.DataFrame(),
        complete=False,
    )

    with pytest.raises(ValueError, match="complete"):
        load_pianovam_notes(
            FIXTURES / "pianovam", timing_source=incomplete
        )


def test_gt_attachment_and_error_taxonomy():
    notes = load_pianovam_notes(FIXTURES / "pianovam")
    gt = load_ground_truth(FIXTURES / "gt_fixture.py")
    labeled = label_errors(attach_ground_truth(notes, gt))

    assert len(labeled) == 4
    assert not bool(labeled.loc[0, "exact_error"])

    assert bool(labeled.loc[1, "exact_error"])
    assert not bool(labeled.loc[1, "hand_error"])
    assert bool(labeled.loc[1, "within_hand_finger_error"])

    assert bool(labeled.loc[2, "hand_error"])
    assert not bool(labeled.loc[2, "within_hand_finger_error"])

    assert bool(labeled.loc[3, "exact_error"])
    assert bool(labeled.loc[3, "hand_error"])
    assert not bool(labeled.loc[3, "within_hand_finger_error"])
    assert labeled.loc[3, "gt_finger_id"] == "R2"


def test_loader_and_study_preserve_malformed_rows_as_integrity(tmp_path):
    source = tmp_path / "pianovam"
    source.mkdir()
    (source / "malformed.tsv").write_text(
        "onset\tkey_offset\tnote\thand\tfinger\tvelocity\n"
        "0.0\t0.4\t60\tR\t1\t80\n"
        "bad\t0.8\t61\tR\t2\t80\n"
        "0.2\tinf\t200\tR\t3\t80\n"
        "0.3\t0.7\t64\tNoinfo\tNoinfo\t80\n",
        encoding="utf-8",
    )
    ground_truth = tmp_path / "gt.py"
    ground_truth.write_text(
        "GT_MAP = {'malformed': ["
        "('R', 1), ('R', 2), ('R', 3), ('R', 4)]}\n",
        encoding="utf-8",
    )

    notes = load_pianovam_notes(source)

    assert len(notes) == 4
    assert pd.isna(notes.loc[1, "onset_sec"])
    assert notes.loc[2, "pitch"] == 200
    config = AuditConfig(
        schema_version=1,
        noninteractive=True,
        random_seed=7,
        repository_root=Path.cwd(),
        pianovam_fingering_dir=source,
        ground_truth_module=ground_truth,
        pig_search_roots=(),
        detector_search_roots=(),
        artifact_root=tmp_path / "artifacts",
        target_budgets=(),
        overwrite_sources=False,
        materialize_missing_detector_outputs=False,
        strict_recommendation_gate=True,
    )

    study = build_study(config)

    integrity = study.queue_masks_full["data_integrity_must_resolve"]
    assert integrity.tolist() == [False, True, True, True]
    assert study.features["hmm_feature_available"].tolist() == [
        True,
        False,
        False,
        True,
    ]


def test_fractional_finger_is_preserved_for_integrity_audit(tmp_path):
    source = tmp_path / "pianovam"
    source.mkdir()
    (source / "fractional.tsv").write_text(
        "onset\tkey_offset\tnote\thand\tfinger\tvelocity\n"
        "0.0\t0.4\t60\tR\t1\t80\n"
        "0.5\t0.9\t62\tR\t2.5\t80\n",
        encoding="utf-8",
    )
    ground_truth = tmp_path / "gt.py"
    ground_truth.write_text(
        "GT_MAP = {'fractional': [('R', 1), ('R', 2)]}\n",
        encoding="utf-8",
    )

    notes = load_pianovam_notes(source)
    flags = compute_audit_flags(notes.assign(compound_fingering=False))

    assert notes.loc[1, "pred_finger"] == 2.5
    assert pd.isna(notes.loc[1, "pred_finger_id"])
    assert "non_integral_finger" in flags.integrity_reasons.iloc[1]
    config = AuditConfig(
        schema_version=1,
        noninteractive=True,
        random_seed=7,
        repository_root=Path.cwd(),
        pianovam_fingering_dir=source,
        ground_truth_module=ground_truth,
        pig_search_roots=(),
        detector_search_roots=(),
        artifact_root=tmp_path / "artifacts",
        target_budgets=(),
        overwrite_sources=False,
        materialize_missing_detector_outputs=False,
        strict_recommendation_gate=True,
    )

    study = build_study(config)

    assert study.queue_masks_full[
        "data_integrity_must_resolve"
    ].tolist() == [False, True]
    assert study.features["hmm_feature_available"].tolist() == [True, True]
    assert pd.isna(study.features.loc[1, "finger_pair"])
    assert not bool(study.features.loc[1, "hmm_disagreement_available"])

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ManualCheck.hard_part_selector import (
    load_fingering_tsv,
    select_hard_parts,
)
from fingering_audit import report as report_module
from fingering_audit import pipeline as pipeline_module
from fingering_audit.canonical import load_pianovam_notes, load_pig_canonical
from fingering_audit.config import load_config
from fingering_audit.contracts import AuditConfig, PigValidation
from fingering_audit.features.audit_flags import compute_audit_flags
from fingering_audit.manifest import RunManifest
from fingering_audit.physical_policy import derive_physical_policy
from fingering_audit.report import REQUIRED_RESULTS
from fingering_audit.study import (
    NOINFO_VARIANTS,
    StudyData,
    _combined_sets,
    _legacy_default_mask,
    _musical_input_mask,
    _oof_noinfo_tail,
    _valid_assignment,
    build_study,
    summarize_study,
)

CALIBRATED_VARIANTS = {
    f"ni_w{window}_q{quantile}"
    for window in (5, 9, 17)
    for quantile in (995, 990, 975)
}
FIXTURES = Path(__file__).parent / "fixtures"
APPROVED_BASE_RISK_IDS = (
    "mandatory_missing",
    "legacy_current_default",
    "bl_span_practical",
    "bl_span_comfortable",
    "bl_span_relative",
    "bl_crossing",
    "bl_step_crossing",
    "bl_rate_q995",
    "bl_rate_q990",
    "bl_rate_q975",
    "bl_hmm_disagreement",
    "bl_practical_or_rate995",
    "bl_practical_or_crossing",
    "bl_two_signal_strict",
    "wl_model_agreement",
    "wl_strict_obvious",
    "hy_direct_plus_corroborated",
    "hy_two_of_three_families",
    "hy_hierarchical",
)
EXPECTED_COMBINED_QUEUE_IDS = frozenset(
    f"{base_id}__{variant}"
    for base_id in APPROVED_BASE_RISK_IDS
    for variant in NOINFO_VARIANTS
)
EXPECTED_STANDALONE_QUEUE_IDS = frozenset(
    {*NOINFO_VARIANTS, *CALIBRATED_VARIANTS}
)
EXPECTED_QUEUE_SELECTION_IDS = (
    EXPECTED_COMBINED_QUEUE_IDS | EXPECTED_STANDALONE_QUEUE_IDS
)


def test_required_results_include_queue_tables():
    assert "noinfo_sensitivity.csv" in REQUIRED_RESULTS
    assert "queue_summary.csv" in REQUIRED_RESULTS
    assert "queue_workload_per_finger.csv" in REQUIRED_RESULTS


def test_queue_report_columns_name_auditable_metrics():
    assert getattr(report_module, "QUEUE_REPORT_COLUMNS", None) == [
        "base_risk_method",
        "physical_policy_status",
        "noinfo_min_run",
        "noinfo_context_radius",
        "hard_count",
        "hard_percentage_all_notes",
        "gt_error_recall",
        "assigned_gt_error_recall",
        "gt_precision",
        "error_enrichment",
        "incremental_count_beyond_physical",
        "incremental_errors_beyond_physical",
    ]


def test_exported_queue_selection_contract_is_exact_and_independent():
    assert getattr(report_module, "APPROVED_BASE_RISK_IDS", None) == (
        APPROVED_BASE_RISK_IDS
    )
    assert getattr(report_module, "EXPECTED_COMBINED_QUEUE_IDS", None) == (
        EXPECTED_COMBINED_QUEUE_IDS
    )
    assert getattr(report_module, "EXPECTED_STANDALONE_QUEUE_IDS", None) == (
        EXPECTED_STANDALONE_QUEUE_IDS
    )
    assert len(EXPECTED_COMBINED_QUEUE_IDS) == 171
    assert len(EXPECTED_STANDALONE_QUEUE_IDS) == 18
    assert len(EXPECTED_QUEUE_SELECTION_IDS) == 189


def test_complete_exact_queue_selection_universe_passes():
    reconcile = getattr(
        pipeline_module,
        "reconcile_exact_queue_selection_universe",
        lambda selection_ids: None,
    )

    assert reconcile(EXPECTED_QUEUE_SELECTION_IDS) is True


def test_missing_combined_queue_selection_fails_exact_universe():
    reconcile = getattr(
        pipeline_module,
        "reconcile_exact_queue_selection_universe",
        lambda selection_ids: None,
    )
    missing = next(iter(EXPECTED_COMBINED_QUEUE_IDS))
    selections = dict.fromkeys(EXPECTED_QUEUE_SELECTION_IDS, True)
    selections.pop(missing)

    assert reconcile(selections) is False


def test_missing_standalone_queue_selection_fails_exact_universe():
    reconcile = getattr(
        pipeline_module,
        "reconcile_exact_queue_selection_universe",
        lambda selection_ids: None,
    )
    missing = next(iter(EXPECTED_STANDALONE_QUEUE_IDS))
    selections = dict.fromkeys(EXPECTED_QUEUE_SELECTION_IDS, True)
    selections.pop(missing)

    assert reconcile(selections) is False


def test_extra_queue_selection_fails_exact_universe():
    reconcile = getattr(
        pipeline_module,
        "reconcile_exact_queue_selection_universe",
        lambda selection_ids: None,
    )

    assert (
        reconcile(
            EXPECTED_QUEUE_SELECTION_IDS
            | {"unexpected_method__ni_k2_r1"}
        )
        is False
    )


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_pig_present_run_persists_and_passes_validated_physical_policy(
    tmp_path, monkeypatch
):
    config = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    observed = {}
    original_build = pipeline_module.build_study
    original_gate = pipeline_module.enforce_recommendation_gate

    def tracking_build(config, physical_policy=None):
        observed["policy"] = physical_policy
        return original_build(config, physical_policy=physical_policy)

    def tracking_gate(validations, rule_ids):
        observed["validated_rule_ids"] = tuple(rule_ids)
        return original_gate(validations, observed["validated_rule_ids"])

    monkeypatch.setattr(pipeline_module, "build_study", tracking_build)
    monkeypatch.setattr(
        pipeline_module, "enforce_recommendation_gate", tracking_gate
    )

    with pytest.raises(ValueError, match="reconciliation"):
        pipeline_module.run_research(config, run_label="pig-present-fixture")

    run_dir = next(tmp_path.iterdir())
    policy = observed["policy"]
    assert policy is not None
    assert set(observed["validated_rule_ids"]) == set(policy.validations)
    assert (run_dir / "data/physical_policy.yaml").is_file()
    assert not (run_dir / "SUCCESS.json").exists()


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_pig_absent_run_keeps_diagnostics_and_closes_success_gate(
    tmp_path, monkeypatch
):
    config = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
        pig_search_roots=(),
    )
    observed = {}
    original_build = pipeline_module.build_study

    def tracking_build(config, physical_policy=None):
        study = original_build(config, physical_policy=physical_policy)
        observed["policy"] = physical_policy
        observed["study"] = study
        return study

    monkeypatch.setattr(pipeline_module, "build_study", tracking_build)

    with pytest.raises(ValueError, match="reconciliation"):
        pipeline_module.run_research(config, run_label="pig-absent-fixture")

    run_dir = next(tmp_path.iterdir())
    assert observed["policy"] is None
    assert (
        "physical_candidate_diagnostic"
        in observed["study"].queue_masks_full
    )
    assert not observed["study"].queue_masks_full["physical_must_alert"].any()
    queue_masks = pd.read_parquet(run_dir / "data/queue_masks.parquet")
    assert {
        "physical_candidate_diagnostic",
        "physical_must_alert",
        "data_integrity_must_resolve",
    } <= set(queue_masks)
    assert (run_dir / "results/noinfo_sensitivity.csv").is_file()
    assert (run_dir / "results/queue_summary.csv").is_file()
    assert (run_dir / "results/queue_workload_per_finger.csv").is_file()
    assert not (run_dir / "SUCCESS.json").exists()


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_pipeline_rejects_gt_only_missing_selection_before_reports(
    tmp_path, monkeypatch
):
    config = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    original_build = pipeline_module.build_study
    missing = next(iter(EXPECTED_QUEUE_SELECTION_IDS))

    def build_with_missing_gt_selection(config, physical_policy=None):
        study = original_build(config, physical_policy=physical_policy)
        selections_gt = dict(study.selections_gt)
        selections_gt.pop(missing)
        return replace(study, selections_gt=selections_gt)

    monkeypatch.setattr(
        pipeline_module,
        "build_study",
        build_with_missing_gt_selection,
    )

    with pytest.raises(
        ValueError,
        match="exact_gt_queue_selection_universe",
    ):
        pipeline_module.run_research(
            config,
            run_label="gt-only-missing",
        )

    run_dir = next(tmp_path.iterdir())
    assert not (run_dir / "SUCCESS.json").exists()
    assert not (
        run_dir / "RECOMMENDATION_GATE_CLOSED.json"
    ).exists()
    assert not (run_dir / "report/research_report.md").exists()


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_failed_physical_validation_emits_diagnostics_and_closes_gate(
    tmp_path, monkeypatch
):
    config = replace(
        load_config(FIXTURES / "research-minimal.yaml"),
        artifact_root=tmp_path,
    )
    pig_root = FIXTURES / "PIG/PianoFingeringDataset_v1.02"
    policy = derive_physical_policy(load_pig_canonical(pig_root), pig_root)
    failed_rule = "simultaneous_pair_span"
    failing_policy = replace(
        policy,
        enabled_rules=policy.enabled_rules - {failed_rule},
        validations={
            **policy.validations,
            failed_rule: PigValidation(
                rule_id=failed_rule,
                status="fail",
                violation_count=1,
                violating_ids=("fixture-note",),
            ),
        },
    )
    observed = {}

    monkeypatch.setattr(
        pipeline_module,
        "derive_physical_policy",
        lambda pig_notes, root: failing_policy,
        raising=False,
    )

    def capture_closed(manifest, reason, reconciliations):
        observed["reason"] = reason
        observed["reconciliations"] = dict(reconciliations)

    def reject_success(manifest, reconciliations):
        pytest.fail("failed physical validation must not finalize SUCCESS")

    monkeypatch.setattr(
        RunManifest, "close_recommendation_gate", capture_closed
    )
    monkeypatch.setattr(RunManifest, "finalize", reject_success)

    run_dir = pipeline_module.run_research(
        config, run_label="pig-validation-failure"
    )

    assert failed_rule in observed["reason"]
    assert (
        observed["reconciliations"]["exact_queue_selection_universe"]
        is True
    )
    assert (run_dir / "data/physical_policy.yaml").is_file()
    assert (run_dir / "data/queue_masks.parquet").is_file()
    assert (run_dir / "results/queue_summary.csv").is_file()
    assert not (run_dir / "SUCCESS.json").exists()


@pytest.fixture
def noinfo_jump_study(tmp_path):
    source = tmp_path / "pianovam"
    source.mkdir()
    source_path = source / "jump.tsv"
    source_path.write_text(
        "onset\tkey_offset\tnote\thand\tfinger\tvelocity\n"
        "0.0\t0.05\t60\tR\t1\t80\n"
        "0.1\t0.15\t100\tR\tNoinfo\t80\n"
        "0.2\t0.25\t62\tR\t5\t80\n",
        encoding="utf-8",
    )
    ground_truth = tmp_path / "gt.py"
    ground_truth.write_text(
        "GT_MAP = {'jump': [('R', 1), ('R', 3), ('R', 5)]}\n",
        encoding="utf-8",
    )
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
    expected = select_hard_parts(
        load_fingering_tsv(source_path),
        enabled_rules=[
            "impossible_fingering",
            "fast_jump",
            "noinfo_cluster",
        ],
    )
    return build_study(config), expected


@pytest.fixture
def study():
    notes = pd.DataFrame(
        {
            "note_id": [f"n{index}" for index in range(6)],
            "recording_id": ["a", "a", "a", "b", "b", "b"],
            "pred_hand": ["R", "R", "R", "L", "L", pd.NA],
            "pred_finger": pd.array([1, 2, 3, 1, 2, pd.NA], dtype="Int64"),
            "pred_finger_id": ["R1", "R2", "R3", "L1", "L2", pd.NA],
        }
    )
    labels = notes.assign(
        gt_finger_id=["R1", "R2", "R3", "L1", "L2", "L3"],
        exact_error=[False, True, False, True, False, True],
        hand_error=[False, False, False, False, False, True],
        within_hand_finger_error=[False, True, False, True, False, False],
    )
    physical = pd.Series([True, False, False, False, False, False])
    integrity = pd.Series([False, False, False, False, False, True])
    queue_full = {
        "physical_candidate_diagnostic": physical.copy(),
        "physical_must_alert": physical,
        "data_integrity_must_resolve": integrity,
    }
    for index, variant in enumerate(NOINFO_VARIANTS):
        mask = pd.Series(False, index=notes.index)
        mask.loc[1 + index % 4] = True
        queue_full[variant] = mask
    for index, variant in enumerate(sorted(CALIBRATED_VARIANTS)):
        mask = pd.Series(False, index=notes.index)
        mask.loc[1 + index % 4] = True
        queue_full[variant] = mask
    queue_gt = {name: mask.copy() for name, mask in queue_full.items()}
    base_full = {"base": pd.Series([False, False, True, False, False, False])}
    base_gt = {name: mask.copy() for name, mask in base_full.items()}
    selections_full = {**base_full, **_combined_sets(base_full, queue_full)}
    selections_gt = {**base_gt, **_combined_sets(base_gt, queue_gt)}
    metadata = pd.DataFrame(
        [
            {
                "set_id": set_id,
                "strategy": "fixture",
                "evidence_grade": "fixture",
                "threshold_summary": "fixture",
            }
            for set_id in selections_full
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "calibration": "fixed",
                "variant": variant,
                "min_run": min_run,
                "radius": radius,
            }
            for variant, (min_run, radius) in NOINFO_VARIANTS.items()
        ]
        + [
            {
                "calibration": "training_fold",
                "variant": variant,
                "min_run": pd.NA,
                "radius": pd.NA,
            }
            for variant in sorted(CALIBRATED_VARIANTS)
        ]
    )
    return StudyData(
        notes=notes,
        labels=labels,
        features=pd.DataFrame(index=notes.index),
        selections_full=selections_full,
        selections_gt=selections_gt,
        set_metadata=metadata,
        fold_thresholds=pd.DataFrame(),
        queue_masks_full=queue_full,
        queue_masks_gt=queue_gt,
        noinfo_sensitivity=sensitivity,
    )


def test_each_combined_set_contains_its_mandatory_masks(study):
    physical = study.queue_masks_full["physical_must_alert"]
    integrity = study.queue_masks_full["data_integrity_must_resolve"]
    assert study.selections_full["base"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    for set_id, selected in study.selections_full.items():
        if "__ni_" not in set_id:
            continue
        variant = set_id.split("__", 1)[1]
        assert (physical <= selected).all()
        assert (study.queue_masks_full[variant] <= selected).all()
        assert not (selected & integrity).any()


def test_noinfo_table_has_nine_fixed_rows_and_finger_outputs(study):
    tables = summarize_study(study, "fixture", seed=7)
    fixed = tables["noinfo_sensitivity"].query("calibration == 'fixed'")
    assert len(fixed) == 9
    assert set(fixed["min_run"]) == {2, 3, 5}
    assert set(fixed["radius"]) == {1, 2, 4}
    assert {
        "hard_count",
        "hard_percentage_all_notes",
        "gt_error_recall",
        "assigned_gt_error_recall",
        "gt_precision",
        "error_enrichment",
        "incremental_count_beyond_physical",
        "incremental_errors_beyond_physical",
    } <= set(fixed)
    combined = {set_id for set_id in study.selections_full if "__ni_" in set_id}
    assert combined <= set(tables["per_finger"]["set_id"])


def test_all_standalone_noinfo_variants_have_finger_outputs(study):
    tables = summarize_study(study, "fixture", seed=7)
    expected = set(NOINFO_VARIANTS) | CALIBRATED_VARIANTS

    assert expected <= set(tables["per_finger"]["set_id"])
    assert expected <= set(tables["workload_per_finger"]["set_id"])


def test_written_queue_tables_reconcile_with_filter_and_finger_outputs(
    study, tmp_path
):
    tables = summarize_study(study, "fixture", seed=7)
    report_module.write_reports(
        tmp_path,
        tables,
        corpus_notes=len(study.notes),
        assigned_notes=int(study.notes["pred_finger_id"].notna().sum()),
        missing_notes=int(study.notes["pred_finger_id"].isna().sum()),
        pig_status="fixture",
    )

    results = tmp_path / "results"
    queue_path = results / "queue_summary.csv"
    workload_path = results / "queue_workload_per_finger.csv"
    assert queue_path.is_file()
    assert workload_path.is_file()

    queue = pd.read_csv(queue_path)
    workload = pd.read_csv(workload_path)
    filters = pd.read_csv(results / "filter_sets.csv").set_index("set_id")
    per_finger = pd.read_csv(results / "per_finger.csv")
    expected_ids = {
        set_id
        for set_id in tables["filter_sets"]["set_id"]
        if set_id.startswith("ni_") or "__ni_" in set_id
    }
    assert expected_ids == set(queue["set_id"])
    assert expected_ids == set(workload["set_id"])
    assert {
        "set_id",
        "noinfo_variant",
        "noinfo_calibration",
        "gt_hard_count",
        *report_module.QUEUE_REPORT_COLUMNS,
    } <= set(queue.columns)

    queue_counts = queue.set_index("set_id")["hard_count"].astype(int)
    assert queue_counts.to_dict() == (
        filters.loc[queue_counts.index, "hard_count"].astype(int).to_dict()
    )
    assert queue_counts.to_dict() == (
        workload.groupby("set_id")["hard_count"].sum().astype(int).to_dict()
    )
    all_gt = per_finger.query("scope == 'all_gt'")
    assert queue.set_index("set_id")["gt_hard_count"].astype(int).to_dict() == (
        all_gt.groupby("set_id")["selected_notes"]
        .sum()
        .loc[queue_counts.index]
        .astype(int)
        .to_dict()
    )

    combined = queue.query("set_id == 'base__ni_k2_r1'").iloc[0]
    calibrated = queue.query("set_id == 'ni_w5_q995'").iloc[0]
    assert combined["base_risk_method"] == "base"
    assert combined["physical_policy_status"] == "fixture"
    assert combined["noinfo_min_run"] == 2
    assert combined["noinfo_context_radius"] == 1
    assert calibrated["noinfo_variant"] == "ni_w5_q995"
    assert calibrated["noinfo_calibration"] == "training_fold"

    markdown = (tmp_path / "report/research_report.md").read_text(
        encoding="utf-8"
    )
    assert "queue_summary.csv" in markdown
    assert "queue_workload_per_finger.csv" in markdown
    assert "per_finger.csv" in markdown


def _study_with_calibrated_fold_rows(
    study, *, windows=(5, 5)
):
    variant = "ni_w5_q995"
    sensitivity = study.noinfo_sensitivity
    base = sensitivity.query("variant == @variant").iloc[0].to_dict()
    folds = pd.DataFrame.from_records(
        [
            {
                **base,
                "window": window,
                "quantile": 0.995,
                "held_out_recording": recording,
                "threshold": threshold,
                "train_nonzero_notes": 3,
            }
            for window, recording, threshold in zip(
                windows, ("a", "b"), (0.25, 0.75)
            )
        ]
    )
    return replace(
        study,
        noinfo_sensitivity=pd.concat(
            [sensitivity.query("variant != @variant"), folds],
            ignore_index=True,
        ),
    )


def test_multifold_calibrated_sensitivity_writes_one_queue_row_per_selection(
    study, tmp_path
):
    multifold = _study_with_calibrated_fold_rows(study)
    tables = summarize_study(multifold, "fixture", seed=7)

    calibrated = tables["noinfo_sensitivity"].query(
        "variant == 'ni_w5_q995'"
    )
    assert len(calibrated) == 2
    assert set(calibrated["held_out_recording"]) == {"a", "b"}

    files = report_module.write_reports(
        tmp_path,
        tables,
        corpus_notes=len(study.notes),
        assigned_notes=int(study.notes["pred_finger_id"].notna().sum()),
        missing_notes=int(study.notes["pred_finger_id"].isna().sum()),
        pig_status="fixture",
    )

    written_sensitivity = pd.read_csv(
        tmp_path / "results/noinfo_sensitivity.csv"
    )
    queue = pd.read_csv(tmp_path / "results/queue_summary.csv")
    expected_queue_rows = tables["filter_sets"]["set_id"].astype(str).map(
        lambda set_id: set_id.startswith("ni_") or "__ni_" in set_id
    )
    assert len(
        written_sensitivity.query("variant == 'ni_w5_q995'")
    ) == 2
    assert queue["set_id"].is_unique
    assert len(queue) == int(expected_queue_rows.sum())
    assert all(path.is_file() for path in files)


def test_inconsistent_multifold_variant_metadata_is_rejected(study, tmp_path):
    inconsistent = _study_with_calibrated_fold_rows(
        study, windows=(5, 9)
    )
    tables = summarize_study(inconsistent, "fixture", seed=7)

    with pytest.raises(
        ValueError,
        match="inconsistent.*noinfo_window",
    ):
        report_module.write_reports(
            tmp_path,
            tables,
            corpus_notes=len(study.notes),
            assigned_notes=int(study.notes["pred_finger_id"].notna().sum()),
            missing_notes=int(study.notes["pred_finger_id"].isna().sum()),
            pig_status="fixture",
        )


def test_legacy_default_preserves_original_noinfo_cluster_context(tmp_path):
    source = tmp_path / "legacy.tsv"
    source.write_text(
        "onset\tkey_offset\tnote\thand\tfinger\n"
        "0.0\t0.5\t60\tR\t1\n"
        "1.0\t1.5\t60\tR\t1\n"
        "2.0\t2.5\t60\tNoinfo\tNoinfo\n"
        "3.0\t3.5\t60\tNoinfo\tNoinfo\n"
        "4.0\t4.5\t60\tNoinfo\tNoinfo\n"
        "5.0\t5.5\t60\tR\t1\n"
        "6.0\t6.5\t60\tR\t1\n",
        encoding="utf-8",
    )
    notes = load_pianovam_notes(tmp_path)
    audit_notes = notes.assign(compound_fingering=False)
    integrity = compute_audit_flags(audit_notes).integrity
    eligible = _valid_assignment(notes) & ~integrity
    expected = select_hard_parts(
        load_fingering_tsv(source),
        enabled_rules=[
            "impossible_fingering",
            "fast_jump",
            "noinfo_cluster",
        ],
    )["is_hard"]

    actual = _legacy_default_mask(notes, _musical_input_mask(notes)) & eligible

    assert expected.tolist() == [True, True, False, False, False, True, True]
    assert actual.tolist() == expected.tolist()


def test_legacy_default_keeps_hand_known_noinfo_in_fast_jump_sequence(
    noinfo_jump_study,
):
    study, original = noinfo_jump_study
    actual = study.selections_full["legacy_current_default"]

    assert original["is_hard"].tolist() == [True, True, True]
    assert actual.iloc[[0, 2]].tolist() == [True, True]


def test_musical_context_keeps_hand_known_noinfo_pitch_and_timing(
    noinfo_jump_study,
):
    study, _ = noinfo_jump_study

    assert study.features.loc[2, "absolute_pitch_change"] == 38
    assert study.features.loc[2, "prev_ioi_ms"] == 100
    legacy_fast_jump = (
        study.features["absolute_pitch_change"].ge(15)
        & study.features["prev_ioi_ms"].le(180)
        & _valid_assignment(study.notes)
        & ~study.queue_masks_full["data_integrity_must_resolve"]
    ).fillna(False)
    assert legacy_fast_jump.tolist() == [
        False,
        False,
        True,
    ]


def test_standalone_noinfo_id_matches_sensitivity_union(study):
    tables = summarize_study(study, "fixture", seed=7)
    variant = "ni_k2_r1"
    sensitivity = tables["noinfo_sensitivity"].query(
        "calibration == 'fixed' and variant == @variant"
    ).iloc[0]
    standalone = tables["filter_sets"].query("set_id == @variant").iloc[0]
    finger_count = (
        tables["per_finger"]
        .query("set_id == @variant and scope == 'all_gt'")["selected_notes"]
        .sum()
    )
    workload_count = (
        tables["workload_per_finger"]
        .query("set_id == @variant")["hard_count"]
        .sum()
    )

    assert sensitivity["hard_count"] == 2
    assert standalone["hard_count"] == sensitivity["hard_count"]
    assert finger_count == sensitivity["hard_count"]
    assert workload_count == sensitivity["hard_count"]


def test_oof_noinfo_tail_uses_other_recordings_and_nonzero_scores_only():
    notes = pd.DataFrame(
        {
            "note_id": ["a0", "a1", "b0", "c0"],
            "recording_id": ["a", "a", "b", "c"],
            "pred_hand": ["R"] * 4,
            "pred_finger": [1] * 4,
        }
    )
    labels = notes[["note_id", "recording_id"]].copy()
    full, gt, thresholds = _oof_noinfo_tail(
        notes,
        labels,
        pd.Series([0.0, 0.9, 0.2, 0.8]),
        quantile=0.5,
    )
    by_fold = thresholds.set_index("held_out_recording")
    assert by_fold.loc["a", "threshold"] == pytest.approx(0.5)
    assert by_fold.loc["b", "threshold"] == pytest.approx(0.85)
    assert by_fold.loc["c", "threshold"] == pytest.approx(0.55)
    assert gt.tolist() == [False, True, False, True]
    assert full.tolist() == [False, True, False, True]


def test_oof_noinfo_tail_fails_closed_without_nonzero_training_scores():
    notes = pd.DataFrame(
        {
            "note_id": ["a0", "b0"],
            "recording_id": ["a", "b"],
            "pred_hand": ["R", "L"],
            "pred_finger": [1, 1],
        }
    )
    labels = notes[["note_id", "recording_id"]].copy()
    full, gt, thresholds = _oof_noinfo_tail(
        notes, labels, pd.Series([0.0, np.nan]), quantile=0.995
    )
    assert not full.any()
    assert not gt.any()
    assert np.isinf(thresholds["threshold"]).all()


def test_assigned_metrics_and_workload_exclude_integrity_rows():
    notes = pd.DataFrame(
        {
            "note_id": ["bad", "valid"],
            "recording_id": ["r", "r"],
            "pred_hand": ["R", "R"],
            "pred_finger": pd.array([1, 1], dtype="Int64"),
            "pred_finger_id": ["R1", "R1"],
        }
    )
    labels = notes.assign(
        gt_finger_id=["R1", "R1"],
        exact_error=[True, True],
        hand_error=[False, False],
        within_hand_finger_error=[True, True],
    )
    selected = pd.Series([False, True])
    integrity = pd.Series([True, False])
    queues = {
        "physical_candidate_diagnostic": pd.Series([False, False]),
        "physical_must_alert": pd.Series([False, False]),
        "data_integrity_must_resolve": integrity,
    }
    integrity_study = StudyData(
        notes=notes,
        labels=labels,
        features=pd.DataFrame(index=notes.index),
        selections_full={"base": selected},
        selections_gt={"base": selected.copy()},
        set_metadata=pd.DataFrame(
            [
                {
                    "set_id": "base",
                    "strategy": "fixture",
                    "evidence_grade": "fixture",
                    "threshold_summary": "fixture",
                }
            ]
        ),
        fold_thresholds=pd.DataFrame(),
        queue_masks_full=queues,
        queue_masks_gt={name: mask.copy() for name, mask in queues.items()},
        noinfo_sensitivity=pd.DataFrame(),
    )

    tables = summarize_study(integrity_study, "fixture", seed=7)

    row = tables["filter_sets"].iloc[0]
    assert row["assigned_gt_error_recall"] == 1.0
    assert row["hard_percentage_assigned_notes"] == 1.0
    finger = tables["per_finger"].query(
        "set_id == 'base' and finger_id == 'R1'"
    ).set_index("scope")
    assert finger.loc["all_gt", "error_recall"] == 0.5
    assert finger.loc["assigned_gt", "error_recall"] == 1.0
    workload = tables["workload_per_finger"].query(
        "set_id == 'base' and predicted_finger_id == 'R1'"
    ).iloc[0]
    assert workload["eligible_notes"] == 1
    assert workload["hard_percentage"] == 1.0


def test_per_finger_all_gt_and_assigned_gt_use_distinct_universes():
    notes = pd.DataFrame(
        {
            "note_id": ["missing", "assigned"],
            "recording_id": ["r", "r"],
            "pred_hand": [pd.NA, "R"],
            "pred_finger": pd.array([pd.NA, 1], dtype="Int64"),
            "pred_finger_id": [pd.NA, "R1"],
        }
    )
    labels = notes.assign(
        gt_finger_id=["R1", "R1"],
        exact_error=[True, True],
        hand_error=[True, False],
        within_hand_finger_error=[False, True],
    )
    selected = pd.Series([False, True])
    integrity = pd.Series([True, False])
    queues = {
        "physical_candidate_diagnostic": pd.Series([False, False]),
        "physical_must_alert": pd.Series([False, False]),
        "data_integrity_must_resolve": integrity,
    }
    scope_study = StudyData(
        notes=notes,
        labels=labels,
        features=pd.DataFrame(index=notes.index),
        selections_full={"base": selected},
        selections_gt={"base": selected.copy()},
        set_metadata=pd.DataFrame(
            [
                {
                    "set_id": "base",
                    "strategy": "fixture",
                    "evidence_grade": "fixture",
                    "threshold_summary": "fixture",
                }
            ]
        ),
        fold_thresholds=pd.DataFrame(),
        queue_masks_full=queues,
        queue_masks_gt={name: mask.copy() for name, mask in queues.items()},
        noinfo_sensitivity=pd.DataFrame(),
    )

    tables = summarize_study(scope_study, "fixture", seed=7)

    filter_row = tables["filter_sets"].iloc[0]
    finger = tables["per_finger"].query(
        "set_id == 'base' and finger_id == 'R1'"
    ).set_index("scope")
    assert filter_row["gt_error_recall"] == 0.5
    assert filter_row["assigned_gt_error_recall"] == 1.0
    assert finger.loc["all_gt", "error_recall"] == 0.5
    assert finger.loc["assigned_gt", "error_recall"] == 1.0

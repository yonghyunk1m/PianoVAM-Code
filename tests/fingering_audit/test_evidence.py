from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fingering_audit.acquire import PigUnavailableError, ensure_pig
from fingering_audit.canonical import load_pig_canonical
from fingering_audit.config import load_config
from fingering_audit.contracts import EvidenceGrade, RuleKind
from fingering_audit.evidence import (
    RecommendationGateError,
    enforce_recommendation_gate,
    load_evidence_ledger,
    validate_pig,
)
from fingering_audit.physical_policy import (
    PRACTICAL_ABS,
    derive_physical_policy,
    write_physical_policy,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_evidence_ledger_loads_complete_entries():
    ledger = load_evidence_ledger(FIXTURES / "evidence-valid.yaml")
    rule = ledger.rules["fixture_time_conditioned_leap"]
    assert rule.kind is RuleKind.RISK
    assert rule.evidence_grade is EvidenceGrade.RESEARCH_SUPPORTED
    assert rule.unit == "semitones_and_milliseconds"
    assert set(rule.variants) == {"conservative", "central", "permissive"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["rules"][0].pop("unit"), "unit"),
        (lambda raw: raw["rules"][0].update(source_keys=[]), "source_keys"),
        (
            lambda raw: raw["rules"][0].update(source_keys=["absent_source"]),
            "absent_source",
        ),
        (
            lambda raw: raw["rules"][0].update(sensitivity_variants={}),
            "sensitivity_variants",
        ),
        (
            lambda raw: raw["rules"][2].update(may_select_alone=True),
            "exploratory",
        ),
    ],
)
def test_evidence_ledger_rejects_unsafe_entries(tmp_path, mutation, message):
    raw = yaml.safe_load((FIXTURES / "evidence-valid.yaml").read_text())
    mutation(raw)
    raw["sources_bib"] = str((FIXTURES / "sources-valid.bib").resolve())
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_evidence_ledger(path)


def test_pig_loader_preserves_signed_and_compound_tokens():
    pig = load_pig_canonical(FIXTURES / "PIG")
    assert len(pig) == 4
    signed = pig.loc[pig["finger_token"].eq("-2")].iloc[0]
    assert signed["finger"] == 2
    assert signed["finger_sign"] == -1
    compound = pig.loc[pig["finger_token"].eq("4_1")].iloc[0]
    assert compound["finger"] == 4
    assert compound["finger_components"] == (4, 1)
    assert bool(compound["compound_fingering"])
    assert set(pig["hand"]) == {"L", "R"}
    assert pig["piece_id"].eq("001").all()
    assert pig["performer_id"].eq("1").all()


def test_ensure_pig_discovers_configured_complete_dataset():
    cfg = load_config(FIXTURES / "research-minimal.yaml")
    found = ensure_pig(cfg)
    assert found.name == "PianoFingeringDataset_v1.02"


def test_ensure_pig_fails_closed_without_official_download(tmp_path):
    cfg = load_config(FIXTURES / "research-minimal.yaml")
    cfg = replace(cfg, pig_search_roots=(tmp_path / "missing",))
    with pytest.raises(PigUnavailableError, match="upon request"):
        ensure_pig(cfg)


def test_risk_rules_are_not_subject_to_pig_invalidity_gate():
    ledger = load_evidence_ledger(FIXTURES / "evidence-valid.yaml")
    pig = load_pig_canonical(FIXTURES / "PIG")
    result = validate_pig(ledger.rules["fixture_time_conditioned_leap"], pig)
    assert result.status == "not_applicable"
    assert result.violation_count == 0


def test_invalidity_gate_reports_ids_and_blocks_violations():
    ledger = load_evidence_ledger(FIXTURES / "evidence-valid.yaml")
    pig = load_pig_canonical(FIXTURES / "PIG")
    synthetic_violation = pig.copy()
    synthetic_violation.loc[1, ["onset_sec", "offset_sec", "pitch", "finger"]] = [
        0.25,
        0.75,
        61,
        1,
    ]
    result = validate_pig(
        ledger.rules["fixture_same_finger_overlap"], synthetic_violation
    )
    assert result.violation_count == 1
    assert result.violating_ids
    with pytest.raises(RecommendationGateError, match="fixture_same_finger_overlap"):
        enforce_recommendation_gate(
            {"fixture_same_finger_overlap": result},
            ("fixture_same_finger_overlap",),
        )


def test_policy_uses_practical_or_pig_maximum():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[1, ["onset_sec", "offset_sec"]] = [0.25, 0.75]

    policy = derive_physical_policy(pig, FIXTURES / "PIG")

    assert policy.observation_counts["1-2"] == 1
    assert policy.observed_maxima["1-2"] == 2
    assert policy.span_boundaries["1-2"] == PRACTICAL_ABS["1-2"]
    assert policy.validations["simultaneous_pair_span"].violation_count == 0
    assert policy.pig_sha256


def test_policy_uses_observed_pig_maximum_above_practical_boundary():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[1, ["onset_sec", "offset_sec", "pitch"]] = [0.25, 0.75, 72]

    policy = derive_physical_policy(pig, FIXTURES / "PIG")

    assert policy.observed_maxima["1-2"] == 12
    assert policy.span_boundaries["1-2"] == 12
    assert policy.validations["simultaneous_pair_span"].status == "pass"


def test_pair_without_pig_simultaneous_coverage_has_no_boundary():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[1, ["onset_sec", "offset_sec"]] = [0.25, 0.75]

    policy = derive_physical_policy(pig, FIXTURES / "PIG")

    assert set(policy.span_boundaries) == {"1-2"}
    assert "1-3" not in policy.observation_counts
    assert "1-3" not in policy.observed_maxima


def test_compound_tokens_are_excluded_from_simple_invalidity():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[0, "finger"] = 4
    pig.loc[2, ["onset_sec", "offset_sec"]] = [0.1, 0.4]

    policy = derive_physical_policy(pig, FIXTURES / "PIG")

    assert policy.validations[
        "simultaneous_same_finger_different_pitch"
    ].violation_count == 0
    assert "1-4" not in policy.observation_counts


def test_failing_rule_is_disabled_but_still_closes_recommendation_gate():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[1, ["onset_sec", "offset_sec", "finger"]] = [0.25, 0.75, 1]

    policy = derive_physical_policy(pig, FIXTURES / "PIG")
    validation = policy.validations[
        "simultaneous_same_finger_different_pitch"
    ]

    assert validation.status == "fail"
    assert validation.violation_count == 2
    assert validation.violating_ids == ("001-1#0@2", "001-1#1@3")
    assert validation.rule_id not in policy.enabled_rules
    with pytest.raises(RecommendationGateError, match=validation.rule_id):
        enforce_recommendation_gate(
            policy.validations, policy.validations.keys()
        )


def test_physical_policy_serialization_is_complete(tmp_path):
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[1, ["onset_sec", "offset_sec"]] = [0.25, 0.75]
    policy = derive_physical_policy(pig, FIXTURES / "PIG")

    path = write_physical_policy(policy, tmp_path / "policy.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["pig_sha256"] == policy.pig_sha256
    assert payload["timing_epsilon_sec"] == 0.001
    assert payload["span_boundaries"] == dict(policy.span_boundaries)
    assert payload["observed_maxima"] == dict(policy.observed_maxima)
    assert payload["observation_counts"] == dict(policy.observation_counts)
    assert payload["enabled_rules"] == sorted(policy.enabled_rules)
    assert set(payload["validations"]) == set(policy.validations)
    assert payload["validations"]["simultaneous_pair_span"]["violating_ids"] == []


def test_physical_policy_mappings_are_immutable():
    pig = load_pig_canonical(FIXTURES / "PIG")
    pig.loc[1, ["onset_sec", "offset_sec"]] = [0.25, 0.75]
    policy = derive_physical_policy(pig, FIXTURES / "PIG")

    with pytest.raises(TypeError):
        policy.span_boundaries["1-2"] = 99
    with pytest.raises(TypeError):
        policy.validations["simultaneous_pair_span"] = None


def test_physical_invalidity_rules_are_evidence_governed():
    ledger = load_evidence_ledger(
        Path(__file__).parents[2] / "fingering_audit/evidence/thresholds.yaml"
    )

    same_finger = ledger.rules["simultaneous_same_finger_different_pitch"]
    span = ledger.rules["simultaneous_pair_span"]
    assert same_finger.kind is RuleKind.INVALIDITY
    assert same_finger.evidence_grade is EvidenceGrade.PHYSICAL_INVARIANT
    assert "compound" in same_finger.rationale.lower()
    assert span.kind is RuleKind.INVALIDITY
    assert span.evidence_grade is EvidenceGrade.PHYSICAL_INVARIANT
    assert "pig" in span.rationale.lower()

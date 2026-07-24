from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yaml

from .contracts import (
    EvidenceGrade,
    EvidenceLedger,
    PigValidation,
    RuleDefinition,
    RuleKind,
)
from .features.audit_flags import AuditFlags


class RecommendationGateError(RuntimeError):
    pass


_REQUIRED_RULE_FIELDS = {
    "rule_id",
    "kind",
    "feature",
    "unit",
    "operator",
    "evidence_grade",
    "source_keys",
    "applicability",
    "may_select_alone",
    "implementation_version",
    "sensitivity_variants",
}


def _bib_keys(path: Path) -> frozenset[str]:
    text = path.read_text(encoding="utf-8")
    return frozenset(
        match.group(1).strip()
        for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text)
    )


def load_evidence_ledger(path: Path) -> EvidenceLedger:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported evidence schema_version")
    bib_path = Path(raw.get("sources_bib", ""))
    if not bib_path.is_absolute():
        bib_path = path.parent / bib_path
    if not bib_path.is_file():
        raise ValueError(f"sources_bib does not exist: {bib_path}")
    source_keys = _bib_keys(bib_path)
    rules: dict[str, RuleDefinition] = {}
    for index, entry in enumerate(raw.get("rules", [])):
        missing = sorted(_REQUIRED_RULE_FIELDS - set(entry))
        if missing:
            raise ValueError(f"rule {index} missing required fields: {missing}")
        rule_sources = tuple(str(key) for key in entry["source_keys"])
        if not rule_sources:
            raise ValueError(f"{entry['rule_id']}: source_keys must not be empty")
        absent = sorted(set(rule_sources) - source_keys)
        if absent:
            raise ValueError(
                f"{entry['rule_id']}: bibliography keys do not exist: {absent}"
            )
        variants = entry["sensitivity_variants"]
        if not isinstance(variants, dict) or not variants:
            raise ValueError(
                f"{entry['rule_id']}: sensitivity_variants must not be empty"
            )
        grade = EvidenceGrade(entry["evidence_grade"])
        may_select_alone = bool(entry["may_select_alone"])
        if grade is EvidenceGrade.EXPLORATORY and may_select_alone:
            raise ValueError(
                f"{entry['rule_id']}: exploratory rules may not select alone"
            )
        rule = RuleDefinition(
            rule_id=str(entry["rule_id"]),
            kind=RuleKind(entry["kind"]),
            evidence_grade=grade,
            feature=str(entry["feature"]),
            variants=variants,
            unit=str(entry["unit"]),
            operator=str(entry["operator"]),
            source_keys=rule_sources,
            applicability=str(entry["applicability"]),
            may_select_alone=may_select_alone,
            implementation_version=int(entry["implementation_version"]),
            rationale=str(entry.get("rationale", "")),
        )
        if rule.rule_id in rules:
            raise ValueError(f"duplicate rule_id: {rule.rule_id}")
        rules[rule.rule_id] = rule
    if not rules:
        raise ValueError("evidence ledger contains no rules")
    return EvidenceLedger(rules=rules, sources=source_keys)


def _same_finger_overlap_violations(pig_notes: pd.DataFrame) -> tuple[str, ...]:
    violations: list[str] = []
    group_columns = ["piece_id", "performer_id", "hand", "finger"]
    ordered = pig_notes.sort_values(group_columns + ["onset_sec", "offset_sec"])
    for _, group in ordered.groupby(group_columns, sort=False, dropna=False):
        active: list[tuple[float, int, str, bool]] = []
        for row in group.itertuples(index=False):
            onset = float(row.onset_sec)
            active = [item for item in active if item[0] > onset]
            if not bool(row.compound_fingering) and any(
                pitch != int(row.pitch) and not compound
                for _, pitch, _, compound in active
            ):
                earlier = next(
                    note_id
                    for _, pitch, note_id, compound in active
                    if pitch != int(row.pitch) and not compound
                )
                violations.append(f"{earlier}->{row.pig_note_id}")
            active.append(
                (
                    float(row.offset_sec),
                    int(row.pitch),
                    str(row.pig_note_id),
                    bool(row.compound_fingering),
                )
            )
    return tuple(violations)


def validate_pig(rule: RuleDefinition, pig_notes: pd.DataFrame) -> PigValidation:
    if rule.kind is not RuleKind.INVALIDITY:
        return PigValidation(
            rule_id=rule.rule_id,
            status="not_applicable",
            violation_count=0,
        )
    if rule.feature == "simultaneous_same_finger_different_pitch":
        violating_ids = _same_finger_overlap_violations(pig_notes)
    else:
        raise ValueError(
            f"invalidity rule {rule.rule_id} has no PIG validator for "
            f"feature {rule.feature}"
        )
    return PigValidation(
        rule_id=rule.rule_id,
        status="pass" if not violating_ids else "fail",
        violation_count=len(violating_ids),
        violating_ids=violating_ids,
    )


def physical_validations_from_flags(
    canonical: pd.DataFrame, flags: AuditFlags
) -> dict[str, PigValidation]:
    rule_masks = {
        "simultaneous_same_finger_different_pitch": flags.same_finger_candidate,
        "simultaneous_pair_span": flags.span_candidate,
    }
    validations = {}
    for rule_id, mask in rule_masks.items():
        violating_ids = tuple(
            canonical.loc[mask.to_numpy(), "note_id"].astype(str)
        )
        validations[rule_id] = PigValidation(
            rule_id=rule_id,
            status="pass" if not violating_ids else "fail",
            violation_count=len(violating_ids),
            violating_ids=violating_ids,
        )
    return validations


def enforce_recommendation_gate(
    validations: Mapping[str, PigValidation], rule_ids: Iterable[str]
) -> None:
    failures = []
    for rule_id in rule_ids:
        validation = validations.get(rule_id)
        if validation is None:
            failures.append(f"{rule_id}=not_validated")
        elif validation.status not in {"pass", "not_applicable"}:
            failures.append(f"{rule_id}={validation.violation_count}_violations")
    if failures:
        raise RecommendationGateError(
            "PIG recommendation gate failed: " + ", ".join(failures)
        )

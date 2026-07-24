from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..contracts import (
    EvidenceGrade,
    FilterSet,
    RuleResult,
    SelectionResult,
)


def combine_mandatory(
    *,
    risk: pd.Series,
    physical: pd.Series,
    noinfo: pd.Series,
    integrity: pd.Series,
) -> pd.Series:
    values = tuple(pd.Series(value) for value in (risk, physical, noinfo, integrity))
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"mandatory mask length mismatch: {sorted(lengths)}")
    masks = [
        pd.Series(value).fillna(False).astype(bool).reset_index(drop=True)
        for value in values
    ]
    risk_mask, physical_mask, noinfo_mask, integrity_mask = masks
    if (physical_mask & integrity_mask).any():
        raise ValueError("physical and integrity masks overlap")
    if (noinfo_mask & integrity_mask).any():
        raise ValueError("noinfo context and integrity masks overlap")
    return (risk_mask | physical_mask | noinfo_mask) & ~integrity_mask


def validate_filter_set(
    filter_set: FilterSet, rule_results: Mapping[str, RuleResult]
) -> None:
    missing = sorted(set(filter_set.rule_ids) - set(rule_results))
    if missing:
        raise ValueError(f"{filter_set.set_id}: missing rule results: {missing}")
    if len(filter_set.rule_ids) == 1:
        only = rule_results[filter_set.rule_ids[0]]
        if only.evidence_grade is EvidenceGrade.EXPLORATORY:
            raise ValueError(
                f"{filter_set.set_id}: exploratory rule cannot select alone"
            )
    if filter_set.logic not in {"any", "all", "all_safe", "at_least_2"}:
        raise ValueError(f"{filter_set.set_id}: unsupported logic {filter_set.logic}")


def evaluate_filter_set(
    filter_set: FilterSet, rule_results: Mapping[str, RuleResult]
) -> SelectionResult:
    validate_filter_set(filter_set, rule_results)
    rules = [rule_results[rule_id] for rule_id in filter_set.rule_ids]
    selected_frame = pd.concat(
        [
            pd.Series(rule.selected).fillna(False).astype(bool).rename(rule.rule_id)
            for rule in rules
        ],
        axis=1,
    )
    available_frame = pd.concat(
        [
            pd.Series(rule.available).fillna(False).astype(bool).rename(rule.rule_id)
            for rule in rules
        ],
        axis=1,
    )
    effective = selected_frame & available_frame
    if filter_set.logic == "any":
        selected = effective.any(axis=1)
    elif filter_set.logic == "all":
        selected = effective.all(axis=1) & available_frame.all(axis=1)
    elif filter_set.logic == "at_least_2":
        selected = effective.sum(axis=1).ge(2)
    else:
        safe = effective.all(axis=1) & available_frame.all(axis=1)
        selected = ~safe
    reasons = effective.apply(
        lambda row: tuple(row.index[row.to_numpy(dtype=bool)]), axis=1
    )
    return SelectionResult(
        set_id=filter_set.set_id,
        selected=selected.astype(bool),
        reasons=reasons,
    )

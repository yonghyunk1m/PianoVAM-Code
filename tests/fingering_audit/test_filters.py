from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fingering_audit.contracts import (
    EvidenceGrade,
    FilterSet,
    RuleKind,
    RuleResult,
    StrategyKind,
)
from fingering_audit.filters.strategies import (
    combine_mandatory,
    evaluate_filter_set,
    validate_filter_set,
)


def _rule(name, selected, available=None, grade=EvidenceGrade.RESEARCH_SUPPORTED):
    selected = pd.Series(selected)
    if available is None:
        available = pd.Series(True, index=selected.index)
    return RuleResult(
        rule_id=name,
        variant="fixture",
        selected=selected,
        available=pd.Series(available),
        evidence_grade=grade,
        kind=RuleKind.RISK,
        pig_status="not_applicable",
    )


def test_blacklist_any_and_all_semantics():
    results = {
        "a": _rule("a", [True, False, True]),
        "b": _rule("b", [False, True, True]),
    }
    any_set = FilterSet("any", StrategyKind.BLACKLIST, ("a", "b"), "any")
    all_set = FilterSet("all", StrategyKind.BLACKLIST, ("a", "b"), "all")
    assert evaluate_filter_set(any_set, results).selected.tolist() == [True, True, True]
    assert evaluate_filter_set(all_set, results).selected.tolist() == [False, False, True]


def test_whitelist_requires_all_evidence_and_selects_unsafe_notes():
    results = {
        "stable": _rule("stable", [True, True, False], [True, False, True]),
        "agree": _rule("agree", [True, True, True]),
    }
    filter_set = FilterSet(
        "strict", StrategyKind.WHITELIST, ("stable", "agree"), "all_safe"
    )
    selection = evaluate_filter_set(filter_set, results).selected
    assert selection.tolist() == [False, True, True]


def test_two_of_three_hybrid():
    results = {
        "a": _rule("a", [True, True, False]),
        "b": _rule("b", [True, False, True]),
        "c": _rule("c", [False, False, True]),
    }
    filter_set = FilterSet(
        "hybrid", StrategyKind.HYBRID, ("a", "b", "c"), "at_least_2"
    )
    assert evaluate_filter_set(filter_set, results).selected.tolist() == [
        True,
        False,
        True,
    ]


def test_exploratory_rule_cannot_select_alone():
    results = {"x": _rule("x", [True], grade=EvidenceGrade.EXPLORATORY)}
    filter_set = FilterSet("x-only", StrategyKind.BLACKLIST, ("x",), "any")
    with pytest.raises(ValueError, match="exploratory"):
        validate_filter_set(filter_set, results)


def test_mandatory_union_is_complete_and_integrity_disjoint():
    result = combine_mandatory(
        risk=pd.Series([False, True, False, False]),
        physical=pd.Series([True, False, False, False]),
        noinfo=pd.Series([False, False, True, False]),
        integrity=pd.Series([False, False, False, True]),
    )
    assert result.tolist() == [True, True, True, False]


def test_mandatory_union_rejects_overlap_with_integrity():
    with pytest.raises(ValueError, match="integrity"):
        combine_mandatory(
            risk=pd.Series([False]),
            physical=pd.Series([True]),
            noinfo=pd.Series([False]),
            integrity=pd.Series([True]),
        )

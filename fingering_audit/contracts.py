from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class EvidenceGrade(str, Enum):
    PHYSICAL_INVARIANT = "physical_invariant"
    RESEARCH_SUPPORTED = "research_supported"
    EMPIRICALLY_CALIBRATED = "empirically_calibrated"
    EXPLORATORY = "exploratory"


class RuleKind(str, Enum):
    INVALIDITY = "invalidity"
    RISK = "risk"
    INTEGRITY = "integrity"


class StrategyKind(str, Enum):
    BLACKLIST = "blacklist"
    WHITELIST = "whitelist"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class AuditConfig:
    schema_version: int
    noninteractive: bool
    random_seed: int
    repository_root: Path
    pianovam_fingering_dir: Path
    ground_truth_module: Path
    pig_search_roots: tuple[Path, ...]
    detector_search_roots: tuple[Path, ...]
    artifact_root: Path
    target_budgets: tuple[int, ...]
    overwrite_sources: bool
    materialize_missing_detector_outputs: bool
    strict_recommendation_gate: bool
    timing_repository_id: str = "PianoVAM/PianoVAM_v1"
    timing_revision: str = "7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8"
    timing_cache_dir: Path = Path(".cache/fingering_audit/authoritative_timing")
    input_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TimingSource:
    cache_dir: Path
    repository_id: str
    revision: str
    recording_ids: tuple[str, ...]
    provenance: Any
    complete: bool


@dataclass(frozen=True)
class TimingJoin:
    notes: Any
    provenance: Any
    complete: bool


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    kind: RuleKind
    evidence_grade: EvidenceGrade
    feature: str
    variants: Mapping[str, Any]
    unit: str = ""
    operator: str = ""
    source_keys: tuple[str, ...] = ()
    applicability: str = ""
    may_select_alone: bool = False
    implementation_version: int = 1
    rationale: str = ""


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    variant: str
    selected: Any
    available: Any
    evidence_grade: EvidenceGrade
    kind: RuleKind
    pig_status: str


@dataclass(frozen=True)
class FilterSet:
    set_id: str
    strategy: StrategyKind
    rule_ids: tuple[str, ...]
    logic: str
    recommendable: bool = True


@dataclass(frozen=True)
class SelectionResult:
    set_id: str
    selected: Any
    reasons: Any


@dataclass(frozen=True)
class PigValidation:
    rule_id: str
    status: str
    violation_count: int
    violating_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceLedger:
    rules: Mapping[str, RuleDefinition]
    sources: frozenset[str]


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_recordings: tuple[str, ...]
    test_recordings: tuple[str, ...]


@dataclass(frozen=True)
class FoldParameters:
    fold_id: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class MetricRecord:
    set_id: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class ResearchResults:
    metrics: Any
    per_finger: Any
    selections: Mapping[str, Any]


@dataclass(frozen=True)
class ReportIndex:
    files: tuple[Path, ...]
    reconciliations: Mapping[str, bool] = field(default_factory=dict)

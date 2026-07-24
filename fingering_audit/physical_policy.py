from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd
import yaml

from .contracts import PigValidation
from .evidence import physical_validations_from_flags
from .features.audit_flags import compute_audit_flags


TIMING_EPSILON_SEC = 0.001

PRACTICAL_ABS = {
    "1-2": max(abs(-5), abs(10)),
    "1-3": max(abs(-4), abs(12)),
    "1-4": max(abs(-3), abs(14)),
    "1-5": max(abs(-1), abs(15)),
    "2-3": max(abs(1), abs(5)),
    "2-4": max(abs(1), abs(7)),
    "2-5": max(abs(2), abs(10)),
    "3-4": max(abs(1), abs(4)),
    "3-5": max(abs(1), abs(7)),
    "4-5": max(abs(1), abs(5)),
}


@dataclass(frozen=True)
class PhysicalPolicy:
    span_boundaries: Mapping[str, int]
    observed_maxima: Mapping[str, int]
    observation_counts: Mapping[str, int]
    enabled_rules: frozenset[str]
    validations: Mapping[str, PigValidation]
    pig_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "span_boundaries", MappingProxyType(dict(self.span_boundaries))
        )
        object.__setattr__(
            self, "observed_maxima", MappingProxyType(dict(self.observed_maxima))
        )
        object.__setattr__(
            self, "observation_counts", MappingProxyType(dict(self.observation_counts))
        )
        object.__setattr__(self, "enabled_rules", frozenset(self.enabled_rules))
        object.__setattr__(
            self, "validations", MappingProxyType(dict(self.validations))
        )


def sha256_dataset_tree(root: Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _pig_to_canonical(pig: pd.DataFrame) -> pd.DataFrame:
    result = pig.rename(
        columns={
            "pig_note_id": "note_id",
            "hand": "pred_hand",
            "finger": "pred_finger",
        }
    ).copy()
    result["recording_id"] = (
        result["piece_id"].astype(str)
        + "-"
        + result["performer_id"].astype(str)
    )
    result["note_idx"] = result["note_index"]
    return result.reset_index(drop=True)


def _simultaneous_pair_maxima(
    canonical: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    integrity = compute_audit_flags(
        canonical, timing_epsilon_sec=TIMING_EPSILON_SEC
    ).integrity
    valid = canonical.loc[~integrity.to_numpy()]
    maxima: dict[str, int] = {}
    counts: dict[str, int] = {}
    for _, group in valid.groupby(["recording_id", "pred_hand"], sort=False):
        active = []
        ordered = group.sort_values(["onset_sec", "note_idx"], kind="stable")
        for row in ordered.itertuples():
            active = [
                earlier
                for earlier in active
                if float(earlier.offset_sec)
                > float(row.onset_sec) + TIMING_EPSILON_SEC
            ]
            for earlier in active:
                first = int(earlier.pred_finger)
                second = int(row.pred_finger)
                if first == second:
                    continue
                pair = f"{min(first, second)}-{max(first, second)}"
                distance = abs(int(earlier.pitch) - int(row.pitch))
                maxima[pair] = max(maxima.get(pair, 0), distance)
                counts[pair] = counts.get(pair, 0) + 1
            active.append(row)
    return maxima, counts


def derive_physical_policy(
    pig_notes: pd.DataFrame, pig_root: Path
) -> PhysicalPolicy:
    simple = pig_notes.loc[~pig_notes["compound_fingering"].astype(bool)].copy()
    canonical = _pig_to_canonical(simple)
    maxima, counts = _simultaneous_pair_maxima(canonical)
    boundaries = {
        pair: max(practical, maxima[pair])
        for pair, practical in PRACTICAL_ABS.items()
        if counts.get(pair, 0) > 0
    }
    flags = compute_audit_flags(
        canonical,
        boundaries,
        timing_epsilon_sec=TIMING_EPSILON_SEC,
    )
    validations = physical_validations_from_flags(canonical, flags)
    return PhysicalPolicy(
        span_boundaries=boundaries,
        observed_maxima=maxima,
        observation_counts=counts,
        enabled_rules=frozenset(validations),
        validations=validations,
        pig_sha256=sha256_dataset_tree(pig_root),
    )


def write_physical_policy(policy: PhysicalPolicy, path: Path) -> Path:
    path = Path(path)
    payload = {
        "schema_version": 1,
        "pig_sha256": policy.pig_sha256,
        "timing_epsilon_sec": TIMING_EPSILON_SEC,
        "span_boundaries": dict(policy.span_boundaries),
        "observed_maxima": dict(policy.observed_maxima),
        "observation_counts": dict(policy.observation_counts),
        "enabled_rules": sorted(policy.enabled_rules),
        "validations": {
            key: asdict(value) for key, value in policy.validations.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path

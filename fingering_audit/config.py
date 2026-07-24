from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from .contracts import AuditConfig


_CONFIG_KEYS = {field.name for field in fields(AuditConfig)} - {"input_roots"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_config(path: Path) -> AuditConfig:
    path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown configuration keys: {unknown}")
    missing = sorted(_CONFIG_KEYS - set(raw))
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")

    config_base = path.parent
    repository_root = _resolve(config_base, raw["repository_root"])
    cfg = AuditConfig(
        schema_version=int(raw["schema_version"]),
        noninteractive=bool(raw["noninteractive"]),
        random_seed=int(raw["random_seed"]),
        repository_root=repository_root,
        pianovam_fingering_dir=_resolve(repository_root, raw["pianovam_fingering_dir"]),
        ground_truth_module=_resolve(repository_root, raw["ground_truth_module"]),
        pig_search_roots=tuple(_resolve(repository_root, p) for p in raw["pig_search_roots"]),
        detector_search_roots=tuple(
            _resolve(repository_root, p) for p in raw["detector_search_roots"]
        ),
        artifact_root=_resolve(repository_root, raw["artifact_root"]),
        target_budgets=tuple(int(v) for v in raw["target_budgets"]),
        overwrite_sources=bool(raw["overwrite_sources"]),
        materialize_missing_detector_outputs=bool(
            raw["materialize_missing_detector_outputs"]
        ),
        strict_recommendation_gate=bool(raw["strict_recommendation_gate"]),
    )
    if not cfg.noninteractive:
        raise ValueError("audit configuration must be noninteractive")
    if cfg.overwrite_sources:
        raise ValueError("overwriting source data is forbidden")
    if cfg.schema_version != 1:
        raise ValueError(f"unsupported schema_version: {cfg.schema_version}")
    return discover_paths(cfg)


def discover_paths(config: AuditConfig) -> AuditConfig:
    inputs = (
        config.pianovam_fingering_dir.resolve(),
        config.ground_truth_module.resolve(),
        *(p.resolve() for p in config.pig_search_roots),
        *(p.resolve() for p in config.detector_search_roots),
    )
    artifact = config.artifact_root.resolve()
    for source in inputs:
        source_dir = source if source.is_dir() or not source.suffix else source.parent
        if _is_within(artifact, source_dir):
            raise ValueError(
                f"artifact/source overlap: {artifact} is inside {source_dir}"
            )
    return replace(config, artifact_root=artifact, input_roots=tuple(inputs))

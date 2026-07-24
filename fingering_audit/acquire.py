from __future__ import annotations

from pathlib import Path

from .config import AuditConfig


class PigUnavailableError(RuntimeError):
    """Raised when the authoritative PIG annotations cannot be obtained safely."""


def _complete_pig_root(path: Path) -> Path | None:
    for candidate in (path / "PianoFingeringDataset_v1.02", path):
        fingering = candidate / "FingeringFiles"
        if fingering.is_dir() and any(fingering.glob("*_fingering.txt")):
            return candidate.resolve()
    return None


def ensure_pig(config: AuditConfig) -> Path:
    """Discover PIG locally; never substitute unofficial or partial annotations."""
    for root in config.pig_search_roots:
        found = _complete_pig_root(Path(root))
        if found is not None:
            return found
    searched = ", ".join(str(path) for path in config.pig_search_roots)
    raise PigUnavailableError(
        "PIG v1.02 annotations are distributed by the official authors upon "
        "request; no checksum-verifiable unattended download is published. "
        f"Searched: {searched}. The strict validity gate remains closed."
    )

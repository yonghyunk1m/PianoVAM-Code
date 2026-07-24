from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Callable
from urllib.request import urlretrieve

import pandas as pd

from .config import AuditConfig
from .contracts import TimingSource


PINNED_TIMING_REPOSITORY = "PianoVAM/PianoVAM_v1"
PINNED_TIMING_REVISION = "7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8"
_RECORDING_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TIMING_COLUMNS = ("onset", "key_offset", "frame_offset", "note", "velocity")


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


def _default_downloader(url: str, destination: Path) -> None:
    urlretrieve(url, destination)


def _timing_file_metadata(path: Path) -> tuple[str, int, int]:
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"empty authoritative timing file: {path}")
    first_line = payload.splitlines()[0].decode("utf-8", errors="strict")
    columns = tuple(
        part.strip().lower()
        for part in first_line.lstrip("# ").split("\t")
    )
    if columns != _TIMING_COLUMNS:
        raise ValueError(
            f"invalid authoritative timing header in {path}: {columns}"
        )
    row_count = sum(
        1
        for line in payload.splitlines()[1:]
        if line.strip() and not line.lstrip().startswith(b"#")
    )
    if row_count == 0:
        raise ValueError(f"authoritative timing file has no rows: {path}")
    return hashlib.sha256(payload).hexdigest(), len(payload), row_count


def _cache_record_path(path: Path) -> Path:
    return path.with_suffix(".tsv.json")


def _write_cache_record(
    path: Path,
    *,
    relative_path: Path,
    digest: str,
    byte_count: int,
    row_count: int,
) -> None:
    record_path = _cache_record_path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}.",
        suffix=".json.tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            {
                "repository_id": PINNED_TIMING_REPOSITORY,
                "revision": PINNED_TIMING_REVISION,
                "relative_source_path": relative_path.as_posix(),
                "sha256": digest,
                "byte_count": byte_count,
                "row_count": row_count,
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
    temporary.replace(record_path)


def _validate_cache_record(
    path: Path,
    *,
    relative_path: Path,
    digest: str,
    byte_count: int,
    row_count: int,
) -> None:
    record_path = _cache_record_path(path)
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cache hash record missing or invalid: {path}") from exc
    expected = {
        "repository_id": PINNED_TIMING_REPOSITORY,
        "revision": PINNED_TIMING_REVISION,
        "relative_source_path": relative_path.as_posix(),
        "sha256": digest,
        "byte_count": byte_count,
        "row_count": row_count,
    }
    if record != expected:
        raise ValueError(f"cache hash mismatch: {path}")


def ensure_authoritative_timing(
    config: AuditConfig,
    recording_ids,
    *,
    downloader: Callable[[str, Path], None] | None = None,
) -> TimingSource:
    """Acquire exactly requested native PianoVAM TSVs at the pinned revision."""
    revision = str(config.timing_revision)
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError(
            "timing_revision must be a 40-character hexadecimal commit"
        )
    if (
        config.timing_repository_id != PINNED_TIMING_REPOSITORY
        or revision != PINNED_TIMING_REVISION
    ):
        raise ValueError(
            "authoritative timing must use the pinned repository and pinned revision"
        )

    requested = tuple(str(value) for value in recording_ids)
    if not requested:
        raise ValueError("at least one recording is required")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate recording IDs are forbidden")
    invalid = [value for value in requested if _RECORDING_ID.fullmatch(value) is None]
    if invalid:
        raise ValueError(f"invalid recording IDs: {invalid}")

    revision_dir = Path(config.timing_cache_dir) / revision
    tsv_dir = revision_dir / "TSV"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    fetch = downloader or _default_downloader
    rows: list[dict] = []
    for recording_id in requested:
        relative_path = Path("TSV") / f"{recording_id}.tsv"
        destination = revision_dir / relative_path
        if not destination.is_file():
            url = (
                "https://huggingface.co/datasets/"
                f"{PINNED_TIMING_REPOSITORY}/resolve/"
                f"{PINNED_TIMING_REVISION}/{relative_path.as_posix()}"
            )
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{recording_id}.",
                    suffix=".tmp",
                    dir=tsv_dir,
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                fetch(url, temporary)
                digest, byte_count, row_count = _timing_file_metadata(temporary)
                temporary.replace(destination)
                _write_cache_record(
                    destination,
                    relative_path=relative_path,
                    digest=digest,
                    byte_count=byte_count,
                    row_count=row_count,
                )
            except Exception as exc:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"failed to acquire authoritative timing for {recording_id}"
                ) from exc

        try:
            digest, byte_count, row_count = _timing_file_metadata(destination)
            _validate_cache_record(
                destination,
                relative_path=relative_path,
                digest=digest,
                byte_count=byte_count,
                row_count=row_count,
            )
        except Exception as exc:
            raise RuntimeError(
                f"cached authoritative timing hash mismatch or validation "
                f"failure for {recording_id}"
            ) from exc
        rows.append(
            {
                "recording_id": recording_id,
                "repository_id": PINNED_TIMING_REPOSITORY,
                "revision": PINNED_TIMING_REVISION,
                "relative_source_path": relative_path.as_posix(),
                "sha256": digest,
                "byte_count": byte_count,
                "row_count": row_count,
                "validation_status": "acquisition_valid",
            }
        )

    provenance = pd.DataFrame.from_records(rows)
    complete = (
        len(provenance) == len(requested)
        and not provenance["recording_id"].duplicated().any()
        and set(provenance["recording_id"]) == set(requested)
        and provenance["validation_status"].eq("acquisition_valid").all()
    )
    if not complete:
        raise RuntimeError("authoritative timing acquisition is incomplete")
    return TimingSource(
        cache_dir=revision_dir.resolve(),
        repository_id=PINNED_TIMING_REPOSITORY,
        revision=PINNED_TIMING_REVISION,
        recording_ids=requested,
        provenance=provenance,
        complete=True,
    )

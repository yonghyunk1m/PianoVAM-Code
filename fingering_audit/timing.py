from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import PINNED_TIMING_REPOSITORY, PINNED_TIMING_REVISION
from .contracts import TimingJoin, TimingSource


_NATIVE_COLUMNS = ("onset", "key_offset", "frame_offset", "note", "velocity")
_PROVENANCE_COLUMNS = {
    "recording_id",
    "repository_id",
    "revision",
    "relative_source_path",
    "sha256",
    "byte_count",
    "row_count",
    "validation_status",
}


def _native_timing(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=_NATIVE_COLUMNS,
    )


def _validated_source_rows(timing_source: TimingSource) -> pd.DataFrame:
    if not timing_source.complete:
        raise ValueError("authoritative TimingSource must be complete")
    if (
        timing_source.repository_id != PINNED_TIMING_REPOSITORY
        or timing_source.revision != PINNED_TIMING_REVISION
    ):
        raise ValueError(
            "TimingSource is not the official pinned repository and revision"
        )
    provenance = timing_source.provenance.copy()
    missing = _PROVENANCE_COLUMNS - set(provenance)
    if missing:
        raise ValueError(f"timing provenance missing columns: {sorted(missing)}")
    expected = set(timing_source.recording_ids)
    if (
        len(timing_source.recording_ids) != len(expected)
        or len(provenance) != len(expected)
        or provenance["recording_id"].duplicated().any()
        or set(provenance["recording_id"]) != expected
    ):
        raise ValueError("timing provenance recording coverage is incomplete")
    if (
        not provenance["repository_id"].eq(PINNED_TIMING_REPOSITORY).all()
        or not provenance["revision"].eq(PINNED_TIMING_REVISION).all()
        or not provenance["validation_status"].eq("acquisition_valid").all()
    ):
        raise ValueError(
            "timing provenance is not the official pinned source identity"
        )

    root = timing_source.cache_dir.resolve()
    if root.name != PINNED_TIMING_REVISION:
        raise ValueError("timing source is outside expected cache layout")
    for row in provenance.itertuples(index=False):
        expected_relative = Path("TSV") / f"{row.recording_id}.tsv"
        relative_path = Path(row.relative_source_path)
        if relative_path != expected_relative or relative_path.is_absolute():
            raise ValueError(
                f"{row.recording_id}: exact relative source path required "
                f"({expected_relative.as_posix()})"
            )
        path = (root / expected_relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("timing provenance path escapes cache") from exc
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row.sha256 or len(payload) != int(row.byte_count):
            raise ValueError(f"timing source hash mismatch: {row.recording_id}")
    return provenance


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _validate_integral_range(
    values: np.ndarray,
    *,
    minimum: int,
    maximum: int,
    label: str,
) -> None:
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or ((values < minimum) | (values > maximum)).any()
    ):
        raise ValueError(f"invalid {label}")


def attach_authoritative_offsets(
    notes: pd.DataFrame,
    timing_source: TimingSource,
) -> TimingJoin:
    """Attach only native key offsets through an exact six-decimal identity join."""
    provenance = _validated_source_rows(timing_source)
    required = {"recording_id", "onset_sec", "pitch", "velocity"}
    missing = required - set(notes)
    if missing:
        raise ValueError(f"canonical notes missing columns: {sorted(missing)}")
    note_recordings = set(notes["recording_id"].astype(str))
    expected_recordings = set(timing_source.recording_ids)
    if note_recordings != expected_recordings:
        raise ValueError(
            "authoritative timing recording coverage does not match canonical notes"
        )

    result = notes.copy()
    result["offset_sec"] = np.nan
    joined_rows: list[dict] = []
    provenance_by_id = provenance.set_index("recording_id", drop=False)
    for recording_id in timing_source.recording_ids:
        note_mask = result["recording_id"].astype(str).eq(recording_id)
        result_positions = np.flatnonzero(note_mask.to_numpy())
        fingering = result.iloc[result_positions][
            ["onset_sec", "pitch", "velocity"]
        ].copy()
        source_row = provenance_by_id.loc[recording_id]
        source_path = (
            timing_source.cache_dir / source_row["relative_source_path"]
        ).resolve()
        native = _native_timing(source_path)
        if len(fingering) != len(native):
            raise ValueError(
                f"{recording_id}: row count mismatch "
                f"({len(fingering)} fingering, {len(native)} native)"
            )
        if len(native) != int(source_row["row_count"]):
            raise ValueError(f"{recording_id}: provenance row count mismatch")

        fingering_onset = _numeric(fingering["onset_sec"])
        fingering_pitch = _numeric(fingering["pitch"])
        fingering_velocity = _numeric(fingering["velocity"])
        native_onset = _numeric(native["onset"])
        native_offset = _numeric(native["key_offset"])
        native_pitch = _numeric(native["note"])
        native_velocity = _numeric(native["velocity"])

        if not np.isfinite(native_onset).all() or not np.isfinite(
            native_offset
        ).all():
            raise ValueError(f"{recording_id}: nonfinite native timing")
        if not np.isfinite(fingering_onset).all():
            raise ValueError(f"{recording_id}: nonfinite fingering onset")
        _validate_integral_range(
            fingering_pitch, minimum=0, maximum=127, label="fingering pitch"
        )
        _validate_integral_range(
            native_pitch, minimum=0, maximum=127, label="native pitch"
        )
        _validate_integral_range(
            fingering_velocity,
            minimum=0,
            maximum=127,
            label="fingering velocity",
        )
        _validate_integral_range(
            native_velocity,
            minimum=0,
            maximum=127,
            label="native velocity",
        )
        if (native_offset < native_onset).any():
            raise ValueError(f"{recording_id}: key offset before onset")

        fingering_keys = pd.DataFrame(
            {
                "_onset_key": np.round(fingering_onset, 6),
                "_pitch_key": fingering_pitch.astype(np.int64),
                "_velocity_fingering": fingering_velocity.astype(np.int64),
                "_result_position": result_positions,
            }
        )
        native_keys = pd.DataFrame(
            {
                "_onset_key": np.round(native_onset, 6),
                "_pitch_key": native_pitch.astype(np.int64),
                "_velocity_native": native_velocity.astype(np.int64),
                "_native_offset": native_offset,
            }
        )
        identity = ["_onset_key", "_pitch_key"]
        if fingering_keys.duplicated(identity).any():
            raise ValueError(f"{recording_id}: duplicate fingering identity key")
        if native_keys.duplicated(identity).any():
            raise ValueError(f"{recording_id}: duplicate native identity key")
        matched = fingering_keys.merge(
            native_keys,
            on=identity,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not matched["_merge"].eq("both").all():
            raise ValueError(f"{recording_id}: identity mismatch")
        if not matched["_velocity_fingering"].eq(
            matched["_velocity_native"]
        ).all():
            raise ValueError(f"{recording_id}: velocity mismatch")
        offset_column = result.columns.get_loc("offset_sec")
        result.iloc[
            matched["_result_position"].astype(int).to_numpy(),
            offset_column,
        ] = matched["_native_offset"].to_numpy()
        joined_rows.append(
            {
                **source_row.to_dict(),
                "fingering_row_count": len(fingering),
                "joined_row_count": len(matched),
                "identity_count_check": True,
                "onset_check": True,
                "pitch_check": True,
                "velocity_check": True,
                "validation_status": "exact_join_valid",
            }
        )

    joined_provenance = pd.DataFrame.from_records(joined_rows)
    complete = (
        len(joined_provenance) == len(expected_recordings)
        and not joined_provenance["recording_id"].duplicated().any()
        and set(joined_provenance["recording_id"]) == expected_recordings
        and int(joined_provenance["joined_row_count"].sum()) == len(result)
        and int(joined_provenance["fingering_row_count"].sum()) == len(result)
        and result["offset_sec"].notna().all()
        and joined_provenance["validation_status"].eq("exact_join_valid").all()
    )
    if not complete:
        raise ValueError("authoritative timing join is incomplete")
    return TimingJoin(
        notes=result,
        provenance=joined_provenance,
        complete=True,
    )

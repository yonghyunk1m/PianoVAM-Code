from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _load_tsv(path: Path) -> pd.DataFrame:
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if first.startswith("#"):
        columns = [c.strip().lower() for c in first.lstrip("# ").split("\t")]
        frame = pd.read_csv(
            path, sep="\t", comment="#", header=None, names=columns
        )
    else:
        frame = pd.read_csv(path, sep="\t")
        frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame


def _finger(value) -> int | None:
    try:
        finger = int(value)
    except (TypeError, ValueError):
        return None
    return finger if 1 <= finger <= 5 else None


def _hand(value) -> str | None:
    hand = str(value).strip().upper()
    return hand if hand in {"L", "R"} else None


def load_pianovam_notes(path: Path) -> pd.DataFrame:
    recordings: list[pd.DataFrame] = []
    for tsv_path in sorted(Path(path).glob("*.tsv")):
        frame = _load_tsv(tsv_path).reset_index(drop=True)
        required = {"onset", "note"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{tsv_path} missing columns: {sorted(missing)}")
        count = len(frame)
        note_idx = np.arange(count, dtype=np.int64)
        hand_source = frame.get("hand", pd.Series(pd.NA, index=frame.index))
        finger_source = frame.get("finger", pd.Series(pd.NA, index=frame.index))
        hands = hand_source.map(_hand)
        fingers = finger_source.map(_finger).where(hands.notna())
        pitches = pd.to_numeric(frame["note"], errors="raise").astype(np.int16)
        invalid_pitch = (pitches < 0) | (pitches > 127)
        if invalid_pitch.any():
            pitch = int(pitches.loc[invalid_pitch].iloc[0])
            raise ValueError(f"invalid MIDI pitch {pitch} in {tsv_path}")
        offsets = (
            pd.to_numeric(frame["key_offset"], errors="coerce")
            if "key_offset" in frame
            else pd.Series(np.nan, index=frame.index)
        )
        velocities = (
            pd.to_numeric(frame["velocity"], errors="coerce").astype("Int64")
            if "velocity" in frame
            else pd.Series(pd.NA, index=frame.index, dtype="Int64")
        )
        finger_ids = hands.str.cat(fingers.astype("Int64").astype("string"))
        finger_ids = finger_ids.where(hands.notna() & fingers.notna())
        recordings.append(
            pd.DataFrame(
                {
                    "recording_id": tsv_path.stem,
                    "note_idx": note_idx,
                    "note_id": [f"{tsv_path.stem}#{idx}" for idx in note_idx],
                    "onset_sec": pd.to_numeric(frame["onset"], errors="raise"),
                    "offset_sec": offsets,
                    "pitch": pitches,
                    "velocity": velocities,
                    "pred_hand": hands,
                    "pred_finger": fingers.astype("Int64"),
                    "pred_finger_id": finger_ids,
                    "source_path": str(tsv_path.resolve()),
                }
            )
        )
    result = pd.concat(recordings, ignore_index=True) if recordings else pd.DataFrame()
    if result.empty:
        raise ValueError(f"no TSV files found in {path}")
    if result["note_id"].duplicated().any():
        raise ValueError("duplicate canonical note IDs")
    return result


def load_ground_truth(module_path: Path) -> pd.DataFrame:
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location("fingering_audit_gt", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load ground truth module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for recording_id, labels in sorted(module.GT_MAP.items()):
        for note_idx, (hand, finger) in enumerate(labels):
            rows.append(
                {
                    "recording_id": recording_id,
                    "note_idx": note_idx,
                    "gt_hand": hand,
                    "gt_finger": int(finger),
                    "gt_finger_id": f"{hand}{int(finger)}",
                }
            )
    return pd.DataFrame(rows)


def attach_ground_truth(notes: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    gt_recordings = set(gt["recording_id"])
    eligible = notes[notes["recording_id"].isin(gt_recordings)]
    merged = gt.merge(
        eligible,
        on=["recording_id", "note_idx"],
        how="left",
        validate="one_to_one",
    )
    if merged["note_id"].isna().any():
        missing = merged.loc[merged["note_id"].isna(), ["recording_id", "note_idx"]]
        raise ValueError(f"ground truth exceeds available notes: {missing.to_dict('records')}")
    return merged


_PIG_HAND_MAP = {"0": "R", "1": "L"}


def _pig_dataset_root(root: Path) -> Path:
    root = Path(root)
    candidates = (root / "PianoFingeringDataset_v1.02", root)
    for candidate in candidates:
        if (candidate / "FingeringFiles").is_dir():
            return candidate
    raise FileNotFoundError(f"PIG FingeringFiles not found under {root}")


def _pig_finger_token(raw: str) -> tuple[int, int, tuple[int, ...]] | None:
    components: list[int] = []
    first_sign = 1
    for index, token in enumerate(str(raw).strip().split("_")):
        try:
            signed = int(token)
        except ValueError:
            return None
        finger = abs(signed)
        if not 1 <= finger <= 5:
            return None
        if index == 0:
            first_sign = -1 if signed < 0 else 1
        components.append(finger)
    if not components:
        return None
    return components[0], first_sign, tuple(components)


def load_pig_canonical(root: Path) -> pd.DataFrame:
    """Load all PIG annotations without discarding signed/compound semantics."""
    dataset_root = _pig_dataset_root(Path(root))
    rows: list[dict] = []
    pattern = re.compile(
        r"^(?P<piece>[^-]+)-(?P<performer>.+?)_fingering$", re.IGNORECASE
    )
    for source_path in sorted((dataset_root / "FingeringFiles").glob("*_fingering.txt")):
        match = pattern.match(source_path.stem)
        if not match:
            raise ValueError(f"unrecognized PIG filename: {source_path.name}")
        piece_id = match.group("piece").zfill(3)
        performer_id = match.group("performer")
        with source_path.open(encoding="utf-8") as stream:
            for source_line, line in enumerate(stream, start=1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                parts = line.split("\t")
                if len(parts) < 8:
                    raise ValueError(
                        f"{source_path}:{source_line}: expected at least 8 columns"
                    )
                parsed = _pig_finger_token(parts[7])
                hand = _PIG_HAND_MAP.get(parts[6].strip())
                if parsed is None or hand is None:
                    raise ValueError(
                        f"{source_path}:{source_line}: invalid hand/finger token"
                    )
                finger, finger_sign, components = parsed
                pitch = int(parts[4])
                if not 0 <= pitch <= 127:
                    raise ValueError(
                        f"{source_path}:{source_line}: invalid MIDI pitch {pitch}"
                    )
                note_index = int(parts[0])
                rows.append(
                    {
                        "pig_note_id": (
                            f"{piece_id}-{performer_id}#{note_index}@{source_line}"
                        ),
                        "piece_id": piece_id,
                        "performer_id": performer_id,
                        "note_index": note_index,
                        "onset_sec": float(parts[1]),
                        "offset_sec": float(parts[2]),
                        "note_name": parts[3],
                        "pitch": pitch,
                        "velocity": int(parts[5]),
                        "hand": hand,
                        "finger_token": parts[7].strip(),
                        "finger": finger,
                        "finger_sign": finger_sign,
                        "finger_components": components,
                        "compound_fingering": len(components) > 1,
                        "source_path": str(source_path.resolve()),
                        "source_line": source_line,
                    }
                )
    if not rows:
        raise ValueError(f"no PIG annotations found under {dataset_root}")
    result = pd.DataFrame.from_records(rows)
    if result["pig_note_id"].duplicated().any():
        raise ValueError("duplicate PIG canonical note IDs")
    return result

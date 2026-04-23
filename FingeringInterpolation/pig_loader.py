"""
PIG (Piano Fingering Dataset) v1.02 loader.

Dataset: https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/
Format: PianoFingeringDataset_v1.02/FingeringFiles/{piece_id}-{performer_id}_fingering.txt

File columns (tab-separated, first line is version comment):
  0: note_index
  1: onset_time (sec)
  2: offset_time (sec)
  3: note_name (e.g. C4)
  4: MIDI_pitch (0-127)
  5: velocity
  6: hand (0=Right, 1=Left)
  7: finger (1-5; negative = thumb-under/cross-over; compound e.g. "4_1")

Train/test split (following Saitō & Nakamura 2022):
  Test : Bach, Mozart, Chopin pieces
  Train: all other composers (Miscellaneous set)
"""
from __future__ import annotations
import csv
import os
import glob
from pathlib import Path

TEST_COMPOSERS = {"Bach", "Mozart", "Chopin"}
_HAND_MAP = {"0": "R", "1": "L"}


def _parse_finger(raw: str) -> int | None:
    """
    Parse finger token → int 1-5, or None if unparseable.
    Handles negatives (thumb-under) and compound tokens (e.g. '4_1').
    """
    raw = raw.strip()
    if not raw or raw == "0":
        return None
    # Take first finger for compound tokens like "4_1"
    token = raw.split("_")[0]
    try:
        f = abs(int(token))
        return f if 1 <= f <= 5 else None
    except ValueError:
        return None


def load_pig_file(path: str) -> dict[str, list[tuple[int, int]]]:
    """
    Load one PIG annotation file → per-hand (pitch, finger) lists sorted by onset.

    Returns:
        {"R": [(pitch, finger), ...], "L": [(pitch, finger), ...]}
    """
    notes: dict[str, list] = {"R": [], "L": []}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            try:
                onset  = float(parts[1])
                pitch  = int(parts[4])
                hand   = _HAND_MAP.get(parts[6].strip())
                finger = _parse_finger(parts[7])
            except (ValueError, IndexError):
                continue

            if hand is None or finger is None:
                continue
            if not (0 <= pitch <= 127):
                continue

            notes[hand].append((onset, pitch, finger))

    result = {}
    for hand, entries in notes.items():
        entries.sort(key=lambda x: x[0])
        result[hand] = [(pitch, finger) for _, pitch, finger in entries]

    return result


def _load_list(pig_root: str) -> dict[str, str]:
    """
    Load List.csv → {piece_id: composer}.
    Handles BOM in first column name.
    """
    list_path = os.path.join(pig_root, "List.csv")
    if not os.path.exists(list_path):
        return {}
    mapping = {}
    with open(list_path, encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("Id", "").strip().zfill(3)
            composer = row.get("Composer", "").strip()
            mapping[pid] = composer
    return mapping


def load_pig_split(
    pig_root: str,
    split: str = "train",
) -> dict[str, list[list[tuple[int, int]]]]:
    """
    Load PIG dataset split.

    Args:
        pig_root: Root of PIG dataset (contains PianoFingeringDataset_v1.02/).
        split:    "train" (Miscellaneous) or "test" (Bach, Mozart, Chopin).

    Returns:
        {"R": [[(pitch, finger), ...], ...], "L": [...]}
        One list per annotated file (piece × performer combination).
    """
    # Locate FingeringFiles directory
    fingering_dir = None
    for candidate in [
        os.path.join(pig_root, "PianoFingeringDataset_v1.02", "FingeringFiles"),
        os.path.join(pig_root, "FingeringFiles"),
    ]:
        if os.path.isdir(candidate):
            fingering_dir = candidate
            break
    if fingering_dir is None:
        raise FileNotFoundError(f"FingeringFiles not found under: {pig_root}")

    piece_composer = _load_list(os.path.join(pig_root, "PianoFingeringDataset_v1.02"))
    if not piece_composer:
        piece_composer = _load_list(pig_root)

    result: dict[str, list] = {"R": [], "L": []}

    for path in sorted(glob.glob(os.path.join(fingering_dir, "*_fingering.txt"))):
        stem = Path(path).stem  # e.g. "001-1_fingering"
        piece_id = stem.split("-")[0].zfill(3)

        composer = piece_composer.get(piece_id, "")
        is_test  = composer in TEST_COMPOSERS

        if split == "train" and is_test:
            continue
        if split == "test" and not is_test:
            continue

        hands = load_pig_file(path)
        for hand in ("R", "L"):
            if hands[hand]:
                result[hand].append(hands[hand])

    return result

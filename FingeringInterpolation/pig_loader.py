"""
PIG (Piano Fingering Dataset) loader.

Dataset: https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/
Format per note row:
    onset_time  offset_time  channel  pitch  onset_vel  offset_vel  finger

finger convention: 1-5 for each hand; channel 0=right, 1=left (or as annotated).
Each file contains one hand's notes. File naming: <id>-<performer>_<hand>.txt
  hand suffix: 'rh' = right, 'lh' = left.

Returns list of (pitch, finger) tuples per piece/hand for HMM training.
"""
import os
import glob
from pathlib import Path


# PIG file columns (0-indexed):
#   0: onset, 1: offset, 2: channel, 3: pitch, 4: onset_vel, 5: offset_vel, 6: finger
_PITCH_COL  = 3
_FINGER_COL = 6


def load_pig_file(path: str) -> list[tuple[int, int]]:
    """
    Load one PIG annotation file → list of (pitch, finger) sorted by onset.
    Skips header lines (starting with //).
    finger is 1-indexed (1=thumb, 5=pinky); negative = thumb-under / cross-over, treated as abs.
    """
    notes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                onset  = float(parts[0])
                pitch  = int(parts[_PITCH_COL])
                finger = abs(int(parts[_FINGER_COL]))  # abs: handle thumb-under
            except ValueError:
                continue
            if 1 <= finger <= 5 and 0 <= pitch <= 127:
                notes.append((onset, pitch, finger))

    notes.sort(key=lambda x: x[0])
    return [(pitch, finger) for _, pitch, finger in notes]


def load_pig_split(pig_root: str, split: str = "train") -> dict[str, list[list[tuple[int, int]]]]:
    """
    Load PIG dataset split.

    PIG directory layout (two common variants supported):
      Option A — separate hand dirs:
        pig_root/
          PianoFingeringDataset_v1.00/
            annotations/
              <id>-<performer>_<RH|LH>.txt

      Option B — flat:
        pig_root/
          *_rh.txt  /  *_lh.txt  (case-insensitive)

    Args:
        pig_root: Root of PIG dataset.
        split: "train" (Miscellaneous) or "test" (Bach, Mozart, Chopin).

    Returns:
        {"R": [piece1_notes, piece2_notes, ...], "L": [...]}
        where each piece_notes = [(pitch, finger), ...]
    """
    # Discover annotation files
    ann_files = []
    for pattern in ["**/*.txt", "*.txt"]:
        ann_files = glob.glob(os.path.join(pig_root, pattern), recursive=True)
        if ann_files:
            break

    if not ann_files:
        raise FileNotFoundError(f"No .txt annotation files found under: {pig_root}")

    # Split logic: test pieces are Bach (b), Mozart (m), Chopin (c) — by filename prefix
    TEST_PREFIXES = {"b", "m", "c"}  # first character of PIG piece id

    def is_test(filepath: str) -> bool:
        stem = Path(filepath).stem.lower()
        # e.g. "BACH_1_rh" or "B001-05_rh"
        return stem[0] in TEST_PREFIXES

    result = {"R": [], "L": []}
    for path in sorted(ann_files):
        stem = Path(path).stem.lower()
        if split == "train" and is_test(path):
            continue
        if split == "test" and not is_test(path):
            continue

        hand = None
        if stem.endswith("_rh") or stem.endswith("-rh"):
            hand = "R"
        elif stem.endswith("_lh") or stem.endswith("-lh"):
            hand = "L"
        else:
            continue  # skip files without hand suffix

        notes = load_pig_file(path)
        if notes:
            result[hand].append(notes)

    return result


def load_pig_all_hands(pig_root: str, split: str = "train") -> list[list[tuple[int, int]]]:
    """
    Load all hand sequences from a PIG split (R and L together).
    Used for training a single hand-agnostic model (not recommended; prefer per-hand).
    """
    data = load_pig_split(pig_root, split)
    return data["R"] + data["L"]

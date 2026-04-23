"""
Load PianoVAM fingerinfo pkl + MIDI → per-hand (pitch, finger|None) sequences.

fingerinfo encoding (internal):
  1-5  = Left hand, finger 1 (thumb) to 5 (pinky)
  6-10 = Right hand, finger 1 (thumb) to 5 (pinky)  [stored as 6+f-1]
  "Noinfo" or None = no finger detected
"""
import os
import glob
import pickle
from pathlib import Path

import pretty_midi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_finger(raw) -> tuple[str | None, int | None]:
    """
    Decode internal finger encoding → (hand, finger_1indexed) or (None, None).
    hand: "L" or "R"; finger: 1-5.
    """
    if raw == "Noinfo" or raw is None:
        return (None, None)
    if isinstance(raw, int) and 1 <= raw <= 10:
        if raw <= 5:
            return ("L", raw)
        return ("R", raw - 5)
    return (None, None)


def load_midi_pitches(midi_path: str) -> list[int]:
    """Load MIDI notes sorted by onset → list of MIDI pitches."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(
        [n for inst in midi.instruments for n in inst.notes],
        key=lambda n: n.start,
    )
    return [n.pitch for n in notes]


def load_fingerinfo(pkl_path: str) -> list:
    """Load a fingerinfo pkl file → raw list (int 1-10 or 'Noinfo')."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _pitch_based_hand(pitch: int, l_pitches: list[int], r_pitches: list[int]) -> str:
    """
    Assign a Noinfo note to L or R based on pitch proximity to labeled notes.
    Falls back to middle-C split (60) if no labeled notes exist.
    """
    if not l_pitches and not r_pitches:
        return "L" if pitch < 60 else "R"
    if not l_pitches:
        return "R"
    if not r_pitches:
        return "L"
    l_median = sorted(l_pitches)[len(l_pitches) // 2]
    r_median = sorted(r_pitches)[len(r_pitches) // 2]
    return "L" if abs(pitch - l_median) <= abs(pitch - r_median) else "R"


def fingerinfo_to_hand_sequences(
    pitches: list[int],
    fingerinfo: list,
    assign_noinfo: bool = True,
) -> dict[str, list[tuple[int, int, int | None]]]:
    """
    Split onset-sorted notes into per-hand sequences for HMM.

    Each entry is (note_idx, pitch, finger_or_None):
      - finger is 1-indexed within that hand (1=thumb, 5=pinky), or None if unlabeled.

    "Noinfo" notes (hand unknown):
      - If assign_noinfo=True: assigned to L or R by pitch proximity to labeled notes.
        These become unlabeled (finger=None) entries in the chosen hand's sequence.
      - If assign_noinfo=False: kept in a separate "Noinfo" list (not passed to HMM).

    Returns:
        {"L": [(idx, pitch, finger|None), ...],
         "R": [(idx, pitch, finger|None), ...],
         "Noinfo": [(idx, pitch, None), ...]}   # only populated if assign_noinfo=False
    """
    result: dict[str, list] = {"L": [], "R": [], "Noinfo": []}
    noinfo_notes = []

    for i, (pitch, raw) in enumerate(zip(pitches, fingerinfo)):
        hand, finger = _decode_finger(raw)
        if hand == "L":
            result["L"].append((i, pitch, finger))
        elif hand == "R":
            result["R"].append((i, pitch, finger))
        else:
            noinfo_notes.append((i, pitch))

    if assign_noinfo and noinfo_notes:
        l_pitches = [p for _, p, _ in result["L"]]
        r_pitches = [p for _, p, _ in result["R"]]
        for i, pitch in noinfo_notes:
            hand = _pitch_based_hand(pitch, l_pitches, r_pitches)
            result[hand].append((i, pitch, None))  # finger unknown
        # Re-sort by original note index to preserve onset order
        result["L"].sort(key=lambda x: x[0])
        result["R"].sort(key=lambda x: x[0])
    else:
        for i, pitch in noinfo_notes:
            result["Noinfo"].append((i, pitch, None))

    return result


def save_fingerinfo(fingerinfo: list, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(fingerinfo, f)


# ---------------------------------------------------------------------------
# Loading a full recording
# ---------------------------------------------------------------------------

def load_recording(
    midi_path: str,
    pkl_path: str,
) -> tuple[list[int], list, dict[str, list]]:
    """
    Load one PianoVAM recording.

    Returns:
        pitches    : list of MIDI pitches (onset-sorted)
        fingerinfo : raw fingerinfo list (int or "Noinfo")
        hand_seqs  : {"L": [(idx, pitch, finger|None), ...], "R": ..., "Noinfo": ...}
    """
    pitches = load_midi_pitches(midi_path)
    fingerinfo = load_fingerinfo(pkl_path)

    if len(pitches) != len(fingerinfo):
        raise ValueError(
            f"Length mismatch: pitches={len(pitches)}, fingerinfo={len(fingerinfo)}\n"
            f"  MIDI: {midi_path}\n  PKL:  {pkl_path}"
        )

    hand_seqs = fingerinfo_to_hand_sequences(pitches, fingerinfo)
    return pitches, fingerinfo, hand_seqs


# ---------------------------------------------------------------------------
# Rebuild fingerinfo after HMM interpolation
# ---------------------------------------------------------------------------

def apply_hmm_results(
    fingerinfo: list,
    hand_seqs: dict[str, list],
    hmm_results: dict[str, list[int]],
) -> list:
    """
    Merge HMM-predicted fingers back into a copy of fingerinfo.

    Args:
        fingerinfo  : original fingerinfo list
        hand_seqs   : {"L": [(idx, pitch, finger|None), ...], "R": ..., "Noinfo": ...}
        hmm_results : {"L": [finger_1indexed, ...], "R": [...]}
                      Only entries where original was None are actually updated.
    Returns:
        Updated fingerinfo list (new object, original not mutated).
    """
    updated = list(fingerinfo)

    for hand in ("L", "R"):
        if hand not in hmm_results:
            continue
        entries = hand_seqs[hand]
        preds   = hmm_results[hand]
        if len(entries) != len(preds):
            raise ValueError(
                f"HMM result length mismatch for hand {hand}: "
                f"entries={len(entries)}, preds={len(preds)}"
            )
        for (idx, _, orig_finger), pred_finger in zip(entries, preds):
            if orig_finger is None:  # only fill Noinfo slots
                internal = pred_finger if hand == "L" else pred_finger + 5
                updated[idx] = internal

    return updated


# ---------------------------------------------------------------------------
# Batch loader for training statistics (optional: get PianoVAM label coverage)
# ---------------------------------------------------------------------------

def count_noinfo(fingerinfo: list) -> tuple[int, int]:
    """Returns (n_noinfo, n_total)."""
    n_total  = len(fingerinfo)
    n_noinfo = sum(1 for f in fingerinfo if f == "Noinfo" or f is None)
    return n_noinfo, n_total

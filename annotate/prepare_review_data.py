#!/usr/bin/env python3
"""
Prepare a review bundle for the React-based annotate app.

Outputs:
  - annotate/public/data/notes.json
  - annotate/public/data/human_verdicts.json
  - annotate/public/videos/<video>
  - annotate/public/audio/<audio>

If a fingering TSV is provided, ManualCheck's rule-based hard-note selector is
applied and the resulting hard reasons become the annotate app's review queue.
Without a TSV, the script still creates a playable review bundle from MIDI
alone, but there is no hard-note prioritization.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mido import MidiFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from ManualCheck.hard_part_selector import (  # noqa: E402
    DEFAULT_RULES,
    RULE_DESCRIPTIONS,
    load_fingering_tsv,
    select_hard_parts,
)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

HARD_RULE_PRIORITY = {
    "impossible_fingering": 100,
    "noinfo_cluster": 96,
    "fast_jump": 92,
    "hand_overlap": 88,
    "rapid_alternation": 84,
    "stepwise_order_violation": 78,
    "noinfo": 72,
}

INT_TO_LABEL = {
    1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
    6: "R1", 7: "R2", 8: "R3", 9: "R4", 10: "R5",
}


def midi_to_note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def hand_finger_to_int(hand: str | None, finger: int | None) -> int | None:
    if hand not in {"L", "R"} or finger is None or not 1 <= finger <= 5:
        return None
    return finger if hand == "L" else finger + 5


def review_priority_for_reasons(reasons: list[str]) -> int | None:
    if not reasons:
        return None
    return max(HARD_RULE_PRIORITY.get(reason, 70) for reason in reasons)


def review_reason_for_reasons(reasons: list[str]) -> str | None:
    if not reasons:
        return None
    labels = [RULE_DESCRIPTIONS.get(reason, reason) for reason in reasons]
    return "ManualCheck rules: " + " | ".join(labels)


def ensure_asset(src: Path, dest_dir: Path, public_prefix: str, *, copy: bool) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            raise RuntimeError(f"Refusing to replace non-file asset target: {dest}")
    if copy:
        shutil.copy2(src, dest)
    else:
        dest.symlink_to(src.resolve())
    return f"/{public_prefix}/{src.name}"


def midi_notes_from_file(midi_path: Path) -> list[dict[str, Any]]:
    midi = MidiFile(str(midi_path))
    active: dict[int, list[tuple[float, int]]] = defaultdict(list)
    notes: list[dict[str, Any]] = []
    current_time = 0.0

    for msg in midi:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note].append((current_time, int(msg.velocity)))
            continue
        if msg.type not in {"note_off", "note_on"}:
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            continue
        if not active[msg.note]:
            continue
        onset_sec, velocity = active[msg.note].pop(0)
        offset_sec = current_time
        notes.append({
            "pitch": int(msg.note),
            "pitch_name": midi_to_note_name(int(msg.note)),
            "onset_sec": round(onset_sec, 6),
            "offset_sec": round(offset_sec, 6),
            "velocity": velocity,
        })

    notes.sort(key=lambda note: (note["onset_sec"], note["pitch"], note["offset_sec"]))
    return notes


def build_notes_from_midi(
    midi_path: Path,
    trial: str,
    piece: str,
    difficulty: str | None,
    video_path: str,
    audio_path: str,
) -> list[dict[str, Any]]:
    base_notes = midi_notes_from_file(midi_path)
    notes: list[dict[str, Any]] = []
    for global_idx, note in enumerate(base_notes):
        notes.append({
            "global_idx": global_idx,
            "trial": trial,
            "piece": piece,
            "difficulty": difficulty,
            "task": "manual_review",
            "note_idx": global_idx,
            "pitch": note["pitch"],
            "pitch_name": note["pitch_name"],
            "velocity": note["velocity"],
            "onset_sec": note["onset_sec"],
            "offset_sec": note["offset_sec"],
            "video_path": video_path,
            "audio_path": audio_path,
            "algorithm_status": "no_candidate",
            "algorithm_int": None,
            "algorithm_hand": None,
            "algorithm_finger": None,
            "algorithm_score": None,
            "imputed_int": None,
            "imputed_score": None,
            "candidates": [],
            "is_hard": False,
            "hard_reasons": [],
            "review_priority": None,
            "review_reason": None,
        })
    return notes


def build_notes_from_tsv(
    tsv_path: Path,
    trial: str,
    piece: str,
    difficulty: str | None,
    video_path: str,
    audio_path: str,
    enabled_rules: list[str],
) -> list[dict[str, Any]]:
    df = load_fingering_tsv(str(tsv_path))
    df = select_hard_parts(df, enabled_rules)

    notes: list[dict[str, Any]] = []
    for global_idx, row in enumerate(df.itertuples(index=False)):
        hand = str(getattr(row, "hand", "Noinfo") or "Noinfo").strip()
        finger = parse_optional_int(getattr(row, "finger_int", None))
        algorithm_int = hand_finger_to_int(hand, finger)
        hard_reasons = [reason for reason in str(getattr(row, "hard_reasons", "")).split(",") if reason]
        review_priority = review_priority_for_reasons(hard_reasons)
        onset_sec = parse_optional_float(getattr(row, "onset", None))
        offset_sec = parse_optional_float(getattr(row, "key_offset", None))
        if onset_sec is None:
            raise ValueError(f"TSV is missing onset for note index {global_idx}")
        if offset_sec is None or offset_sec < onset_sec:
            offset_sec = onset_sec + 0.5

        pitch = parse_optional_int(getattr(row, "note", None))
        if pitch is None:
            raise ValueError(f"TSV is missing MIDI pitch for note index {global_idx}")

        notes.append({
            "global_idx": global_idx,
            "trial": trial,
            "piece": piece,
            "difficulty": difficulty,
            "task": "manual_review",
            "note_idx": global_idx,
            "pitch": pitch,
            "pitch_name": midi_to_note_name(pitch),
            "velocity": parse_optional_int(getattr(row, "velocity", None)),
            "onset_sec": round(onset_sec, 6),
            "offset_sec": round(offset_sec, 6),
            "video_path": video_path,
            "audio_path": audio_path,
            "algorithm_status": "labeled" if algorithm_int else "no_candidate",
            "algorithm_int": algorithm_int,
            "algorithm_hand": "Left" if hand == "L" else "Right" if hand == "R" else None,
            "algorithm_finger": finger if algorithm_int else None,
            "algorithm_score": 1.0 if algorithm_int else None,
            "imputed_int": None,
            "imputed_score": None,
            "candidates": [],
            "is_hard": bool(getattr(row, "is_hard", False)),
            "hard_reasons": hard_reasons,
            "review_priority": review_priority,
            "review_reason": review_reason_for_reasons(hard_reasons),
        })

    return notes


def build_initial_verdicts(notes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    for note in notes:
        finger_int = note.get("algorithm_int")
        if not finger_int:
            continue
        note_id = f"{note['trial']}#{note['note_idx']}"
        verdicts[note_id] = {
            "finger_int": finger_int,
            "finger_label": INT_TO_LABEL[finger_int],
            "algorithm_int": finger_int,
            "algorithm_status": note.get("algorithm_status"),
            "source": "algo",
        }
    return verdicts


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare annotate review bundle")
    parser.add_argument("--midi", required=True, help="Source MIDI file")
    parser.add_argument("--video", required=True, help="Source video file")
    parser.add_argument("--audio", required=True, help="Source audio file")
    parser.add_argument("--tsv", default=None,
                        help="Optional fingering TSV with onset/note/hand/finger columns")
    parser.add_argument("--piece", default=None, help="Piece title shown in the UI")
    parser.add_argument("--difficulty", default=None, help="Optional difficulty tag")
    parser.add_argument("--trial", default=None, help="Recording/session id shown in the UI")
    parser.add_argument("--rules", default=",".join(DEFAULT_RULES),
                        help="Comma-separated ManualCheck rule names when --tsv is used")
    parser.add_argument("--public-root", default=str(Path(__file__).resolve().parent / "public"),
                        help="annotate public/ directory")
    parser.add_argument("--copy-media", action="store_true",
                        help="Copy media instead of creating symlinks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    midi_path = Path(args.midi).resolve()
    video_path_src = Path(args.video).resolve()
    audio_path_src = Path(args.audio).resolve()
    tsv_path = Path(args.tsv).resolve() if args.tsv else None
    public_root = Path(args.public_root).resolve()
    data_root = public_root / "data"

    trial = args.trial or (tsv_path.stem if tsv_path else midi_path.stem)
    piece = args.piece or trial
    enabled_rules = [rule.strip() for rule in args.rules.split(",") if rule.strip()]

    for path in [midi_path, video_path_src, audio_path_src]:
        if not path.exists():
            raise FileNotFoundError(path)
    if tsv_path and not tsv_path.exists():
        raise FileNotFoundError(tsv_path)

    video_path = ensure_asset(video_path_src, public_root / "videos", "videos", copy=args.copy_media)
    audio_path = ensure_asset(audio_path_src, public_root / "audio", "audio", copy=args.copy_media)

    if tsv_path:
        notes = build_notes_from_tsv(
            tsv_path,
            trial=trial,
            piece=piece,
            difficulty=args.difficulty,
            video_path=video_path,
            audio_path=audio_path,
            enabled_rules=enabled_rules,
        )
    else:
        notes = build_notes_from_midi(
            midi_path,
            trial=trial,
            piece=piece,
            difficulty=args.difficulty,
            video_path=video_path,
            audio_path=audio_path,
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": trial,
        "piece": piece,
        "difficulty": args.difficulty,
        "review_source": "ManualCheck" if tsv_path else "MIDI only",
        "rule_descriptions": RULE_DESCRIPTIONS,
        "enabled_rules": enabled_rules if tsv_path else [],
        "finger_int_encoding": {
            "1-5": "Left thumb..pinky",
            "6-10": "Right thumb..pinky",
            "0": "unsure",
            "-1": "no_press",
        },
        "notes": notes,
    }

    verdicts = build_initial_verdicts(notes)

    write_json(data_root / "notes.json", payload)
    write_json(data_root / "human_verdicts.json", verdicts)

    n_hard = sum(1 for note in notes if note.get("is_hard"))
    print(f"Prepared annotate bundle for '{trial}'")
    print(f"  notes:          {len(notes)}")
    print(f"  hard notes:     {n_hard}")
    print(f"  seeded verdicts:{len(verdicts)}")
    print(f"  notes.json:     {data_root / 'notes.json'}")
    print(f"  verdicts:       {data_root / 'human_verdicts.json'}")


if __name__ == "__main__":
    main()

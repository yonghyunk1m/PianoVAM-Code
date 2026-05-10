"""
Export per-note fingering from fingerinfo pkl + MIDI to TSV/JSON.
Run from project root: python PreProcessing/Fingering-Export/export_fingering.py
"""
import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import pretty_midi

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import config


def find_fingerinfo_pkl(pickles_dir, midi_basename):
    """
    Find fingerinfo or fingering pkl matching MIDI basename.
    Prefer: fingerinfo_*.pkl (direct per-note), else fingering_<basename>.pkl (keyhandlist).
    """
    for pattern in [
        f"fingerinfo_{midi_basename}.pkl",
        f"fingerinfo_{midi_basename}_*.pkl",
        f"fingerinfo_{midi_basename}.mp4_*.pkl",
        f"fingering_{midi_basename}.pkl",
    ]:
        candidates = glob.glob(os.path.join(pickles_dir, pattern))
        if candidates:
            return candidates[0]
    return None


def finger_to_standard(f):
    """
    Convert internal finger (1-10) to standard piano notation: L1-L5, R1-R5.
    Left: 1=thumb, 5=pinky. Right: 6->R1(thumb), 10->R5(pinky).
    """
    if f == "Noinfo" or f is None:
        return "Noinfo"
    if isinstance(f, int) and 1 <= f <= 10:
        if f <= 5:
            return f"L{f}"  # Left hand: thumb=1, pinky=5
        return f"R{f - 5}"   # Right hand: thumb=1, pinky=5
    return str(f)


def finger_to_hand_finger(f, swap_hands=False):
    """
    Convert internal finger (1-10) to (hand, finger) where hand in L/R, finger in 1-5.
    Returns ("L"|"R", 1-5) or ("Noinfo", None).
    swap_hands: If True, swap L<->R (for testing camera mirroring / convention mismatch).
    """
    if f == "Noinfo" or f is None:
        return ("Noinfo", None)
    if isinstance(f, int) and 1 <= f <= 10:
        if f <= 5:
            hand, finger = ("L", f)
        else:
            hand, finger = ("R", f - 5)
        if swap_hands:
            hand = "R" if hand == "L" else "L"
        return (hand, finger)
    return ("Noinfo", None)


def load_midi_notes_sorted(midi_path):
    """Load MIDI notes sorted by onset (matches fingerinfo index order)."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = [note for inst in midi.instruments for note in inst.notes]
    notes.sort(key=lambda n: n.start)
    return notes


def export_tsv(notes, fingerinfo, output_path, use_tsv_ref=False, tsv_path=None, finger_format="separate", swap_hands=False):
    """
    Export to TSV. finger_format: "combined" (L1,R5) or "separate" (hand, finger columns).
    """
    rows = []
    for i, n in enumerate(notes):
        raw = fingerinfo[i] if i < len(fingerinfo) else "Noinfo"
        if finger_format == "separate":
            hand, finger = finger_to_hand_finger(raw, swap_hands=swap_hands)
            finger_str = str(finger) if finger is not None else "Noinfo"
        else:
            hand, finger_str = None, finger_to_standard(raw)

        base = [n.start, n.end, n.end, n.pitch, int(n.velocity)]
        if use_tsv_ref and tsv_path and os.path.exists(tsv_path):
            import pandas as pd
            df = pd.read_csv(tsv_path, sep="\t", comment="#", header=None,
                             names=["onset", "key_offset", "frame_offset", "note", "velocity"])
            if i < len(df):
                row = df.iloc[i]
                base = [row["onset"], row["key_offset"], row["frame_offset"], n.pitch, int(n.velocity)]

        if finger_format == "separate":
            rows.append(base + [hand, finger_str])
        else:
            rows.append(base + [finger_str])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        if finger_format == "separate":
            f.write("onset\tkey_offset\tframe_offset\tnote\tvelocity\thand\tfinger\n")
        else:
            f.write("onset\tkey_offset\tframe_offset\tnote\tvelocity\tfinger\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def export_json(notes, fingerinfo, output_path, finger_format="separate", swap_hands=False):
    """Export to JSON. finger_format: "combined" (finger: L1) or "separate" (hand, finger)."""
    data = []
    for i, n in enumerate(notes):
        raw = fingerinfo[i] if i < len(fingerinfo) else "Noinfo"
        entry = {
            "onset": round(n.start, 6),
            "offset": round(n.end, 6),
            "pitch": int(n.pitch),
            "velocity": int(n.velocity),
        }
        if finger_format == "separate":
            hand, finger = finger_to_hand_finger(raw, swap_hands=swap_hands)
            entry["hand"] = hand
            entry["finger"] = finger if finger is not None else "Noinfo"
        else:
            entry["finger"] = finger_to_standard(raw)
        data.append(entry)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"notes": data}, f, indent=2)


def process_one(midi_path, pickles_dir, output_dir, tsv_dir, format_tsv, format_json, use_tsv_ref, finger_format="separate", use_miditotoken=False, swap_hands=False):
    """Process one MIDI file and export fingering."""
    basename = Path(midi_path).stem
    pkl_path = find_fingerinfo_pkl(pickles_dir, basename)
    if not pkl_path:
        return False, "no fingerinfo/fingering pkl"

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    notes = load_midi_notes_sorted(midi_path)

    # fingering_*.pkl = keyhandlist (need decide_fingering); fingerinfo_*.pkl = direct
    if Path(pkl_path).stem.startswith("fingerinfo_"):
        fingerinfo = pkl_data
    else:
        from decider_standalone import decide_fingering
        fps = 60
        if use_miditotoken:
            from miditotoken_standalone import miditotoken_from_path
            tokenlist = miditotoken_from_path(midi_path, fps)
            if tokenlist is None:
                return False, "miditotoken failed (install miditok, symusic)"
            tokenlist_copy = [list(t) for t in tokenlist]
        else:
            tokenlist = []
            for i, n in enumerate(notes):
                tokenlist.append([
                    int(n.start * fps),
                    max(0, min(87, n.pitch - 21)),
                    int(n.end * fps),
                    i,
                ])
            tokenlist_copy = [list(t) for t in tokenlist]
        fingerinfo, _ = decide_fingering(tokenlist_copy, pkl_data)

    if len(fingerinfo) != len(notes):
        return False, f"len mismatch: fingerinfo={len(fingerinfo)}, notes={len(notes)}"

    tsv_ref = os.path.join(tsv_dir, basename + ".tsv") if tsv_dir else None

    if format_tsv:
        out_tsv = os.path.join(output_dir, basename + ".tsv")
        export_tsv(notes, fingerinfo, out_tsv, use_tsv_ref=use_tsv_ref, tsv_path=tsv_ref, finger_format=finger_format, swap_hands=swap_hands)
    if format_json:
        out_json = os.path.join(output_dir, basename + ".json")
        export_json(notes, fingerinfo, out_json, finger_format=finger_format, swap_hands=swap_hands)

    return True, None


def main():
    parser = argparse.ArgumentParser(description="Export per-note fingering from fingerinfo + MIDI")
    parser.add_argument("--dataset-root", default=config.DATASET_ROOT, help="PianoVAM dataset root")
    parser.add_argument("--pickles-dir", default=None, help="Override fingering pickles directory (default: <dataset-root>/fingering_pickles)")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--format", choices=["tsv", "json", "both"], default="both")
    parser.add_argument("--finger-format", choices=["combined", "separate"], default="separate",
                       help="separate (default): hand and finger columns; combined: finger as L1,R5")
    parser.add_argument("--use-tsv-ref", action="store_true",
                       help="Use key_offset/frame_offset from existing TSV when available")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--files", type=str, default=None,
                       help="Comma-separated basenames to process (e.g. 2024-02-15_20-07-54)")
    parser.add_argument("--use-miditotoken", action="store_true",
                       help="Use miditok for frame alignment (requires miditok, symusic)")
    parser.add_argument("--swap-hands", action="store_true",
                       help="Swap L<->R (test: camera mirroring / L-R convention mismatch)")
    args = parser.parse_args()

    pickles_dir = args.pickles_dir or os.path.join(args.dataset_root, "fingering_pickles")
    midi_dir = os.path.join(args.dataset_root, "MIDI")
    tsv_dir = os.path.join(args.dataset_root, "TSV")
    output_dir = args.output_dir

    if not os.path.exists(pickles_dir):
        print(f"Error: fingering_pickles not found: {pickles_dir}")
        sys.exit(1)
    if not os.path.exists(midi_dir):
        print(f"Error: MIDI dir not found: {midi_dir}")
        sys.exit(1)

    if args.files:
        bases = [b.strip() for b in args.files.split(",")]
        midi_files = [os.path.join(midi_dir, b + ".mid") for b in bases]
        midi_files = [p for p in midi_files if os.path.exists(p)]
    else:
        midi_files = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
        if args.limit:
            midi_files = midi_files[: args.limit]

    format_tsv = args.format in ("tsv", "both")
    format_json = args.format in ("json", "both")

    ok = 0
    fail = 0
    total = len(midi_files)
    for idx, midi_path in enumerate(midi_files):
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"Processing {idx + 1}/{total}...")
        success, err = process_one(
            midi_path, pickles_dir, output_dir, tsv_dir,
            format_tsv, format_json, args.use_tsv_ref, args.finger_format, args.use_miditotoken, args.swap_hands
        )
        if success:
            ok += 1
        else:
            fail += 1
            print(f"Skip {Path(midi_path).name}: {err}")

    print(f"\nDone. OK: {ok}, Failed: {fail}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

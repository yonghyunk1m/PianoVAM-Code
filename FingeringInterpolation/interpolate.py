"""
Apply trained HMM to fill "Noinfo" fingers in PianoVAM fingerinfo pkls.

Reads fingerinfo_*.pkl + MIDI, runs constrained Viterbi per hand,
writes updated fingerinfo pkl and/or a fingering TSV.

Usage:
    # Single file
    python FingeringInterpolation/interpolate.py \\
        --midi PianoVAM_v1.0/MIDI/2024-09-04_16-13-44.mid \\
        --pkl  PianoVAM_v1.0/fingering_pickles/fingerinfo_2024-09-04_16-13-44_*.pkl

    # Batch (all recordings with fingerinfo pkls)
    python FingeringInterpolation/interpolate.py --batch \\
        --dataset-root PianoVAM_v1.0 \\
        --output-dir   PianoVAM_v1.0/Fingering_HMM
"""
import argparse
import glob
import os
import sys
from pathlib import Path

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))

from FingeringInterpolation.hmm import constrained_viterbi, load_model
from FingeringInterpolation.pianovam_loader import (
    load_recording,
    apply_hmm_results,
    save_fingerinfo,
    count_noinfo,
)
sys.path.insert(0, os.path.join(os.path.dirname(_DIR), "PreProcessing", "Fingering-Export"))
from export_fingering import export_tsv, load_midi_notes_sorted

DEFAULT_MODEL_DIR = os.path.join(_DIR, "models")


def find_fingerinfo_pkl(pickles_dir: str, basename: str) -> str | None:
    for pattern in [
        f"fingerinfo_{basename}_*.pkl",
        f"fingerinfo_{basename}.mp4_*.pkl",
    ]:
        candidates = glob.glob(os.path.join(pickles_dir, pattern))
        if candidates:
            return candidates[0]
    return None


def run_hmm_interpolation(
    pitches: list[int],
    fingerinfo: list,
    hand_seqs: dict,
    models: dict[str, dict],
) -> tuple[list, dict]:
    """
    Run constrained Viterbi for each hand.

    Returns:
        updated_fingerinfo : list with Noinfo filled in where possible
        stats              : {"L": (n_filled, n_total), "R": ...}
    """
    hmm_results = {}
    stats = {}

    for hand in ("L", "R"):
        entries = hand_seqs[hand]
        if not entries:
            continue
        model = models.get(hand)
        if model is None:
            print(f"  [WARN] No model for {hand} hand, skipping.")
            continue

        hand_pitches = [p for _, p, _ in entries]
        hand_labels  = [f for _, _, f in entries]   # finger or None
        n_noinfo = sum(1 for f in hand_labels if f is None)

        predicted = constrained_viterbi(hand_pitches, hand_labels, model)
        hmm_results[hand] = predicted
        stats[hand] = (n_noinfo, len(entries))

    updated = apply_hmm_results(fingerinfo, hand_seqs, hmm_results)
    return updated, stats


def process_one(
    midi_path: str,
    pkl_path: str,
    output_dir: str,
    models: dict[str, dict],
    write_pkl: bool = True,
    write_tsv: bool = True,
    tsv_ref_dir: str | None = None,
) -> dict:
    """Process one recording. Returns stats dict."""
    basename = Path(midi_path).stem

    try:
        pitches, fingerinfo, hand_seqs = load_recording(midi_path, pkl_path)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    n_noinfo_before, n_total = count_noinfo(fingerinfo)

    updated_fingerinfo, hand_stats = run_hmm_interpolation(
        pitches, fingerinfo, hand_seqs, models
    )

    n_noinfo_after, _ = count_noinfo(updated_fingerinfo)

    os.makedirs(output_dir, exist_ok=True)

    if write_pkl:
        out_pkl = os.path.join(output_dir, f"fingerinfo_{basename}_hmm.pkl")
        save_fingerinfo(updated_fingerinfo, out_pkl)

    if write_tsv:
        notes = load_midi_notes_sorted(midi_path)
        tsv_ref = None
        if tsv_ref_dir:
            candidate = os.path.join(tsv_ref_dir, basename + ".tsv")
            tsv_ref = candidate if os.path.exists(candidate) else None
        out_tsv = os.path.join(output_dir, basename + ".tsv")
        export_tsv(
            notes, updated_fingerinfo, out_tsv,
            use_tsv_ref=bool(tsv_ref), tsv_path=tsv_ref,
            finger_format="separate",
        )

    return {
        "status": "ok",
        "n_total": n_total,
        "n_noinfo_before": n_noinfo_before,
        "n_noinfo_after": n_noinfo_after,
        "n_filled": n_noinfo_before - n_noinfo_after,
        "hand_stats": hand_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="HMM fingering interpolation for PianoVAM")
    # Single-file mode
    parser.add_argument("--midi", help="Path to MIDI file")
    parser.add_argument("--pkl",  help="Path to fingerinfo pkl file")
    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Process all recordings in dataset-root")
    parser.add_argument("--dataset-root", default="PianoVAM_v1.0")
    parser.add_argument("--files", help="Comma-separated basenames to process")
    # Model
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"Directory with hmm_R.npz and hmm_L.npz (default: {DEFAULT_MODEL_DIR})")
    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <dataset-root>/Fingering_HMM)")
    parser.add_argument("--tsv-ref-dir", default=None,
                        help="Dir with reference TSVs for key_offset/frame_offset (optional)")
    parser.add_argument("--no-pkl", action="store_true", help="Skip writing pkl files")
    parser.add_argument("--no-tsv", action="store_true", help="Skip writing TSV files")
    args = parser.parse_args()

    # Load models
    models = {}
    for hand in ("R", "L"):
        model_path = os.path.join(args.model_dir, f"hmm_{hand}.npz")
        if os.path.exists(model_path):
            models[hand] = load_model(model_path)
            print(f"Loaded {hand}-hand model: {model_path}")
        else:
            print(f"[WARN] Model not found: {model_path}")

    if not models:
        print("Error: No models found. Run train.py first.")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.dataset_root, "Fingering_HMM")
    tsv_ref_dir = args.tsv_ref_dir or os.path.join(args.dataset_root, "TSV")

    # Collect files to process
    if args.batch or args.files:
        pickles_dir = os.path.join(args.dataset_root, "fingering_pickles")
        midi_dir    = os.path.join(args.dataset_root, "MIDI")

        if args.files:
            basenames = [b.strip() for b in args.files.split(",")]
        else:
            midi_files = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
            basenames  = [Path(p).stem for p in midi_files]

        jobs = []
        for base in basenames:
            midi_path = os.path.join(midi_dir, base + ".mid")
            pkl_path  = find_fingerinfo_pkl(pickles_dir, base)
            if not os.path.exists(midi_path):
                print(f"  Skip (no MIDI): {base}")
                continue
            if not pkl_path:
                print(f"  Skip (no fingerinfo pkl): {base}")
                continue
            jobs.append((midi_path, pkl_path))

    elif args.midi and args.pkl:
        jobs = [(args.midi, args.pkl)]
    else:
        parser.print_help()
        sys.exit(1)

    # Process
    total_filled = 0
    total_noinfo = 0
    ok = fail = 0

    for midi_path, pkl_path in jobs:
        base = Path(midi_path).stem
        result = process_one(
            midi_path, pkl_path, output_dir, models,
            write_pkl=not args.no_pkl,
            write_tsv=not args.no_tsv,
            tsv_ref_dir=tsv_ref_dir,
        )
        if result["status"] == "ok":
            ok += 1
            filled = result["n_filled"]
            before = result["n_noinfo_before"]
            total  = result["n_total"]
            total_filled += filled
            total_noinfo += before
            print(f"  {base}: filled {filled}/{before} Noinfo  "
                  f"({total-result['n_noinfo_after']}/{total} labeled)")
        else:
            fail += 1
            print(f"  FAIL {base}: {result['error']}")

    print(f"\n--- Done ---")
    print(f"  Processed: {ok} ok, {fail} failed")
    if total_noinfo > 0:
        print(f"  Filled {total_filled}/{total_noinfo} Noinfo notes "
              f"({100*total_filled/total_noinfo:.1f}%)")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()

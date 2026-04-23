"""
Train HMM models on PIG dataset and save weights.

Usage:
    python FingeringInterpolation/train.py --pig-root /path/to/PIG
    python FingeringInterpolation/train.py --pig-root /path/to/PIG --no-extended
"""
import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))

from FingeringInterpolation.hmm import (
    train_hmm, train_hmm_extended,
    train_hmm_1st, train_hmm_1st_extended,
    save_model,
)
from FingeringInterpolation.pig_loader import load_pig_split

DEFAULT_MODEL_DIR = os.path.join(_DIR, "models")


def main():
    parser = argparse.ArgumentParser(description="Train HMM on PIG dataset")
    parser.add_argument("--pig-root", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--no-extended", action="store_true",
                        help="Train pitch-only model (default: extended with timing)")
    args = parser.parse_args()

    extended = not args.no_extended
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading PIG training split from: {args.pig_root}")
    data = load_pig_split(args.pig_root, split="train")

    for hand in ("R", "L"):
        pieces = data[hand]
        if not pieces:
            print(f"  [WARN] No {hand}-hand pieces, skipping.")
            continue

        print(f"\n--- {hand} hand ({len(pieces)} sequences) ---")

        # 2nd-order model
        if extended:
            model_2nd = train_hmm_extended(pieces)
            path_2nd  = os.path.join(args.output_dir, f"hmm_{hand}.npz")
            print(f"  Training extended 2nd-order HMM...")
        else:
            simple_pieces = [[(p, f) for p, f, *_ in piece] for piece in pieces]
            model_2nd = train_hmm(simple_pieces)
            path_2nd  = os.path.join(args.output_dir, f"hmm_{hand}.npz")
            print(f"  Training pitch-only 2nd-order HMM...")
        save_model(model_2nd, path_2nd)
        print(f"  Saved → {path_2nd}")

        # 1st-order model (for entropy / model-recommended selection)
        if extended:
            model_1st = train_hmm_1st_extended(pieces)
            path_1st  = os.path.join(args.output_dir, f"hmm1st_{hand}.npz")
            print(f"  Training extended 1st-order HMM (for entropy)...")
        else:
            simple_pieces = [[(p, f) for p, f, *_ in piece] for piece in pieces]
            model_1st = train_hmm_1st(simple_pieces)
            path_1st  = os.path.join(args.output_dir, f"hmm1st_{hand}.npz")
            print(f"  Training pitch-only 1st-order HMM (for entropy)...")
        save_model(model_1st, path_1st)
        print(f"  Saved → {path_1st}")

    print("\nDone.")


if __name__ == "__main__":
    main()

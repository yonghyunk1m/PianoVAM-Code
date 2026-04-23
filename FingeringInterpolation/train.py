"""
Train 2nd-order HMM on PIG dataset and save model weights.

Usage:
    python FingeringInterpolation/train.py --pig-root /path/to/PIG
    python FingeringInterpolation/train.py --pig-root /path/to/PIG --output models/hmm.npz
"""
import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))

from FingeringInterpolation.hmm import train_hmm, save_model
from FingeringInterpolation.pig_loader import load_pig_split

DEFAULT_OUTPUT = os.path.join(_DIR, "models", "hmm_{hand}.npz")


def train_one_hand(pieces: list, hand: str, output_path: str) -> None:
    if not pieces:
        print(f"  [WARN] No {hand}-hand pieces found, skipping.")
        return
    print(f"  Training {hand}-hand HMM on {len(pieces)} piece-sequences...")
    model = train_hmm(pieces)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_model(model, output_path)
    print(f"  Saved → {output_path}")

    # Quick sanity: show most common finger per pitch range
    emit = model["emit"]  # (5, 128)
    top = emit.argmax(axis=0)  # best finger per pitch
    print(f"  Emission check (most likely finger per octave):")
    for oct_start in range(21, 109, 12):
        avg = emit[:, oct_start:oct_start+12].mean(axis=1)
        print(f"    MIDI {oct_start:3d}-{oct_start+11}: " +
              " ".join(f"f{i+1}={avg[i]:.3f}" for i in range(5)))


def main():
    parser = argparse.ArgumentParser(description="Train HMM on PIG dataset")
    parser.add_argument("--pig-root", required=True,
                        help="Root directory of PIG dataset")
    parser.add_argument("--output", default=None,
                        help="Output path template (use {hand} for L/R). "
                             f"Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"],
                        help="Which PIG split to train on (default: train)")
    args = parser.parse_args()

    output_template = args.output or DEFAULT_OUTPUT

    print(f"Loading PIG dataset from: {args.pig_root}")
    split = "train" if args.split != "test" else "test"
    data = load_pig_split(args.pig_root, split=split)

    for hand in ("R", "L"):
        pieces = data[hand]
        out_path = output_template.replace("{hand}", hand)
        print(f"\n--- {hand} hand ---")
        train_one_hand(pieces, hand, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

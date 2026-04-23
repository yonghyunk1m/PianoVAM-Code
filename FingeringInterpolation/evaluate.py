"""
Evaluate HMM fingering completion on PIG test set.

Reproduces Table from Saitō & Nakamura 2022:
  - Random selection (avg over 20 seeds)
  - Middle finger only
  - Model-recommended selection

Usage:
    python FingeringInterpolation/evaluate.py --pig-root /path/to/PIG
"""
import argparse
import os
import sys
import random
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))

from FingeringInterpolation.hmm import (
    train_hmm, constrained_viterbi, forward_backward, compute_entropy, load_model
)
from FingeringInterpolation.pig_loader import load_pig_split


# ---------------------------------------------------------------------------
# Labeling strategies
# ---------------------------------------------------------------------------

def random_selection(n: int, ratio: float, true_fingers: list[int], seed: int) -> list:
    rng = random.Random(seed)
    k = int(n * ratio)
    indices = rng.sample(range(n), k)
    labels = [None] * n
    for i in indices:
        labels[i] = true_fingers[i]
    return labels


def specific_finger_selection(true_fingers: list[int], target: set[int]) -> list:
    return [f if f in target else None for f in true_fingers]


def model_recommended_selection(
    pitches: list[int],
    base_labels: list,
    model: dict,
    n_additional: int,
    true_fingers: list[int],
) -> list:
    if n_additional <= 0:
        return list(base_labels)
    posterior = forward_backward(pitches, base_labels, model)
    entropy   = compute_entropy(posterior)
    for i, l in enumerate(base_labels):
        if l is not None:
            entropy[i] = -1.0
    top_indices = np.argsort(entropy)[::-1][:n_additional]
    new_labels = list(base_labels)
    for i in top_indices:
        new_labels[i] = true_fingers[i]
    return new_labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pieces(
    pieces: list[list[tuple[int, int]]],
    model: dict,
    strategy: str,
    ratio: float = 0.5,
    n_seeds: int = 20,
    target_fingers: set[int] | None = None,
) -> float:
    """Evaluate accuracy on unlabeled notes."""
    total_correct = 0
    total_notes   = 0

    seeds = list(range(n_seeds)) if strategy == "random" else [0]

    for seed in seeds:
        seed_correct = 0
        seed_total   = 0
        for piece in pieces:
            pitches      = [p for p, _ in piece]
            true_fingers = [f for _, f in piece]
            n = len(pitches)

            if strategy == "random":
                labels = random_selection(n, ratio, true_fingers, seed)
            elif strategy == "specific":
                tgt = target_fingers or {3}
                labels = specific_finger_selection(true_fingers, tgt)
            elif strategy == "model_rec":
                base  = specific_finger_selection(true_fingers, {3})
                n_add = max(0, int(n * ratio) - sum(l is not None for l in base))
                labels = model_recommended_selection(pitches, base, model, n_add, true_fingers)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            predicted = constrained_viterbi(pitches, labels, model)

            for pred, true, label in zip(predicted, true_fingers, labels):
                if label is None:
                    seed_correct += int(pred == true)
                    seed_total   += 1

        total_correct += seed_correct
        total_notes   += seed_total

    return total_correct / total_notes if total_notes > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate HMM on PIG test set")
    parser.add_argument("--pig-root", required=True)
    parser.add_argument("--model-dir", default=os.path.join(_DIR, "models"))
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="Labeling ratio for random/model-rec strategies")
    parser.add_argument("--seeds", type=int, default=20,
                        help="Random seeds for averaging (random strategy)")
    args = parser.parse_args()

    print("Loading PIG test split...")
    test_data = load_pig_split(args.pig_root, split="test")

    results = {}
    for hand in ("R", "L"):
        pieces = test_data[hand]
        if not pieces:
            print(f"  No {hand}-hand test pieces found.")
            continue

        model_path = os.path.join(args.model_dir, f"hmm_{hand}.npz")
        if not os.path.exists(model_path):
            print(f"  [WARN] Model not found: {model_path} — run train.py first")
            continue
        model = load_model(model_path)

        print(f"\n=== {hand} hand ({len(pieces)} test pieces) ===")
        strategies = [
            ("Random 50%",       "random",    {"ratio": args.ratio, "n_seeds": args.seeds}),
            ("Middle finger",    "specific",  {"target_fingers": {3}}),
            ("Mid+ModelRec 50%", "model_rec", {"ratio": args.ratio}),
        ]
        for name, strat, kwargs in strategies:
            acc = evaluate_pieces(pieces, model, strategy=strat, **kwargs)
            print(f"  {name:25s}: {acc*100:.2f}%")
            results[f"{hand}_{strat}"] = acc

    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"  {k}: {v*100:.2f}%")


if __name__ == "__main__":
    main()

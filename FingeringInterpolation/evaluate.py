"""
Evaluate HMM fingering completion on PIG test set.

Reproduces results from Saitō & Nakamura 2022:
  - Random selection (avg over 20 seeds)
  - Middle finger only
  - Middle finger + model-recommended selection (paper's key result, Fig.3(c)(d))

Key corrections vs naive implementation:
  - Accuracy = over ALL notes (labeled + unlabeled); labeled are correct by construction
  - Entropy for model-recommended selection uses 1st-order HMM (paper Section 2.3)
  - Paper target: R=0 ~67%, R=40% random ~88%, middle+model_rec ~94%

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
    constrained_viterbi, constrained_viterbi_extended,
    forward_backward_1st, forward_backward_1st_extended,
    compute_entropy, load_model,
)
from FingeringInterpolation.pig_loader import load_pig_split


# ---------------------------------------------------------------------------
# Labeling strategies
# ---------------------------------------------------------------------------

def random_selection(n: int, ratio: float, true_fingers: list[int], seed: int) -> list:
    rng = random.Random(seed)
    k = int(n * ratio)
    indices = set(rng.sample(range(n), k))
    return [true_fingers[i] if i in indices else None for i in range(n)]


def specific_finger_selection(true_fingers: list[int], target: set[int]) -> list:
    return [f if f in target else None for f in true_fingers]


def model_recommended_selection(
    pitches: list[int],
    onsets: list[float],
    base_labels: list,
    model_1st: dict,
    n_additional: int,
    true_fingers: list[int],
) -> list:
    """
    Select additional notes by highest entropy under 1st-order HMM.
    Paper: "運指の不定性の評価には1次のHMMを用いる"
    Uses extended model if timing info available.
    """
    if n_additional <= 0:
        return list(base_labels)
    if model_1st.get("type") == "extended":
        posterior = forward_backward_1st_extended(pitches, onsets, base_labels, model_1st)
    else:
        posterior = forward_backward_1st(pitches, base_labels, model_1st)
    entropy = compute_entropy(posterior)
    for i, l in enumerate(base_labels):
        if l is not None:
            entropy[i] = -1.0
    top_indices = np.argsort(entropy)[::-1][:n_additional]
    new_labels = list(base_labels)
    for i in top_indices:
        new_labels[i] = true_fingers[i]
    return new_labels


# ---------------------------------------------------------------------------
# Evaluation (paper definition: accuracy over ALL notes)
# ---------------------------------------------------------------------------

def evaluate_pieces(
    pieces: list,
    model_2nd: dict,
    model_1st: dict,
    strategy: str,
    ratio: float = 0.5,
    n_seeds: int = 20,
    target_fingers: set[int] | None = None,
) -> tuple[float, float]:
    """
    Returns (accuracy_all_notes, labeling_ratio_actual).
    Paper definition: accuracy over ALL notes (labeled trivially correct).
    Supports both pitch-only (pitch, finger) and extended (pitch, finger, onset) pieces.
    """
    total_correct = total_notes = total_labeled = 0
    is_extended   = model_2nd.get("type") == "extended"
    seeds = list(range(n_seeds)) if strategy == "random" else [0]

    for seed in seeds:
        for piece in pieces:
            if is_extended:
                pitches      = [p for p, _, _ in piece]
                true_fingers = [f for _, f, _ in piece]
                onsets       = [t for _, _, t in piece]
            else:
                pitches      = [p for p, _ in piece]
                true_fingers = [f for _, f in piece]
                onsets       = list(range(len(pitches)))  # dummy
            n = len(pitches)

            if strategy == "random":
                labels = random_selection(n, ratio, true_fingers, seed)
            elif strategy == "specific":
                tgt    = target_fingers or {3}
                labels = specific_finger_selection(true_fingers, tgt)
            elif strategy == "model_rec":
                base  = [None] * n
                n_add = int(n * ratio)
                labels = model_recommended_selection(
                    pitches, onsets, base, model_1st, n_add, true_fingers)
            elif strategy == "middle_then_rec":
                base   = specific_finger_selection(true_fingers, {3})
                n_base = sum(l is not None for l in base)
                n_add  = max(0, int(n * ratio) - n_base)
                labels = model_recommended_selection(
                    pitches, onsets, base, model_1st, n_add, true_fingers)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if is_extended:
                predicted = constrained_viterbi_extended(pitches, onsets, labels, model_2nd)
            else:
                predicted = constrained_viterbi(pitches, labels, model_2nd)

            for pred, true in zip(predicted, true_fingers):
                total_correct += int(pred == true)
                total_notes   += 1
            total_labeled += sum(l is not None for l in labels)

    acc   = total_correct / total_notes if total_notes > 0 else 0.0
    r_act = total_labeled / total_notes if total_notes > 0 else 0.0
    return acc, r_act


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate HMM on PIG test set")
    parser.add_argument("--pig-root", required=True)
    parser.add_argument("--model-dir", default=os.path.join(_DIR, "models"))
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    print("Loading PIG test split...")
    test_data = load_pig_split(args.pig_root, split="test")

    results = {}
    for hand in ("R", "L"):
        pieces_test = test_data[hand]
        if not pieces_test:
            continue

        model_2nd_path = os.path.join(args.model_dir, f"hmm_{hand}.npz")
        model_1st_path = os.path.join(args.model_dir, f"hmm1st_{hand}.npz")

        if not os.path.exists(model_2nd_path):
            print(f"  [WARN] 2nd-order model not found: {model_2nd_path}")
            continue

        model_2nd = load_model(model_2nd_path)

        if not os.path.exists(model_1st_path):
            print(f"  [WARN] 1st-order model not found: {model_1st_path} — run train.py first")
            continue
        model_1st = load_model(model_1st_path)

        print(f"\n=== {hand} hand ({len(pieces_test)} test sequences) ===")
        strategies = [
            ("R=0 (no labels)",      "random",         {"ratio": 0.0,        "n_seeds": 1}),
            ("Random 40%",           "random",          {"ratio": 0.4,        "n_seeds": args.seeds}),
            ("Random 50%",           "random",          {"ratio": args.ratio, "n_seeds": args.seeds}),
            ("Middle finger only",   "specific",        {"target_fingers": {3}}),
            ("Model-rec 50%",        "model_rec",       {"ratio": args.ratio}),
            ("Middle+ModelRec 50%",  "middle_then_rec", {"ratio": args.ratio}),
        ]
        for name, strat, kwargs in strategies:
            acc, r_act = evaluate_pieces(
                pieces_test, model_2nd, model_1st, strategy=strat, **kwargs)
            print(f"  {name:25s}: A={acc*100:.1f}%  R={r_act*100:.1f}%")
            results[f"{hand}_{strat}"] = acc

    print("\n--- Summary (paper targets: R=40% random ~88%, middle+rec ~94%) ---")
    for k, v in results.items():
        print(f"  {k}: {v*100:.1f}%")


if __name__ == "__main__":
    main()

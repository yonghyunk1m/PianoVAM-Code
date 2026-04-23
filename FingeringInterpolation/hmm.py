"""
2nd-order HMM for piano fingering completion (Saitō & Nakamura 2022).

Trains on PIG dataset, applies constrained Viterbi to fill "Noinfo" gaps.
All operations are per-hand (L/R processed separately).
"""
import numpy as np

N_FINGERS = 5   # fingers 1-5 (thumb to pinky)
N_PITCHES = 128  # MIDI pitch range
EPS = 1e-10
LOG_ZERO = -1e18


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hmm(pieces: list[list[tuple[int, int]]]) -> dict:
    """
    Train 2nd-order HMM from training pieces.

    Args:
        pieces: List of pieces. Each piece is a list of (pitch, finger) tuples,
                finger is 1-indexed (1=thumb, 5=pinky).
    Returns:
        dict with keys:
            trans  (5, 5, 5): P(fn | fn-2, fn-1)
            emit   (5, 128) : P(pitch | finger)
            init   (5, 5)   : P(f0, f1)
    """
    trans = np.zeros((N_FINGERS, N_FINGERS, N_FINGERS)) + EPS
    emit  = np.zeros((N_FINGERS, N_PITCHES)) + EPS
    init  = np.zeros((N_FINGERS, N_FINGERS)) + EPS

    for piece in pieces:
        if len(piece) < 2:
            continue
        pitches = [p for p, _ in piece]
        fingers = [f - 1 for _, f in piece]  # 0-indexed internally

        if len(fingers) >= 2:
            init[fingers[0], fingers[1]] += 1

        for n in range(2, len(fingers)):
            trans[fingers[n-2], fingers[n-1], fingers[n]] += 1

        for pitch, finger in zip(pitches, fingers):
            emit[finger, pitch] += 1

    trans /= trans.sum(axis=2, keepdims=True)
    emit  /= emit.sum(axis=1, keepdims=True)
    init  /= init.sum()

    return {"trans": trans, "emit": emit, "init": init}


# ---------------------------------------------------------------------------
# Constrained Viterbi (core inference)
# ---------------------------------------------------------------------------

def constrained_viterbi(
    pitches: list[int],
    labels: list[int | None],
    model: dict,
) -> list[int]:
    """
    2nd-order HMM constrained Viterbi.

    Args:
        pitches: MIDI pitches, length N.
        labels:  Finger (1-5) or None for unlabeled notes, length N.
        model:   dict from train_hmm.
    Returns:
        Predicted finger sequence (1-indexed), length N.
    """
    N = len(pitches)
    if N == 0:
        return []

    trans = model["trans"]   # (F, F, F)
    emit  = model["emit"]    # (F, 128)
    init  = model["init"]    # (F, F)
    F = N_FINGERS

    def log_emit(n: int, f: int) -> float:
        if labels[n] is not None:
            return 0.0 if f == labels[n] - 1 else LOG_ZERO
        return np.log(emit[f, pitches[n]] + EPS)

    if N == 1:
        if labels[0] is not None:
            return [labels[0]]
        best = int(np.argmax([emit[f, pitches[0]] for f in range(F)]))
        return [best + 1]

    # dp[f0, f1] = best log-prob of sequence ending with (..., f0, f1)
    dp = np.full((F, F), LOG_ZERO)
    for f0 in range(F):
        for f1 in range(F):
            dp[f0, f1] = (
                np.log(init[f0, f1] + EPS)
                + log_emit(0, f0)
                + log_emit(1, f1)
            )

    # backpointer[step][f1, f2] = best f0
    history = []
    for n in range(2, N):
        new_dp = np.full((F, F), LOG_ZERO)
        new_bp = np.full((F, F), -1, dtype=np.int32)
        log_trans = np.log(trans + EPS)   # (F, F, F)

        for f1 in range(F):
            le_cache = np.array([log_emit(n, f2) for f2 in range(F)])
            for f2 in range(F):
                le = le_cache[f2]
                if le == LOG_ZERO:
                    continue
                scores = dp[:, f1] + log_trans[:, f1, f2]
                best_f0 = int(np.argmax(scores))
                new_dp[f1, f2] = scores[best_f0] + le
                new_bp[f1, f2] = best_f0

        history.append(new_bp)
        dp = new_dp

    # Backtrack from best terminal state
    best_last = np.unravel_index(np.argmax(dp), dp.shape)
    f_seq = list(best_last)  # [f_{N-2}, f_{N-1}]

    for bp_step in reversed(history):
        f0 = bp_step[f_seq[0], f_seq[1]]  # look up current front pair
        f_seq.insert(0, f0)

    return [f + 1 for f in f_seq]  # back to 1-indexed


# ---------------------------------------------------------------------------
# Forward-backward for posterior / entropy
# ---------------------------------------------------------------------------

def forward_backward(
    pitches: list[int],
    labels: list[int | None],
    model: dict,
) -> np.ndarray:
    """
    Compute per-note posterior P(fn | all pitches, labels).

    Returns:
        posterior: shape (N, 5)
    """
    N = len(pitches)
    F = N_FINGERS
    trans = model["trans"]
    emit  = model["emit"]
    init  = model["init"]

    def get_emit(n: int, f: int) -> float:
        if labels[n] is not None:
            return 1.0 if f == labels[n] - 1 else 0.0
        return float(emit[f, pitches[n]])

    # alpha[n, f_prev, f_curr]
    alpha = np.zeros((N, F, F))
    for f0 in range(F):
        for f1 in range(F):
            alpha[1, f0, f1] = init[f0, f1] * get_emit(0, f0) * get_emit(1, f1)
    s = alpha[1].sum()
    if s > 0:
        alpha[1] /= s

    for n in range(2, N):
        for f1 in range(F):
            for f2 in range(F):
                e = get_emit(n, f2)
                alpha[n, f1, f2] = e * np.sum(alpha[n-1, :, f1] * trans[:, f1, f2])
        s = alpha[n].sum()
        if s > 0:
            alpha[n] /= s

    posterior = np.zeros((N, F))
    for n in range(N):
        posterior[n] = alpha[n].sum(axis=0)
        s = posterior[n].sum()
        if s > 0:
            posterior[n] /= s

    return posterior


def compute_entropy(posterior: np.ndarray) -> np.ndarray:
    """Shannon entropy per note. Shape (N,) → (N,)."""
    p = np.clip(posterior, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: dict, path: str) -> None:
    np.savez(path, **model)


def load_model(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}

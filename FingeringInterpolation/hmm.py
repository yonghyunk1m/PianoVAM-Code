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

# Extended model: pitch-interval and IOI bins
_PITCH_DELTA_MIN = -24
_PITCH_DELTA_MAX = +24
N_PITCH_DELTAS   = _PITCH_DELTA_MAX - _PITCH_DELTA_MIN + 1  # 49
_IOI_BOUNDARIES  = [0.1, 0.3, 0.6]   # seconds: very fast / fast / medium / slow
N_IOI_BINS       = len(_IOI_BOUNDARIES) + 1  # 4


def _pitch_delta_idx(delta: int) -> int:
    return max(0, min(N_PITCH_DELTAS - 1, delta - _PITCH_DELTA_MIN))


def _ioi_bin(ioi_sec: float) -> int:
    for i, b in enumerate(_IOI_BOUNDARIES):
        if ioi_sec < b:
            return i
    return N_IOI_BINS - 1


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
# Extended model: pitch-interval + IOI emissions (matches paper [18,19])
# ---------------------------------------------------------------------------

def train_hmm_extended(pieces: list[list[tuple[int, int, float]]]) -> dict:
    """
    Train extended 2nd-order HMM using pitch interval and IOI.

    Args:
        pieces: List of pieces. Each piece is a list of
                (pitch, finger, onset_time) tuples, finger 1-indexed.
    Returns:
        dict with keys:
            trans       (5, 5, 5) : P(fn | fn-2, fn-1)
            emit0       (5, 128)  : P(pitch | f) for the first note only
            init        (5, 5)   : P(f0, f1)
            pitch_emit  (5, 5, 49): P(Δpitch | f_prev, f_curr)
            ioi_emit    (5, 5, 4) : P(IOI_bin | f_prev, f_curr)
    """
    trans      = np.zeros((N_FINGERS, N_FINGERS, N_FINGERS)) + EPS
    emit0      = np.zeros((N_FINGERS, N_PITCHES)) + EPS
    init       = np.zeros((N_FINGERS, N_FINGERS)) + EPS
    pitch_emit = np.zeros((N_FINGERS, N_FINGERS, N_PITCH_DELTAS)) + EPS
    ioi_emit   = np.zeros((N_FINGERS, N_FINGERS, N_IOI_BINS)) + EPS

    for piece in pieces:
        if len(piece) < 2:
            continue
        pitches = [p for p, _, _ in piece]
        fingers = [f - 1 for _, f, _ in piece]
        onsets  = [t for _, _, t in piece]

        if len(fingers) >= 2:
            init[fingers[0], fingers[1]] += 1

        emit0[fingers[0], pitches[0]] += 1

        for n in range(1, len(fingers)):
            fp = fingers[n - 1]
            fc = fingers[n]
            dp  = pitches[n] - pitches[n - 1]
            ioi = onsets[n]  - onsets[n - 1]
            pitch_emit[fp, fc, _pitch_delta_idx(dp)] += 1
            ioi_emit  [fp, fc, _ioi_bin(ioi)]        += 1

        for n in range(2, len(fingers)):
            trans[fingers[n-2], fingers[n-1], fingers[n]] += 1

    trans      /= trans.sum(axis=2, keepdims=True)
    emit0      /= emit0.sum(axis=1, keepdims=True)
    init       /= init.sum()
    pitch_emit /= pitch_emit.sum(axis=2, keepdims=True)
    ioi_emit   /= ioi_emit.sum(axis=2, keepdims=True)

    return {
        "trans": trans, "emit0": emit0, "init": init,
        "pitch_emit": pitch_emit, "ioi_emit": ioi_emit,
        "type": "extended",
    }


def constrained_viterbi_extended(
    pitches: list[int],
    onsets:  list[float],
    labels:  list[int | None],
    model:   dict,
) -> list[int]:
    """
    Constrained Viterbi for extended HMM (pitch interval + IOI emissions).

    Args:
        pitches: MIDI pitches, length N.
        onsets:  Onset times in seconds, length N.
        labels:  Finger (1-5) or None, length N.
        model:   dict from train_hmm_extended.
    Returns:
        Predicted finger sequence (1-indexed), length N.
    """
    N = len(pitches)
    if N == 0:
        return []

    trans      = model["trans"]       # (F, F, F)
    emit0      = model["emit0"]       # (F, 128)
    init       = model["init"]        # (F, F)
    pitch_emit = model["pitch_emit"]  # (F, F, 49)
    ioi_emit   = model["ioi_emit"]    # (F, F, 4)
    F = N_FINGERS

    log_trans      = np.log(trans + EPS)
    log_pitch_emit = np.log(pitch_emit + EPS)
    log_ioi_emit   = np.log(ioi_emit + EPS)
    log_emit0      = np.log(emit0 + EPS)

    def log_emit_first(n: int, f: int) -> float:
        """Emission for position 0 — absolute pitch."""
        if labels[n] is not None:
            return 0.0 if f == labels[n] - 1 else LOG_ZERO
        return log_emit0[f, pitches[n]]

    def log_emit_pair(n: int, fp: int, fc: int) -> float:
        """Emission for n≥1 — pitch interval + IOI conditioned on (fp, fc)."""
        if labels[n] is not None:
            return 0.0 if fc == labels[n] - 1 else LOG_ZERO
        dp  = pitches[n] - pitches[n - 1]
        ioi = onsets[n]  - onsets[n - 1]
        return (log_pitch_emit[fp, fc, _pitch_delta_idx(dp)]
                + log_ioi_emit [fp, fc, _ioi_bin(ioi)])

    if N == 1:
        if labels[0] is not None:
            return [labels[0]]
        return [int(np.argmax(log_emit0[:, pitches[0]])) + 1]

    # Initialise for positions 0 and 1
    dp = np.full((F, F), LOG_ZERO)
    for f0 in range(F):
        le0 = log_emit_first(0, f0)
        if le0 == LOG_ZERO:
            continue
        for f1 in range(F):
            le1 = log_emit_pair(1, f0, f1)
            dp[f0, f1] = np.log(init[f0, f1] + EPS) + le0 + le1

    history = []
    for n in range(2, N):
        new_dp = np.full((F, F), LOG_ZERO)
        new_bp = np.full((F, F), -1, dtype=np.int32)

        for f1 in range(F):
            for f2 in range(F):
                le = log_emit_pair(n, f1, f2)
                if le == LOG_ZERO:
                    continue
                scores  = dp[:, f1] + log_trans[:, f1, f2]
                best_f0 = int(np.argmax(scores))
                new_dp[f1, f2] = scores[best_f0] + le
                new_bp[f1, f2] = best_f0

        history.append(new_bp)
        dp = new_dp

    best_last = np.unravel_index(np.argmax(dp), dp.shape)
    f_seq = list(best_last)
    for bp_step in reversed(history):
        f_seq.insert(0, int(bp_step[f_seq[0], f_seq[1]]))

    return [f + 1 for f in f_seq]


def train_hmm_1st_extended(pieces: list[list[tuple[int, int, float]]]) -> dict:
    """1st-order extended HMM for entropy estimation."""
    trans      = np.zeros((N_FINGERS, N_FINGERS)) + EPS
    emit0      = np.zeros((N_FINGERS, N_PITCHES)) + EPS
    init       = np.zeros(N_FINGERS) + EPS
    pitch_emit = np.zeros((N_FINGERS, N_FINGERS, N_PITCH_DELTAS)) + EPS
    ioi_emit   = np.zeros((N_FINGERS, N_FINGERS, N_IOI_BINS)) + EPS

    for piece in pieces:
        if not piece:
            continue
        pitches = [p for p, _, _ in piece]
        fingers = [f - 1 for _, f, _ in piece]
        onsets  = [t for _, _, t in piece]

        init[fingers[0]]  += 1
        emit0[fingers[0], pitches[0]] += 1
        for n in range(1, len(fingers)):
            fp, fc = fingers[n-1], fingers[n]
            trans[fp, fc] += 1
            dp  = pitches[n] - pitches[n-1]
            ioi = onsets[n]  - onsets[n-1]
            pitch_emit[fp, fc, _pitch_delta_idx(dp)] += 1
            ioi_emit  [fp, fc, _ioi_bin(ioi)]        += 1

    trans      /= trans.sum(axis=1, keepdims=True)
    emit0      /= emit0.sum(axis=1, keepdims=True)
    init       /= init.sum()
    pitch_emit /= pitch_emit.sum(axis=2, keepdims=True)
    ioi_emit   /= ioi_emit.sum(axis=2, keepdims=True)

    return {
        "trans": trans, "emit0": emit0, "init": init,
        "pitch_emit": pitch_emit, "ioi_emit": ioi_emit,
        "order": 1, "type": "extended",
    }


def forward_backward_1st_extended(
    pitches: list[int],
    onsets:  list[float],
    labels:  list[int | None],
    model:   dict,
) -> np.ndarray:
    """Forward-backward for 1st-order extended HMM → posterior (N, 5)."""
    N = len(pitches)
    F = N_FINGERS
    trans      = model["trans"]
    emit0      = model["emit0"]
    pitch_emit = model["pitch_emit"]
    ioi_emit   = model["ioi_emit"]

    def get_emit0(n: int, f: int) -> float:
        if labels[n] is not None:
            return 1.0 if f == labels[n] - 1 else 0.0
        return float(emit0[f, pitches[n]])

    def get_emit_pair(n: int, fp: int, fc: int) -> float:
        if labels[n] is not None:
            return 1.0 if fc == labels[n] - 1 else 0.0
        dp  = pitches[n] - pitches[n - 1]
        ioi = onsets[n]  - onsets[n - 1]
        return (float(pitch_emit[fp, fc, _pitch_delta_idx(dp)])
                * float(ioi_emit [fp, fc, _ioi_bin(ioi)]))

    # alpha[n, f] — marginalised over previous finger
    alpha = np.zeros((N, F))
    for f in range(F):
        alpha[0, f] = model["init"][f] * get_emit0(0, f)
    s = alpha[0].sum()
    if s > 0:
        alpha[0] /= s

    for n in range(1, N):
        for fc in range(F):
            alpha[n, fc] = sum(
                alpha[n-1, fp] * trans[fp, fc] * get_emit_pair(n, fp, fc)
                for fp in range(F)
            )
        s = alpha[n].sum()
        if s > 0:
            alpha[n] /= s

    beta = np.ones((N, F))
    for n in range(N - 2, -1, -1):
        for fp in range(F):
            beta[n, fp] = sum(
                trans[fp, fc] * get_emit_pair(n+1, fp, fc) * beta[n+1, fc]
                for fc in range(F)
            )
        s = beta[n].sum()
        if s > 0:
            beta[n] /= s

    posterior = alpha * beta
    row_sums  = posterior.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return posterior / row_sums


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
# 1st-order HMM (used for entropy / model-recommended selection)
# Paper: "運指の不定性の評価には1次のHMMを用いる"
# (For entropy estimation, use 1st-order HMM)
# ---------------------------------------------------------------------------

def train_hmm_1st(pieces: list[list[tuple[int, int]]]) -> dict:
    """
    Train 1st-order HMM. Same interface as train_hmm but with
    trans shape (5, 5): P(fn | fn-1).
    """
    trans = np.zeros((N_FINGERS, N_FINGERS)) + EPS
    emit  = np.zeros((N_FINGERS, N_PITCHES)) + EPS
    init  = np.zeros(N_FINGERS) + EPS

    for piece in pieces:
        if not piece:
            continue
        pitches = [p for p, _ in piece]
        fingers = [f - 1 for _, f in piece]

        init[fingers[0]] += 1
        for n in range(1, len(fingers)):
            trans[fingers[n-1], fingers[n]] += 1
        for pitch, finger in zip(pitches, fingers):
            emit[finger, pitch] += 1

    trans /= trans.sum(axis=1, keepdims=True)
    emit  /= emit.sum(axis=1, keepdims=True)
    init  /= init.sum()

    return {"trans": trans, "emit": emit, "init": init, "order": 1}


def forward_backward_1st(
    pitches: list[int],
    labels: list[int | None],
    model: dict,
) -> np.ndarray:
    """
    Forward-backward for 1st-order HMM → posterior shape (N, 5).
    Used for entropy-based model-recommended note selection.
    """
    N = len(pitches)
    F = N_FINGERS
    trans = model["trans"]   # (F, F)
    emit  = model["emit"]    # (F, 128)
    init  = model["init"]    # (F,)

    def get_emit(n: int, f: int) -> float:
        if labels[n] is not None:
            return 1.0 if f == labels[n] - 1 else 0.0
        return float(emit[f, pitches[n]])

    # Forward
    alpha = np.zeros((N, F))
    for f in range(F):
        alpha[0, f] = init[f] * get_emit(0, f)
    s = alpha[0].sum()
    if s > 0:
        alpha[0] /= s

    for n in range(1, N):
        for f in range(F):
            alpha[n, f] = get_emit(n, f) * np.sum(alpha[n-1] * trans[:, f])
        s = alpha[n].sum()
        if s > 0:
            alpha[n] /= s

    # Backward
    beta = np.ones((N, F))
    for n in range(N - 2, -1, -1):
        for f in range(F):
            beta[n, f] = np.sum(trans[f] * np.array([get_emit(n+1, g) for g in range(F)]) * beta[n+1])
        s = beta[n].sum()
        if s > 0:
            beta[n] /= s

    posterior = alpha * beta
    row_sums = posterior.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return posterior / row_sums


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: dict, path: str) -> None:
    np.savez(path, **model)


def load_model(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}

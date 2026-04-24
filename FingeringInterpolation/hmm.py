"""
2nd-order HMM for piano fingering (Nakamura et al. 2020 / Saitō & Nakamura 2022).

Emission: physical keyboard position interval (dX, dY) — 93 bins.
NOT semitone pitch interval. dX = white-key columns, dY = black/white key type.
Timing: fixed log-penalty for biomechanically wrong fast transitions.

Reference implementation: FingeringHMM_v180925.hpp (Nakamura 2020 source code).
"""
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FINGERS = 5     # 1=thumb … 5=pinky
EPS       = 1e-3    # matches Nakamura SmoothInit eps=1e-3
LOG_ZERO  = -1e18

# Physical keyboard position model (Nakamura 2020)
_WIDTH_X        = 15                            # dX clipped to [-15, +15]
N_KEYPOS_BINS   = 3 * (2 * _WIDTH_X + 1)       # 93 bins
_SHORT_TIME_S   = 0.03                          # 30ms IOI threshold
_SHORT_TIME_COST = -5.0                         # fixed log-penalty
# Optimized weights from Nakamura 2020 Table 3 (Bayesian optimisation result)
_W1             = 0.556  # weight for 1-step keypos emission  (paper: α1=0.556)
_W2             = 0.407  # weight for 2-step keypos emission  (paper: α2=0.407)
_LAM1           = 0.474  # 2nd→1st order transition smoothing (paper: λ1=0.474)


# ---------------------------------------------------------------------------
# Physical key position helpers (directly ported from KeyPos_v161230.hpp)
# ---------------------------------------------------------------------------

def _pitch_to_keypos(pitch: int) -> tuple[int, int]:
    """
    MIDI pitch → physical key position (x, y).
    Convention: C4=60 → (0,0), D4=62 → (1,0), Eb4=63 → (1,1).
    x = white-key column, y = 0 (white) or 1 (black).
    """
    pc  = pitch % 12
    oct = pitch // 12 - 1
    if   pc in (0, 1):  x = 0
    elif pc in (2, 3):  x = 1
    elif pc == 4:        x = 2
    elif pc in (5, 6):  x = 3
    elif pc in (7, 8):  x = 4
    elif pc in (9, 10): x = 5
    else:                x = 6     # pc == 11
    x += 7 * (oct - 4)
    y = 0 if pc in (0, 2, 4, 5, 7, 9, 11) else 1
    return (x, y)


def _keypos_interval(p_to: int, p_from: int) -> tuple[int, int]:
    """Physical interval from p_from to p_to → (dX, dY)."""
    x1, y1 = _pitch_to_keypos(p_to)
    x2, y2 = _pitch_to_keypos(p_from)
    return (x1 - x2, y1 - y2)


def _keypos_idx(dx: int, dy: int) -> int:
    """(dX, dY) → bin index in [0, 92]. Formula: 3*(dX+15)+dY+1."""
    dx = max(-_WIDTH_X, min(_WIDTH_X, dx))
    return 3 * (dx + _WIDTH_X) + dy + 1


def _short_time_penalty(hand: str, fp: int, fc: int, ioi_s: float, dp: int) -> float:
    """
    Fixed log-penalty when IOI<30ms AND finger/pitch directions conflict.
    hand='R': penalise (fc-fp)*dp < 0.
    hand='L': penalise (fc-fp)*dp > 0.
    """
    if ioi_s >= _SHORT_TIME_S:
        return 0.0
    diff = (fc - fp) * dp
    if hand == "R" and diff < 0:
        return _SHORT_TIME_COST
    if hand == "L" and diff > 0:
        return _SHORT_TIME_COST
    return 0.0


# ---------------------------------------------------------------------------
# Simple pitch-only 2nd-order HMM (fallback / baseline)
# ---------------------------------------------------------------------------

def train_hmm(pieces: list[list[tuple[int, int]]]) -> dict:
    """Pitch-only 2nd-order HMM. pieces: [(pitch, finger), ...] per piece."""
    trans = np.zeros((N_FINGERS, N_FINGERS, N_FINGERS)) + EPS
    emit  = np.zeros((N_FINGERS, 128)) + EPS
    init  = np.zeros((N_FINGERS, N_FINGERS)) + EPS

    for piece in pieces:
        if len(piece) < 2:
            continue
        pitches = [p for p, _ in piece]
        fingers = [f - 1 for _, f in piece]
        if len(fingers) >= 2:
            init[fingers[0], fingers[1]] += 1
        for n in range(2, len(fingers)):
            trans[fingers[n-2], fingers[n-1], fingers[n]] += 1
        for p, f in zip(pitches, fingers):
            emit[f, p] += 1

    trans /= trans.sum(axis=2, keepdims=True)
    emit  /= emit.sum(axis=1, keepdims=True)
    init  /= init.sum()
    return {"trans": trans, "emit": emit, "init": init, "type": "simple"}


# ---------------------------------------------------------------------------
# Extended 2nd-order HMM (Nakamura 2020 model)
# pieces: [(pitch, finger, onset_time), ...]
# ---------------------------------------------------------------------------

def train_hmm_extended(pieces: list[list[tuple[int, int, float]]], hand: str = "R") -> dict:
    """
    Train extended 2nd-order HMM matching Nakamura et al. 2020.

    Emission:
      outProb [fp, fc, dkey]  — 1-step keypos: P(dkey_{n,n-1} | fp, fc)
      outProb2[fpp, fc, dkey] — 2-step keypos: P(dkey_{n,n-2} | fpp, fc)

    No absolute pitch emission. Timing via fixed short-time penalty at inference.

    Args:
        pieces: list of [(pitch, finger, onset_time)], finger 1-indexed.
        hand:   "R" or "L" — affects short-time penalty direction at inference.
    """
    trans     = np.zeros((N_FINGERS, N_FINGERS, N_FINGERS)) + EPS  # P(fc|fpp,fp)
    init      = np.zeros((N_FINGERS, N_FINGERS)) + EPS              # P(fp0, fp1)
    outProb   = np.zeros((N_FINGERS, N_FINGERS, N_KEYPOS_BINS)) + EPS  # 1-step
    outProb2  = np.zeros((N_FINGERS, N_FINGERS, N_KEYPOS_BINS)) + EPS  # 2-step

    for piece in pieces:
        if len(piece) < 2:
            continue
        pitches = [p for p, _, _ in piece]
        fingers = [f - 1 for _, f, _ in piece]   # 0-indexed

        if len(fingers) >= 2:
            init[fingers[0], fingers[1]] += 1

        for n in range(1, len(fingers)):
            fp, fc = fingers[n-1], fingers[n]
            dx, dy = _keypos_interval(pitches[n], pitches[n-1])
            outProb[fp, fc, _keypos_idx(dx, dy)] += 1

        for n in range(2, len(fingers)):
            fpp, fp, fc = fingers[n-2], fingers[n-1], fingers[n]
            trans[fpp, fp, fc] += 1
            dx2, dy2 = _keypos_interval(pitches[n], pitches[n-2])
            outProb2[fpp, fc, _keypos_idx(dx2, dy2)] += 1

    # 1st-order transition (for λ1 smoothing)
    trans1 = np.zeros((N_FINGERS, N_FINGERS)) + EPS
    for piece in pieces:
        if len(piece) < 2:
            continue
        fingers = [f - 1 for _, f, _ in piece]
        for n in range(1, len(fingers)):
            trans1[fingers[n-1], fingers[n]] += 1
    trans1 /= trans1.sum(axis=1, keepdims=True)

    trans_ml = trans / trans.sum(axis=2, keepdims=True)

    # λ1 smoothing: mix 2nd-order MLE with 1st-order (Nakamura 2020 Eq.10)
    trans_smooth = np.zeros_like(trans_ml)
    for fpp in range(N_FINGERS):
        for fp in range(N_FINGERS):
            trans_smooth[fpp, fp] = (
                (1 - _LAM1) * trans_ml[fpp, fp] + _LAM1 * trans1[fp]
            )
    # Re-normalise after mixing (should already sum to 1 but enforce numerically)
    trans_smooth /= trans_smooth.sum(axis=2, keepdims=True)

    init     /= init.sum()
    outProb  /= outProb.sum(axis=2, keepdims=True)
    outProb2 /= outProb2.sum(axis=2, keepdims=True)

    return {
        "trans": trans_smooth, "init": init,
        "outProb": outProb, "outProb2": outProb2,
        "hand": hand, "type": "extended",
    }


# ---------------------------------------------------------------------------
# Constrained Viterbi — extended model
# ---------------------------------------------------------------------------

def constrained_viterbi_extended(
    pitches: list[int],
    onsets:  list[float],
    labels:  list[int | None],
    model:   dict,
) -> list[int]:
    """
    Constrained 2nd-order Viterbi matching Nakamura 2020.

    Score at step n≥2:
      log P(fc | fpp, fp) [trans]
      + W1 * log P(dkey_{n,n-1} | fp, fc) [outProb]
      + W2 * log P(dkey_{n,n-2} | fpp, fc) [outProb2]
      + shortTimePenalty(n, n-1) + shortTimePenalty(n, n-2)
    """
    N = len(pitches)
    if N == 0:
        return []

    trans    = model["trans"]     # (F,F,F)
    init     = model["init"]      # (F,F)
    outProb  = model["outProb"]   # (F,F,93) — 1-step
    outProb2 = model["outProb2"]  # (F,F,93) — 2-step
    hand     = model.get("hand", "R")
    F = N_FINGERS

    log_trans    = np.log(trans + EPS)
    log_out1     = np.log(outProb + EPS)
    log_out2     = np.log(outProb2 + EPS)

    def constrained(n: int, fc: int) -> bool:
        return labels[n] is not None and fc != labels[n] - 1

    def label_ok(n: int, fc: int) -> float:
        return LOG_ZERO if constrained(n, fc) else 0.0

    def emit1(n: int, fp: int, fc: int) -> float:
        """1-step keypos emission, zero if labeled wrong."""
        if constrained(n, fc):
            return LOG_ZERO
        dx, dy = _keypos_interval(pitches[n], pitches[n-1])
        return _W1 * log_out1[fp, fc, _keypos_idx(dx, dy)]

    def emit2_from(n: int, fpp: int, fc: int) -> float:
        """2-step keypos emission."""
        if constrained(n, fc):
            return LOG_ZERO
        dx, dy = _keypos_interval(pitches[n], pitches[n-2])
        return _W2 * log_out2[fpp, fc, _keypos_idx(dx, dy)]

    def stp(n_curr: int, n_prev: int, fp: int, fc: int) -> float:
        ioi = onsets[n_curr] - onsets[n_prev]
        dp  = pitches[n_curr] - pitches[n_prev]
        return _short_time_penalty(hand, fp, fc, ioi, dp)

    if N == 1:
        if labels[0] is not None:
            return [labels[0]]
        return [1]  # default thumb

    # --- Initialise n=0,1 ---
    dp = np.full((F, F), LOG_ZERO)
    for f0 in range(F):
        if constrained(0, f0):
            continue
        for f1 in range(F):
            if constrained(1, f1):
                continue
            dx, dy = _keypos_interval(pitches[1], pitches[0])
            dp[f0, f1] = (np.log(init[f0, f1] + EPS)
                          + _W1 * log_out1[f0, f1, _keypos_idx(dx, dy)]
                          + stp(1, 0, f0, f1))

    if N == 2:
        best = np.unravel_index(np.argmax(dp), dp.shape)
        return [f + 1 for f in best]

    # --- Forward ---
    history = []
    for n in range(2, N):
        new_dp = np.full((F, F), LOG_ZERO)
        new_bp = np.full((F, F), -1, dtype=np.int32)
        for f1 in range(F):
            for f2 in range(F):
                e1 = emit1(n, f1, f2)
                if e1 == LOG_ZERO:
                    continue
                e2_cache = [emit2_from(n, fpp, f2) for fpp in range(F)]
                st12 = stp(n, n-1, f1, f2)
                best_score = LOG_ZERO
                best_fpp   = -1
                for fpp in range(F):
                    if dp[fpp, f1] == LOG_ZERO:
                        continue
                    st02 = stp(n, n-2, fpp, f2)
                    score = (dp[fpp, f1]
                             + log_trans[fpp, f1, f2]
                             + e1 + e2_cache[fpp]
                             + st12 + st02)
                    if score > best_score:
                        best_score = score
                        best_fpp   = fpp
                new_dp[f1, f2] = best_score
                new_bp[f1, f2] = best_fpp
        history.append(new_bp)
        dp = new_dp

    # --- Backtrack ---
    best_last = np.unravel_index(np.argmax(dp), dp.shape)
    f_seq = list(best_last)   # [f_{N-2}, f_{N-1}]
    for bp_step in reversed(history):
        f_seq.insert(0, int(bp_step[f_seq[0], f_seq[1]]))

    return [f + 1 for f in f_seq]


# ---------------------------------------------------------------------------
# 1st-order extended model (for entropy / model-recommended selection)
# ---------------------------------------------------------------------------

def train_hmm_1st(pieces: list[list[tuple[int, int]]]) -> dict:
    """Pitch-only 1st-order HMM."""
    trans    = np.zeros((N_FINGERS, N_FINGERS)) + EPS
    emit     = np.zeros((N_FINGERS, 128)) + EPS
    init     = np.zeros(N_FINGERS) + EPS

    for piece in pieces:
        if not piece:
            continue
        pitches = [p for p, _ in piece]
        fingers = [f - 1 for _, f in piece]
        init[fingers[0]] += 1
        for n in range(1, len(fingers)):
            trans[fingers[n-1], fingers[n]] += 1
        for p, f in zip(pitches, fingers):
            emit[f, p] += 1

    trans /= trans.sum(axis=1, keepdims=True)
    emit  /= emit.sum(axis=1, keepdims=True)
    init  /= init.sum()
    return {"trans": trans, "emit": emit, "init": init, "order": 1, "type": "simple"}


def train_hmm_1st_extended(
    pieces: list[list[tuple[int, int, float]]],
    hand: str = "R",
) -> dict:
    """1st-order extended HMM for entropy estimation."""
    trans    = np.zeros((N_FINGERS, N_FINGERS)) + EPS
    init     = np.zeros(N_FINGERS) + EPS
    outProb  = np.zeros((N_FINGERS, N_FINGERS, N_KEYPOS_BINS)) + EPS

    for piece in pieces:
        if not piece:
            continue
        pitches = [p for p, _, _ in piece]
        fingers = [f - 1 for _, f, _ in piece]
        init[fingers[0]] += 1
        for n in range(1, len(fingers)):
            fp, fc = fingers[n-1], fingers[n]
            trans[fp, fc] += 1
            dx, dy = _keypos_interval(pitches[n], pitches[n-1])
            outProb[fp, fc, _keypos_idx(dx, dy)] += 1

    trans   /= trans.sum(axis=1, keepdims=True)
    init    /= init.sum()
    outProb /= outProb.sum(axis=2, keepdims=True)
    return {
        "trans": trans, "init": init, "outProb": outProb,
        "hand": hand, "order": 1, "type": "extended",
    }


# ---------------------------------------------------------------------------
# Forward-backward (1st-order, for entropy / model-recommended selection)
# ---------------------------------------------------------------------------

def forward_backward_1st(
    pitches: list[int],
    labels:  list[int | None],
    model:   dict,
) -> np.ndarray:
    """Pitch-only 1st-order forward-backward → posterior (N, 5)."""
    N = len(pitches)
    F = N_FINGERS
    trans = model["trans"]
    emit  = model["emit"]
    init  = model["init"]

    def get_emit(n, f):
        if labels[n] is not None:
            return 1.0 if f == labels[n] - 1 else 0.0
        return float(emit[f, pitches[n]])

    alpha = np.zeros((N, F))
    for f in range(F):
        alpha[0, f] = init[f] * get_emit(0, f)
    s = alpha[0].sum()
    if s > 0:
        alpha[0] /= s
    for n in range(1, N):
        for fc in range(F):
            alpha[n, fc] = get_emit(n, fc) * np.sum(alpha[n-1] * trans[:, fc])
        s = alpha[n].sum()
        if s > 0:
            alpha[n] /= s

    beta = np.ones((N, F))
    for n in range(N-2, -1, -1):
        for fp in range(F):
            beta[n, fp] = np.sum(trans[fp] * np.array([get_emit(n+1, g) for g in range(F)]) * beta[n+1])
        s = beta[n].sum()
        if s > 0:
            beta[n] /= s

    post = alpha * beta
    post /= post.sum(axis=1, keepdims=True).clip(1e-20)
    return post


def forward_backward_1st_extended(
    pitches: list[int],
    onsets:  list[float],
    labels:  list[int | None],
    model:   dict,
) -> np.ndarray:
    """Extended 1st-order forward-backward → posterior (N, 5)."""
    N = len(pitches)
    F = N_FINGERS
    trans   = model["trans"]
    outProb = model["outProb"]
    init    = model["init"]
    hand    = model.get("hand", "R")

    def get_emit0(n, f):
        if labels[n] is not None:
            return 1.0 if f == labels[n] - 1 else 0.0
        return 1.0   # uniform first-note prior (no absolute pitch)

    def get_emit_pair(n, fp, fc):
        if labels[n] is not None:
            return 1.0 if fc == labels[n] - 1 else 0.0
        dx, dy = _keypos_interval(pitches[n], pitches[n-1])
        ioi    = onsets[n] - onsets[n-1]
        dp     = pitches[n] - pitches[n-1]
        base   = float(outProb[fp, fc, _keypos_idx(dx, dy)])
        stp    = np.exp(_short_time_penalty(hand, fp, fc, ioi, dp))
        return base * stp

    alpha = np.zeros((N, F))
    for f in range(F):
        alpha[0, f] = init[f] * get_emit0(0, f)
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
    for n in range(N-2, -1, -1):
        for fp in range(F):
            beta[n, fp] = sum(
                trans[fp, fc] * get_emit_pair(n+1, fp, fc) * beta[n+1, fc]
                for fc in range(F)
            )
        s = beta[n].sum()
        if s > 0:
            beta[n] /= s

    post = alpha * beta
    post /= post.sum(axis=1, keepdims=True).clip(1e-20)
    return post


# ---------------------------------------------------------------------------
# Simple pitch-only Viterbi (unchanged, for backward compatibility)
# ---------------------------------------------------------------------------

def constrained_viterbi(
    pitches: list[int],
    labels:  list[int | None],
    model:   dict,
) -> list[int]:
    """2nd-order Viterbi for pitch-only model."""
    N = len(pitches)
    if N == 0:
        return []

    trans = model["trans"]
    emit  = model["emit"]
    init  = model["init"]
    F = N_FINGERS

    def log_emit(n, f):
        if labels[n] is not None:
            return 0.0 if f == labels[n] - 1 else LOG_ZERO
        return np.log(emit[f, pitches[n]] + EPS)

    if N == 1:
        if labels[0] is not None:
            return [labels[0]]
        return [int(np.argmax([emit[f, pitches[0]] for f in range(F)])) + 1]

    dp = np.full((F, F), LOG_ZERO)
    for f0 in range(F):
        for f1 in range(F):
            dp[f0, f1] = (np.log(init[f0, f1] + EPS)
                          + log_emit(0, f0) + log_emit(1, f1))

    history = []
    for n in range(2, N):
        new_dp = np.full((F, F), LOG_ZERO)
        new_bp = np.full((F, F), -1, dtype=np.int32)
        log_tr = np.log(trans + EPS)
        for f1 in range(F):
            le_cache = [log_emit(n, f2) for f2 in range(F)]
            for f2 in range(F):
                le = le_cache[f2]
                if le == LOG_ZERO:
                    continue
                scores  = dp[:, f1] + log_tr[:, f1, f2]
                best_f0 = int(np.argmax(scores))
                new_dp[f1, f2] = scores[best_f0] + le
                new_bp[f1, f2] = best_f0
        history.append(new_bp)
        dp = new_dp

    best_last = np.unravel_index(np.argmax(dp), dp.shape)
    f_seq = list(best_last)
    for bp in reversed(history):
        f_seq.insert(0, int(bp[f_seq[0], f_seq[1]]))

    return [f + 1 for f in f_seq]


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def compute_entropy(posterior: np.ndarray) -> np.ndarray:
    p = np.clip(posterior, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: dict, path: str) -> None:
    np.savez(path, **{k: v for k, v in model.items() if isinstance(v, np.ndarray)},
             **{k: str(v) for k, v in model.items() if not isinstance(v, np.ndarray)})


def load_model(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    model = {}
    for k in data.files:
        v = data[k]
        if v.ndim == 0:   # scalar string stored as 0-d array
            model[k] = str(v)
        else:
            model[k] = v
    return model

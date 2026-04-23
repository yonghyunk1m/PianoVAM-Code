"""
Rule-based hard part selector for manual fingering verification.

Identifies notes that should be prioritized for human review based on
physical impossibility, MediaPipe tracking difficulty, and musical complexity.

Input:  fingering TSV (onset, key_offset, frame_offset, note, velocity, hand, finger)
Output: DataFrame with 'is_hard' and 'hard_reasons' columns
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# TSV loader
# ---------------------------------------------------------------------------

def load_fingering_tsv(path: str) -> pd.DataFrame:
    """Load a fingering TSV. Handles both comment-header and plain-header formats."""
    with open(path) as f:
        first = f.readline().strip()
    if first.startswith("#"):
        col_names = [c.strip().lower() for c in first.lstrip("# ").strip().split("\t")]
        df = pd.read_csv(path, sep="\t", comment="#", header=None, names=col_names)
    else:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip().lower() for c in df.columns]

    if "finger" not in df.columns:
        df["finger"] = "Noinfo"
    if "hand" not in df.columns:
        df["hand"] = "Noinfo"

    def parse_finger(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    df["finger_int"] = df["finger"].apply(parse_finger)
    return df


# ---------------------------------------------------------------------------
# Helper: max comfortable hand span in semitones between two fingers
# Based on standard piano pedagogy (adult average hand).
# ---------------------------------------------------------------------------
#   finger pair (lo, hi) → max semitone span (threshold for flagging)
#   Tuned so that an octave (12st) with thumb+pinky is NOT flagged;
#   only truly extreme stretches (>15st, e.g. 10th or 12th) are flagged.
_MAX_SPAN = {
    (1, 2): 6,   (1, 3): 10,  (1, 4): 12, (1, 5): 15,
    (2, 3): 4,   (2, 4): 7,   (2, 5): 10,
    (3, 4): 4,   (3, 5): 7,
    (4, 5): 4,
}

def _max_span(f1: int, f2: int) -> int:
    lo, hi = min(f1, f2), max(f1, f2)
    return _MAX_SPAN.get((lo, hi), 12)


# ===========================================================================
# RULE 1  — Physically impossible fingering
# ===========================================================================

def rule_impossible_fingering(df: pd.DataFrame) -> pd.Series:
    """
    Two sub-checks:

    (a) Non-thumb finger crossing: consecutive same-hand notes where the finger
        direction opposes the expected direction without a thumb (finger 1).
        R hand: ascending pitch → finger should increase (or thumb cross).
        L hand: ascending pitch → finger should decrease (or thumb cross).
        Flagged only when neither the previous nor current finger is the thumb.

    (b) Same-hand chord span overreach: two notes in the same hand that overlap
        in time (held simultaneously) whose pitch distance exceeds the comfortable
        span for that finger pair. This catches physically impossible chord
        stretches, not sequential jumps.
    """
    flags = pd.Series(False, index=df.index)

    has_offset = "key_offset" in df.columns

    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < 2:
            continue

        notes   = sub["note"].values
        fingers = sub["finger_int"].values
        onsets  = sub["onset"].values if "onset" in df.columns else np.zeros(len(sub))
        offsets = sub["key_offset"].values if has_offset else onsets + 0.5
        idxs    = list(sub.index)

        for i in range(1, len(notes)):
            f_prev, f_curr = fingers[i-1], fingers[i]
            p_prev, p_curr = notes[i-1], notes[i]
            if f_prev is None or f_curr is None:
                continue

            # (a) Finger-cross without thumb
            # Cross means: pitch and finger go in opposite expected directions
            # For R: normal = pitch_up↔finger_up; cross = pitch_up↔finger_down
            # For L: normal = pitch_up↔finger_down; cross = pitch_up↔finger_up
            # Combined: cross when (pitch_up == finger_up) == (hand == "L")
            if p_prev != p_curr:  # skip repeated notes
                pitch_up  = bool(p_curr > p_prev)
                finger_up = bool(f_curr > f_prev)
                is_cross  = (pitch_up == finger_up) == (hand == "L")
                if is_cross and f_prev != 1 and f_curr != 1:
                    flags.iloc[idxs[i-1]] = True
                    flags.iloc[idxs[i]]   = True

            # (b) Chord span overreach: notes overlap in time → held simultaneously
            if onsets[i] < offsets[i-1]:   # note i starts before note i-1 ends
                pitch_span = abs(int(p_curr) - int(p_prev))
                if pitch_span > _max_span(f_prev, f_curr):
                    flags.iloc[idxs[i-1]] = True
                    flags.iloc[idxs[i]]   = True

    return flags


# ===========================================================================
# RULE 2  — Fast position jump (hand blur)
# ===========================================================================

def rule_fast_jump(
    df: pd.DataFrame,
    jump_semitones: int = 10,
    jump_window_ms: float = 180.0,
) -> pd.Series:
    """
    A hand jumps a large interval in a short time → hand is blurry in video,
    MediaPipe tracking likely inaccurate.

    Flags both the note before and after the jump, plus the 2 notes on each side
    (the hand is unstable during the approach and landing).
    """
    flags = pd.Series(False, index=df.index)
    if "onset" not in df.columns:
        return flags

    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < 2:
            continue

        notes  = sub["note"].values
        onsets = sub["onset"].values * 1000   # → ms
        idxs   = list(sub.index)

        for i in range(1, len(notes)):
            dt = onsets[i] - onsets[i-1]
            dp = abs(int(notes[i]) - int(notes[i-1]))
            if dp >= jump_semitones and dt <= jump_window_ms:
                # Flag the jump itself plus 2-note context on each side
                for j in range(max(0, i-2), min(len(idxs), i+3)):
                    flags.iloc[idxs[j]] = True

    return flags


# ===========================================================================
# RULE 3  — Fast phrase (many notes per second)
# ===========================================================================

def rule_fast_phrase(
    df: pd.DataFrame,
    ioi_threshold_ms: float = 100.0,
    min_run_length: int = 4,
) -> pd.Series:
    """
    Runs of consecutive notes (per hand) with IOI < threshold.
    Short IOI = fast passage where hand shape is ambiguous in video.
    Flags the entire run.
    """
    flags = pd.Series(False, index=df.index)
    if "onset" not in df.columns:
        return flags

    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < min_run_length:
            continue

        onsets = sub["onset"].values * 1000
        idxs   = list(sub.index)
        ioi    = np.diff(onsets)

        run_start = None
        for i, dt in enumerate(ioi):
            if dt < ioi_threshold_ms:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_end = i  # inclusive
                    if (run_end - run_start + 1) >= min_run_length:
                        for j in range(run_start, run_end + 2):
                            if j < len(idxs):
                                flags.iloc[idxs[j]] = True
                    run_start = None

        # Close open run at end
        if run_start is not None:
            run_end = len(ioi)
            if (run_end - run_start + 1) >= min_run_length:
                for j in range(run_start, min(run_end + 2, len(idxs))):
                    flags.iloc[idxs[j]] = True

    return flags


# ===========================================================================
# RULE 4  — Hand position overlap
# ===========================================================================

def rule_hand_overlap(
    df: pd.DataFrame,
    window_ms: float = 200.0,
    overlap_tolerance_semitones: int = 2,
) -> pd.Series:
    """
    Within a time window, L and R hands occupy overlapping pitch regions.
    Flags all notes in the window when L pitches >= R pitches (overlap or crossing).
    MediaPipe hand identity assignment gets unreliable when hands are close.
    """
    flags = pd.Series(False, index=df.index)
    if "onset" not in df.columns:
        return flags

    onsets = df["onset"].values * 1000
    notes  = df["note"].values
    hands  = df["hand"].values

    for i in range(len(df)):
        if hands[i] not in ("L", "R"):
            continue
        t = onsets[i]
        # Gather all notes within the window
        window_mask = np.abs(onsets - t) <= window_ms
        L_pitches = notes[window_mask & (hands == "L")]
        R_pitches = notes[window_mask & (hands == "R")]
        if len(L_pitches) == 0 or len(R_pitches) == 0:
            continue
        # Overlap: max(L) >= min(R) - tolerance
        if L_pitches.max() >= R_pitches.min() - overlap_tolerance_semitones:
            flags.iloc[i] = True

    return flags


# ===========================================================================
# RULE 5  — Noinfo notes
# ===========================================================================

def rule_noinfo(df: pd.DataFrame) -> pd.Series:
    """Notes with no finger assignment at all."""
    return df["finger"].astype(str).str.lower() == "noinfo"


# ===========================================================================
# RULE 6  — Noinfo cluster (3+ consecutive Noinfo)
# ===========================================================================

def rule_noinfo_cluster(df: pd.DataFrame, min_cluster: int = 3) -> pd.Series:
    """
    3 or more consecutive Noinfo notes → MediaPipe completely lost tracking.
    The entire cluster and its context are unreliable.
    """
    flags   = pd.Series(False, index=df.index)
    is_ni   = (df["finger"].astype(str).str.lower() == "noinfo").values
    idxs    = list(df.index)

    run_start = None
    for i, ni in enumerate(is_ni):
        if ni:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_cluster:
                for j in range(run_start, i):
                    flags.iloc[idxs[j]] = True
            run_start = None
    if run_start is not None and (len(is_ni) - run_start) >= min_cluster:
        for j in range(run_start, len(is_ni)):
            flags.iloc[idxs[j]] = True

    return flags


# ===========================================================================
# RULE 7  — Rapid hand alternation (tremolo-like)
# ===========================================================================

def rule_rapid_alternation(
    df: pd.DataFrame,
    alt_ioi_ms: float = 120.0,
    min_alternations: int = 4,
) -> pd.Series:
    """
    Rapid L/R alternation (tremolo, Alberti bass, etc.).
    When both hands are active and interleaved faster than ~8 notes/sec,
    MediaPipe struggles to distinguish which hand is which.
    Flags runs of ≥ min_alternations consecutive L/R/L/R patterns.
    """
    flags = pd.Series(False, index=df.index)
    if "onset" not in df.columns or len(df) < min_alternations:
        return flags

    onsets = df["onset"].values * 1000
    hands  = df["hand"].values
    idxs   = list(df.index)

    run_start = None
    for i in range(1, len(df)):
        dt   = onsets[i] - onsets[i-1]
        h_ok = hands[i] in ("L", "R") and hands[i-1] in ("L", "R")
        alternating = h_ok and hands[i] != hands[i-1] and dt < alt_ioi_ms
        if alternating:
            if run_start is None:
                run_start = i - 1
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_alternations:
                    for j in range(run_start, i):
                        flags.iloc[idxs[j]] = True
                run_start = None

    if run_start is not None and (len(df) - run_start) >= min_alternations:
        for j in range(run_start, len(df)):
            flags.iloc[idxs[j]] = True

    return flags


# ===========================================================================
# RULE 8  — Finger order violation in stepwise motion
# ===========================================================================

def rule_stepwise_order_violation(
    df: pd.DataFrame,
    step_semitones: int = 2,
) -> pd.Series:
    """
    For stepwise passages (consecutive notes ≤ step_semitones apart within a hand),
    finger numbers should follow pitch direction (R ascending → finger increases,
    or thumb cross; R descending → finger decreases, or thumb cross).
    Flags cases where direction flips without a thumb (finger 1) involved.
    """
    flags = pd.Series(False, index=df.index)

    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < 2:
            continue

        notes   = sub["note"].values
        fingers = sub["finger_int"].values
        idxs    = list(sub.index)

        for i in range(1, len(notes)):
            f_prev, f_curr = fingers[i-1], fingers[i]
            p_prev, p_curr = notes[i-1], notes[i]
            if f_prev is None or f_curr is None:
                continue
            # skip same pitch (repeated note — finger change is valid)
            if p_curr == p_prev:
                continue
            pitch_diff = abs(int(p_curr) - int(p_prev))
            if pitch_diff > step_semitones:
                continue  # not stepwise

            # For R hand: ascending steps → finger should increase (or thumb cross)
            # For L hand: ascending steps → finger should decrease (or thumb cross)
            pitch_up  = p_curr > p_prev
            finger_up = f_curr > f_prev
            is_cross  = (pitch_up == finger_up) == (hand == "L")
            if is_cross and f_prev != 1 and f_curr != 1:
                flags.iloc[idxs[i-1]] = True
                flags.iloc[idxs[i]]   = True

    return flags


# ===========================================================================
# Rule registry
# ===========================================================================

RULES: dict[str, callable] = {
    # Physically impossible
    "impossible_fingering":       rule_impossible_fingering,
    # MediaPipe unreliable situations
    "fast_jump":                  rule_fast_jump,
    "fast_phrase":                rule_fast_phrase,
    "hand_overlap":               rule_hand_overlap,
    "rapid_alternation":          rule_rapid_alternation,
    # Data quality / no assignment
    "noinfo":                     rule_noinfo,
    "noinfo_cluster":             rule_noinfo_cluster,
    # Fingering logic errors
    "stepwise_order_violation":   rule_stepwise_order_violation,
}

RULE_DESCRIPTIONS: dict[str, str] = {
    "impossible_fingering":     "Physically impossible: finger cross w/o thumb, or span overreach",
    "fast_jump":                "Fast position jump: hand blurry in video, MediaPipe inaccurate",
    "fast_phrase":              "Fast phrase (IOI < 100ms, ≥4 notes): tracking unreliable",
    "hand_overlap":             "Hand position overlap: L/R pitch regions intersect",
    "rapid_alternation":        "Rapid L/R alternation (tremolo): hand identity ambiguous",
    "noinfo":                   "No finger assigned (Noinfo)",
    "noinfo_cluster":           "Cluster of 3+ consecutive Noinfo: tracking completely lost",
    "stepwise_order_violation": "Finger order wrong in stepwise motion (w/o thumb cross)",
}

# Default rules to enable in the UI
DEFAULT_RULES = [
    "impossible_fingering",
    "fast_jump",
    "fast_phrase",
    "hand_overlap",
    "noinfo_cluster",
]


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------

def select_hard_parts(
    df: pd.DataFrame,
    enabled_rules: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply enabled rules. Returns df with 'is_hard' and 'hard_reasons' columns.
    """
    if enabled_rules is None:
        enabled_rules = list(RULES.keys())

    reasons: list[list[str]] = [[] for _ in range(len(df))]

    for rule_name in enabled_rules:
        if rule_name not in RULES:
            raise ValueError(f"Unknown rule: '{rule_name}'. Available: {list(RULES.keys())}")
        flags = RULES[rule_name](df)
        for i, flag in enumerate(flags):
            if flag:
                reasons[i].append(rule_name)

    df = df.copy()
    df["hard_reasons"] = [",".join(r) for r in reasons]
    df["is_hard"]      = df["hard_reasons"].str.len() > 0
    return df


def get_hard_segments(
    df: pd.DataFrame,
    context_notes: int = 4,
) -> list[dict]:
    """
    Group consecutive hard notes into segments with surrounding context.
    """
    hard_idx = df.index[df["is_hard"]].tolist()
    if not hard_idx:
        return []

    segments = []
    seg_start = hard_idx[0]
    seg_end   = hard_idx[0]

    for idx in hard_idx[1:]:
        if idx <= seg_end + context_notes:
            seg_end = idx
        else:
            segments.append(_make_segment(df, seg_start, seg_end, context_notes))
            seg_start = idx
            seg_end   = idx
    segments.append(_make_segment(df, seg_start, seg_end, context_notes))

    return segments


def _make_segment(df: pd.DataFrame, start: int, end: int, context: int) -> dict:
    ctx_start = max(0, start - context)
    ctx_end   = min(len(df) - 1, end + context)
    reasons   = sorted(set(",".join(df.loc[start:end, "hard_reasons"]).split(",")) - {""})
    return {
        "start_idx": start,
        "end_idx":   end,
        "ctx_start": ctx_start,
        "ctx_end":   ctx_end,
        "reasons":   reasons,
        "n_hard":    end - start + 1,
        "notes":     df.iloc[ctx_start : ctx_end + 1],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Select hard parts from fingering TSV")
    parser.add_argument("tsv", help="Path to fingering TSV with hand/finger columns")
    parser.add_argument("--rules", default=None,
                        help="Comma-separated rule names (default: all). "
                             f"Available: {','.join(RULES)}")
    parser.add_argument("--context", type=int, default=4,
                        help="Context notes around each hard segment (default: 4)")
    parser.add_argument("--output", default=None,
                        help="Save flagged-note TSV to this path")
    parser.add_argument("--summary", action="store_true",
                        help="Print per-rule hit counts")
    args = parser.parse_args()

    df = load_fingering_tsv(args.tsv)
    rules = [r.strip() for r in args.rules.split(",")] if args.rules else None
    df = select_hard_parts(df, rules)

    n_hard = df["is_hard"].sum()
    print(f"{Path(args.tsv).name}: {n_hard}/{len(df)} notes flagged as hard")

    if args.summary:
        enabled = rules or list(RULES.keys())
        print("\nPer-rule counts:")
        for r in enabled:
            count = df["hard_reasons"].str.contains(r).sum()
            print(f"  {r:35s}: {count}")

    segs = get_hard_segments(df, args.context)
    print(f"\n{len(segs)} hard segments:")
    for s in segs:
        print(f"  notes {s['start_idx']:4d}–{s['end_idx']:4d}  "
              f"({s['n_hard']} hard)  reasons: {s['reasons']}")

    if args.output:
        df[df["is_hard"]].to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved flagged notes → {args.output}")

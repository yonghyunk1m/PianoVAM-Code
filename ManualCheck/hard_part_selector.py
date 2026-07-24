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

from fingering_audit.features.audit_flags import (
    compute_audit_flags,
    noinfo_context_mask,
)


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


def _canonical_for_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt ManualCheck's legacy columns to the shared audit contract."""

    def column(name: str) -> pd.Series:
        if name in df:
            return df[name].reset_index(drop=True)
        return pd.Series([pd.NA] * len(df))

    return pd.DataFrame({
        "recording_id": ["manualcheck"] * len(df),
        "note_id": [f"manualcheck#{i}" for i in range(len(df))],
        "note_idx": range(len(df)),
        "onset_sec": pd.to_numeric(column("onset"), errors="coerce"),
        "offset_sec": pd.to_numeric(column("key_offset"), errors="coerce"),
        "pitch": pd.to_numeric(column("note"), errors="coerce"),
        "pred_hand": column("hand"),
        "pred_finger": pd.array(
            pd.to_numeric(column("finger_int"), errors="coerce"),
            dtype="Int64",
        ),
        "compound_fingering": False,
    })


# ===========================================================================
# RULE 1  — Shared physical diagnostic and legacy crossing risk
# ===========================================================================

def rule_physical_candidate_diagnostic(df: pd.DataFrame) -> pd.Series:
    """Return shared-engine physical candidates, excluding invalid records."""
    flags = compute_audit_flags(_canonical_for_audit(df))
    return pd.Series(
        (flags.physical_candidate & ~flags.integrity).to_numpy(),
        index=df.index,
    )


def rule_non_thumb_crossing(
    df: pd.DataFrame,
    cross_timing_ms: float = 500.0,
) -> pd.Series:
    """Flag quick non-thumb crossings as a review risk, not impossibility."""
    flags = pd.Series(False, index=df.index)

    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < 2:
            continue

        notes   = sub["note"].values
        fingers = sub["finger_int"].values
        onsets  = sub["onset"].values if "onset" in df.columns else np.zeros(len(sub))
        idxs    = list(sub.index)

        for i in range(1, len(notes)):
            f_prev, f_curr = fingers[i-1], fingers[i]
            p_prev, p_curr = notes[i-1], notes[i]
            if pd.isna(f_prev) or pd.isna(f_curr):
                continue

            if p_prev != p_curr:  # skip repeated notes
                dt_ms     = (onsets[i] - onsets[i-1]) * 1000
                pitch_up  = bool(p_curr > p_prev)
                finger_up = bool(f_curr > f_prev)
                is_cross  = (pitch_up == finger_up) == (hand == "L")
                if is_cross and f_prev != 1 and f_curr != 1 and dt_ms <= cross_timing_ms:
                    flags.iloc[idxs[i-1]] = True
                    flags.iloc[idxs[i]]   = True

    return flags


def rule_impossible_fingering(df: pd.DataFrame) -> pd.Series:
    """Deprecated compatibility alias for legacy review-risk output."""
    return (
        rule_physical_candidate_diagnostic(df)
        | rule_non_thumb_crossing(df)
    )


def rule_noinfo_context_k3_r2(df: pd.DataFrame) -> pd.Series:
    """Flag two assigned neighbors around runs of at least three Noinfo notes."""
    canonical = _canonical_for_audit(df)
    integrity = compute_audit_flags(canonical).integrity
    selected = noinfo_context_mask(
        canonical,
        min_run=3,
        radius=2,
        integrity_mask=integrity,
    )
    return pd.Series(selected.to_numpy(), index=df.index)


# ===========================================================================
# RULE 2  — Fast position jump (hand blur)
# ===========================================================================

def rule_fast_jump(
    df: pd.DataFrame,
    jump_semitones: int = 15,
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

def rule_noinfo_cluster(
    df: pd.DataFrame,
    min_cluster: int = 3,
    context_notes: int = 2,
) -> pd.Series:
    """
    3+ consecutive Noinfo notes → MediaPipe completely lost tracking.
    Flags the algorithm-assigned notes immediately before and after the cluster
    (not the noinfo notes themselves, which have nothing to verify).
    """
    flags = pd.Series(False, index=df.index)
    is_ni = (df["finger"].astype(str).str.lower() == "noinfo").values
    idxs  = list(df.index)
    n     = len(is_ni)

    def flag_context(cluster_start: int, cluster_end: int) -> None:
        count = 0
        for j in range(cluster_start - 1, -1, -1):
            if not is_ni[j]:
                flags.iloc[idxs[j]] = True
                count += 1
                if count >= context_notes:
                    break
        count = 0
        for j in range(cluster_end, n):
            if not is_ni[j]:
                flags.iloc[idxs[j]] = True
                count += 1
                if count >= context_notes:
                    break

    run_start = None
    for i, ni in enumerate(is_ni):
        if ni:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_cluster:
                flag_context(run_start, i)
            run_start = None
    if run_start is not None and (n - run_start) >= min_cluster:
        flag_context(run_start, n)

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
    # Shared physical diagnostic and deprecated compatibility name
    "physical_candidate_diagnostic": rule_physical_candidate_diagnostic,
    "non_thumb_crossing":             rule_non_thumb_crossing,
    "impossible_fingering":       rule_impossible_fingering,
    # MediaPipe unreliable situations
    "fast_jump":                  rule_fast_jump,
    "hand_overlap":               rule_hand_overlap,
    "rapid_alternation":          rule_rapid_alternation,
    # Data quality / no assignment
    "noinfo":                     rule_noinfo,
    "noinfo_cluster":             rule_noinfo_cluster,
    "noinfo_context_k3_r2":       rule_noinfo_context_k3_r2,
    # Fingering logic errors
    "stepwise_order_violation":   rule_stepwise_order_violation,
}

RULE_DESCRIPTIONS: dict[str, str] = {
    "physical_candidate_diagnostic": "Shared physical candidate diagnostic (not must-alert without a validated policy)",
    "non_thumb_crossing":       "Non-thumb crossing risk within 500 ms",
    "impossible_fingering":     "Deprecated legacy physical/crossing review risk",
    "fast_jump":                "Fast position jump: hand blurry in video, MediaPipe inaccurate",
    "hand_overlap":             "Hand position overlap: L/R pitch regions intersect",
    "rapid_alternation":        "Rapid L/R alternation (tremolo): hand identity ambiguous",
    "noinfo":                   "No finger assigned (Noinfo)",
    "noinfo_cluster":           "Cluster of 3+ consecutive Noinfo: tracking completely lost",
    "noinfo_context_k3_r2":     "Two assigned-note neighbors around a run of 3+ Noinfo notes",
    "stepwise_order_violation": "Finger order wrong in stepwise motion (w/o thumb cross)",
}

# Default rules to enable in the UI
DEFAULT_RULES = [
    "physical_candidate_diagnostic",
    "non_thumb_crossing",
    "fast_jump",
    "noinfo_context_k3_r2",
]


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------

def select_hard_parts(
    df: pd.DataFrame,
    enabled_rules: list[str] | None = None,
    physical_policy=None,
) -> pd.DataFrame:
    """
    Apply enabled rules and expose stable shared-audit category columns.

    Physical candidates remain diagnostic unless ``physical_policy`` explicitly
    enables the corresponding PIG-validated rule.
    """
    if enabled_rules is None:
        enabled_rules = list(RULES.keys())

    canonical = _canonical_for_audit(df)
    boundaries = (
        physical_policy.span_boundaries
        if physical_policy is not None
        else None
    )
    audit = compute_audit_flags(canonical, boundaries)
    integrity = audit.integrity.astype(bool)
    physical_candidate = audit.physical_candidate.astype(bool) & ~integrity
    physical_must_alert = pd.Series(False, index=canonical.index)
    if physical_policy is not None:
        enabled_physical = physical_policy.enabled_rules
        if "simultaneous_same_finger_different_pitch" in enabled_physical:
            physical_must_alert |= audit.same_finger_candidate
        if "simultaneous_pair_span" in enabled_physical:
            physical_must_alert |= audit.span_candidate
    physical_must_alert &= ~integrity

    noinfo_context = noinfo_context_mask(
        canonical,
        min_run=3,
        radius=2,
        integrity_mask=integrity,
    )
    noinfo_context &= ~integrity

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
    df["physical_must_alert"] = physical_must_alert.to_numpy(dtype=bool)
    df["physical_reasons"] = [
        list(value) if candidate else []
        for value, candidate in zip(
            audit.physical_reasons,
            physical_candidate,
        )
    ]
    df["data_integrity_must_resolve"] = integrity.to_numpy(dtype=bool)
    df["data_integrity_reasons"] = [
        list(value) for value in audit.integrity_reasons
    ]
    noinfo_enabled = "noinfo_context_k3_r2" in enabled_rules
    df["noinfo_context_alert"] = (
        noinfo_context.to_numpy(dtype=bool)
        if noinfo_enabled
        else False
    )
    df["noinfo_context_reasons"] = [
        ["noinfo_context_k3_r2"] if noinfo_enabled and value else []
        for value in noinfo_context
    ]
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

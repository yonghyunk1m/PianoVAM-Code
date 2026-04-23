"""
Rule-based hard part selector for manual fingering verification.

Identifies "hard" notes that should be prioritized for human review.
Rules will be extended once the user specifies them; current rules are placeholders
based on common musical difficulty indicators.

Input:  fingering TSV (onset, key_offset, frame_offset, note, velocity, hand, finger)
Output: list of note indices flagged as "hard"
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# TSV loader
# ---------------------------------------------------------------------------

def load_fingering_tsv(path: str) -> pd.DataFrame:
    """Load a fingering TSV into a DataFrame. Handles both comment-header and plain-header formats."""
    # Peek at first non-empty line to detect format
    with open(path) as f:
        first = f.readline().strip()
    if first.startswith("#"):
        # Comment-style header: "# onset\tkey_offset\t..."
        header_line = first.lstrip("# ").strip()
        col_names = [c.strip().lower() for c in header_line.split("\t")]
        df = pd.read_csv(path, sep="\t", comment="#", header=None, names=col_names)
    else:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip().lower() for c in df.columns]
    # Convert finger to int where possible (keep "Noinfo" as string)
    def parse_finger(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return None
    if "finger" not in df.columns:
        df["finger"]     = "Noinfo"
        df["hand"]       = "Noinfo"
    if "hand" not in df.columns:
        df["hand"]       = "Noinfo"
    df["finger_int"] = df["finger"].apply(parse_finger)
    return df


# ---------------------------------------------------------------------------
# Individual rules — each returns a boolean Series (True = hard)
# ---------------------------------------------------------------------------

def rule_noinfo(df: pd.DataFrame) -> pd.Series:
    """Notes with no finger assignment (Noinfo)."""
    return df["finger"].astype(str).str.lower() == "noinfo"


def rule_thumb_under(df: pd.DataFrame) -> pd.Series:
    """
    Thumb-under / finger-over passages: thumb (1) appears after index/middle (2,3)
    within same hand in a short window — a classic technical difficulty.
    """
    flags = pd.Series(False, index=df.index)
    window = 4  # look at 4 consecutive notes per hand
    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub = df[mask].copy()
        if len(sub) < window:
            continue
        fingers = sub["finger_int"].values
        for i in range(len(fingers) - window + 1):
            chunk = fingers[i:i+window]
            if None in chunk:
                continue
            # thumb appearing after 2/3 in descending pitch (R hand going left = thumb under)
            for j in range(1, len(chunk)):
                if chunk[j] == 1 and chunk[j-1] in (2, 3):
                    flags.iloc[sub.index[i:i+window]] = True
    return flags


def rule_large_interval(df: pd.DataFrame, semitone_threshold: int = 9) -> pd.Series:
    """
    Large melodic intervals (≥ threshold semitones) within same hand.
    These often require hand stretches.
    """
    flags = pd.Series(False, index=df.index)
    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < 2:
            continue
        notes = sub["note"].values
        diffs = np.abs(np.diff(notes.astype(float)))
        hard_idx = np.where(diffs >= semitone_threshold)[0]
        for idx in hard_idx:
            flags.iloc[sub.index[idx]]   = True
            flags.iloc[sub.index[idx+1]] = True
    return flags


def rule_finger_repetition(df: pd.DataFrame) -> pd.Series:
    """
    Same finger used on consecutive different pitches in same hand.
    Unusual and potentially erroneous or very deliberate.
    """
    flags = pd.Series(False, index=df.index)
    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < 2:
            continue
        fingers = sub["finger_int"].values
        notes   = sub["note"].values
        for i in range(1, len(fingers)):
            if (fingers[i] is not None and fingers[i-1] is not None
                    and fingers[i] == fingers[i-1]
                    and notes[i] != notes[i-1]):
                flags.iloc[sub.index[i]]   = True
                flags.iloc[sub.index[i-1]] = True
    return flags


def rule_weak_finger_high_speed(df: pd.DataFrame, fast_ioi_ms: float = 100.0) -> pd.Series:
    """
    Ring or pinky finger (4 or 5) used for fast notes (IOI < threshold ms).
    Weak fingers on fast passages are physically demanding and error-prone.
    """
    flags = pd.Series(False, index=df.index)
    if "onset" not in df.columns:
        return flags
    for hand in ("L", "R"):
        mask = df["hand"] == hand
        sub  = df[mask]
        if len(sub) < 2:
            continue
        onsets  = sub["onset"].values * 1000  # s → ms
        fingers = sub["finger_int"].values
        ioi     = np.diff(onsets)
        for i in range(len(ioi)):
            if ioi[i] < fast_ioi_ms and fingers[i] in (4, 5):
                flags.iloc[sub.index[i]]   = True
                flags.iloc[sub.index[i+1]] = True
    return flags


def rule_hand_crossing(df: pd.DataFrame) -> pd.Series:
    """
    Detect likely hand crossings: left hand note pitch > right hand note pitch
    at overlapping onset times (within 50ms).
    """
    flags = pd.Series(False, index=df.index)
    if "onset" not in df.columns:
        return flags
    onset = df["onset"].values
    note  = df["note"].values
    hand  = df["hand"].values
    overlap_ms = 0.05  # 50ms window
    for i in range(len(df)):
        if hand[i] not in ("L", "R"):
            continue
        for j in range(max(0, i-5), min(len(df), i+5)):
            if i == j or hand[j] not in ("L", "R") or hand[i] == hand[j]:
                continue
            if abs(onset[i] - onset[j]) < overlap_ms:
                if hand[i] == "L" and note[i] > note[j]:   # L above R
                    flags.iloc[i] = True
                    flags.iloc[j] = True
                elif hand[i] == "R" and note[i] < note[j]: # R below L
                    flags.iloc[i] = True
                    flags.iloc[j] = True
    return flags


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------

RULES = {
    "noinfo":             rule_noinfo,
    "thumb_under":        rule_thumb_under,
    "large_interval":     rule_large_interval,
    "finger_repetition":  rule_finger_repetition,
    "weak_finger_fast":   rule_weak_finger_high_speed,
    "hand_crossing":      rule_hand_crossing,
}


def select_hard_parts(
    df: pd.DataFrame,
    enabled_rules: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply enabled rules and return a DataFrame with a 'hard_reasons' column.

    Args:
        df:            Fingering DataFrame from load_fingering_tsv.
        enabled_rules: List of rule names to apply (default: all).
    Returns:
        df with added columns:
          'is_hard'     : bool — True if any rule fires
          'hard_reasons': str  — comma-separated rule names that fired
    """
    if enabled_rules is None:
        enabled_rules = list(RULES.keys())

    reasons = [[] for _ in range(len(df))]

    for rule_name in enabled_rules:
        if rule_name not in RULES:
            raise ValueError(f"Unknown rule: {rule_name}. Available: {list(RULES.keys())}")
        flags = RULES[rule_name](df)
        for i, flag in enumerate(flags):
            if flag:
                reasons[i].append(rule_name)

    df = df.copy()
    df["hard_reasons"] = [",".join(r) for r in reasons]
    df["is_hard"] = df["hard_reasons"].str.len() > 0
    return df


def get_hard_segments(
    df: pd.DataFrame,
    context_notes: int = 4,
) -> list[dict]:
    """
    Group consecutive hard notes into segments with surrounding context.

    Returns list of dicts:
        {start_idx, end_idx, reasons, notes (DataFrame slice)}
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


def _make_segment(df, start, end, context):
    ctx_start = max(0, start - context)
    ctx_end   = min(len(df) - 1, end + context)
    reasons   = set(",".join(df.loc[start:end, "hard_reasons"]).split(",")) - {""}
    return {
        "start_idx":   start,
        "end_idx":     end,
        "ctx_start":   ctx_start,
        "ctx_end":     ctx_end,
        "reasons":     sorted(reasons),
        "n_hard":      end - start + 1,
        "notes":       df.iloc[ctx_start:ctx_end+1],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Select hard parts from fingering TSV")
    parser.add_argument("tsv", help="Path to fingering TSV")
    parser.add_argument("--rules", default=None,
                        help="Comma-separated rules to apply (default: all). "
                             f"Available: {','.join(RULES)}")
    parser.add_argument("--context", type=int, default=4,
                        help="Context notes around hard segments (default: 4)")
    parser.add_argument("--output", default=None,
                        help="Save hard-note TSV to this path")
    args = parser.parse_args()

    df = load_fingering_tsv(args.tsv)
    rules = [r.strip() for r in args.rules.split(",")] if args.rules else None
    df = select_hard_parts(df, rules)

    n_hard = df["is_hard"].sum()
    print(f"{Path(args.tsv).name}: {n_hard}/{len(df)} notes flagged as hard")

    segs = get_hard_segments(df, args.context)
    print(f"{len(segs)} hard segments:")
    for s in segs:
        print(f"  notes {s['start_idx']}–{s['end_idx']}  reasons: {s['reasons']}")

    if args.output:
        df[df["is_hard"]].to_csv(args.output, sep="\t", index=False)
        print(f"Saved hard notes → {args.output}")

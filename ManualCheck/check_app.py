"""
Streamlit app for manual fingering verification.

Workflow:
  1. Select a recording (TSV with fingering)
  2. Rule-based selector highlights hard segments
  3. User steps through segments, watches video, edits finger labels
  4. Saves corrected TSV

Launch: streamlit run ManualCheck/check_app.py
"""
import os
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DIR.parent
sys.path.insert(0, str(_DIR))

from hard_part_selector import (
    load_fingering_tsv, select_hard_parts, get_hard_segments,
    RULES, RULE_DESCRIPTIONS, DEFAULT_RULES,
)

st.set_page_config(layout="wide", page_title="PianoVAM Fingering Checker")

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note(pitch) -> str:
    try:
        p = int(pitch)
        return f"{_NOTE_NAMES[p % 12]}{p // 12 - 1}"
    except (TypeError, ValueError):
        return str(pitch)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_ROOT   = "/workspace/PianoVAM_v1.0"
FINGERING_DIR  = os.path.join(DATASET_ROOT, "Fingering_resync")  # re-extracted Sep 4/5
VIDEO_DIR      = os.path.join(DATASET_ROOT, "Video")
METADATA_PATH  = os.path.join(DATASET_ROOT, "metadata.json")
OUTPUT_DIR     = os.path.join(DATASET_ROOT, "Fingering_Checked")

FINGER_LABELS = {
    "L": [("L1 (thumb)", 1), ("L2 (index)", 2), ("L3 (middle)", 3),
          ("L4 (ring)", 4), ("L5 (pinky)", 5)],
    "R": [("R1 (thumb)", 1), ("R2 (index)", 2), ("R3 (middle)", 3),
          ("R4 (ring)", 4), ("R5 (pinky)", 5)],
}

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "seg_idx":     0,
        "note_idx":    0,
        "df":          None,
        "df_orig":     None,
        "segments":    [],
        "recording":   None,
        "tsv_path":    None,
        "dirty":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ---------------------------------------------------------------------------
# Sidebar: recording selection + rule configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("PianoVAM Fingering Checker")

    # Find available TSV files
    tsv_files = []
    for d in [FINGERING_DIR, os.path.join(DATASET_ROOT, "Fingering")]:
        if os.path.isdir(d):
            tsv_files += sorted(Path(d).glob("*.tsv"))

    if not tsv_files:
        st.warning(f"No TSV files found in {FINGERING_DIR} or Fingering/.\n"
                   "Run the HMM interpolation step first.")
        tsv_files = []

    tsv_names = [f.name for f in tsv_files]
    selected_tsv_name = st.selectbox("Recording (TSV)", tsv_names or ["— none —"])

    st.markdown("---")
    st.subheader("Hard-part rules")
    enabled_rules = []
    for rule, desc in RULE_DESCRIPTIONS.items():
        if st.checkbox(desc, value=(rule in DEFAULT_RULES), key=f"rule_{rule}"):
            enabled_rules.append(rule)

    context_notes = st.slider("Context notes around hard segment", 2, 10, 4)

    load_btn = st.button("Load / Refresh", type="primary")

    st.markdown("---")
    save_btn = st.button("💾 Save corrections", disabled=not st.session_state.dirty)


# ---------------------------------------------------------------------------
# Load recording
# ---------------------------------------------------------------------------
if load_btn and selected_tsv_name and selected_tsv_name != "— none —":
    tsv_path = next((f for f in tsv_files if f.name == selected_tsv_name), None)
    if tsv_path:
        df = load_fingering_tsv(str(tsv_path))
        df = select_hard_parts(df, enabled_rules or None)
        segments = get_hard_segments(df, context_notes)
        st.session_state.df        = df
        st.session_state.df_orig   = df.copy()
        st.session_state.segments  = segments
        st.session_state.seg_idx   = 0
        st.session_state.note_idx  = 0
        st.session_state.tsv_path  = str(tsv_path)
        st.session_state.recording = tsv_path.stem
        st.session_state.dirty     = False
        st.success(f"Loaded {len(df)} notes, {df['is_hard'].sum()} hard, {len(segments)} segments.")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
if save_btn and st.session_state.df is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, st.session_state.recording + ".tsv")
    df_save = st.session_state.df.drop(columns=["finger_int", "is_hard", "hard_reasons"],
                                        errors="ignore")
    df_save.to_csv(out_path, sep="\t", index=False)
    st.session_state.dirty = False
    st.sidebar.success(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if st.session_state.df is None:
    st.info("Select a recording and click **Load / Refresh** to begin.")
    st.stop()

df       = st.session_state.df
segments = st.session_state.segments
recording = st.session_state.recording

st.header(f"Recording: {recording}")

col_stats1, col_stats2, col_stats3 = st.columns(3)
col_stats1.metric("Total notes",    len(df))
col_stats2.metric("Hard notes",     int(df["is_hard"].sum()))
col_stats3.metric("Hard segments",  len(segments))

if not segments:
    st.success("No hard segments found with selected rules.")
    disp = df[["onset", "note", "hand", "finger", "hard_reasons"]].head(50).copy()
    disp["note"] = disp["note"].apply(midi_to_note)
    st.dataframe(disp)
    st.stop()

st.markdown("---")

# Segment navigation
col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])
with col_nav1:
    if st.button("◀ Prev segment") and st.session_state.seg_idx > 0:
        st.session_state.seg_idx -= 1
with col_nav3:
    if st.button("Next segment ▶") and st.session_state.seg_idx < len(segments) - 1:
        st.session_state.seg_idx += 1
with col_nav2:
    seg_idx = st.slider("Segment", 0, max(0, len(segments) - 1),
                        st.session_state.seg_idx, label_visibility="collapsed")
    st.session_state.seg_idx = seg_idx

seg = segments[st.session_state.seg_idx]
st.subheader(f"Segment {seg_idx + 1}/{len(segments)}  "
             f"(notes {seg['start_idx']}–{seg['end_idx']})  "
             f"Reasons: {', '.join(seg['reasons'])}")

# ---------------------------------------------------------------------------
# Video player (if available)
# ---------------------------------------------------------------------------
video_path = os.path.join(VIDEO_DIR, recording + ".mp4")
if os.path.exists(video_path):
    # Compute approximate timestamp from onset
    onset_sec = float(seg["notes"].iloc[0].get("onset", 0))
    st.markdown(f"**Video onset ≈ {onset_sec:.1f}s** — seek manually in player")
    with open(video_path, "rb") as vf:
        st.video(vf.read(), start_time=max(0, int(onset_sec) - 2))
else:
    st.info(f"Video not available: {video_path}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Notes table + inline editing
# ---------------------------------------------------------------------------
st.subheader("Notes in segment (with context)")
seg_notes = seg["notes"].copy()

# Display read-only overview
display_cols = ["onset", "note", "hand", "finger", "is_hard", "hard_reasons"]
display_cols = [c for c in display_cols if c in seg_notes.columns]
disp = seg_notes[display_cols].copy()
disp["note"] = disp["note"].apply(midi_to_note)
st.dataframe(
    disp.style.apply(
        lambda row: ["background-color: #ffe0e0" if row.get("is_hard", False) else ""
                     for _ in row],
        axis=1
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Edit panel: step through hard notes in segment
# ---------------------------------------------------------------------------
hard_in_seg = seg_notes[seg_notes["is_hard"]] if "is_hard" in seg_notes.columns else pd.DataFrame()

if len(hard_in_seg) > 0:
    st.subheader("Edit hard notes")

    note_positions = list(hard_in_seg.index)
    note_sel_idx = st.selectbox(
        "Select note to edit",
        range(len(note_positions)),
        format_func=lambda i: (
            f"Note {note_positions[i]}  "
            f"onset={df.loc[note_positions[i], 'onset']:.3f}s  "
            f"{midi_to_note(df.loc[note_positions[i], 'note'])}  "
            f"hand={df.loc[note_positions[i], 'hand']}  "
            f"finger={df.loc[note_positions[i], 'finger']}  "
            f"[{df.loc[note_positions[i], 'hard_reasons']}]"
        ),
    )

    note_idx = note_positions[note_sel_idx]
    row = df.loc[note_idx]

    col_edit1, col_edit2 = st.columns(2)
    with col_edit1:
        new_hand = st.radio("Hand", ["L", "R", "Noinfo"],
                            index=["L", "R", "Noinfo"].index(row["hand"])
                            if row["hand"] in ("L", "R", "Noinfo") else 2,
                            horizontal=True, key=f"hand_{note_idx}")

    with col_edit2:
        current_finger = str(row["finger"])
        finger_opts = ["1", "2", "3", "4", "5", "Noinfo"]
        finger_sel = st.radio("Finger (1=thumb, 5=pinky)", finger_opts,
                               index=finger_opts.index(current_finger)
                               if current_finger in finger_opts else 5,
                               horizontal=True, key=f"finger_{note_idx}")

    if st.button("Apply edit", key=f"apply_{note_idx}"):
        df.at[note_idx, "hand"]   = new_hand
        df.at[note_idx, "finger"] = finger_sel
        st.session_state.df    = df
        st.session_state.dirty = True
        st.success(f"Note {note_idx}: hand={new_hand}, finger={finger_sel}")
        st.rerun()

# ---------------------------------------------------------------------------
# Show all edits summary
# ---------------------------------------------------------------------------
if st.session_state.dirty:
    orig = st.session_state.df_orig
    curr = st.session_state.df
    changed = (orig["finger"] != curr["finger"]) | (orig["hand"] != curr["hand"])
    n_changed = changed.sum()
    if n_changed > 0:
        st.markdown(f"**{n_changed} unsaved edit(s)** — click 💾 Save in sidebar")
        with st.expander("Show changes"):
            st.dataframe(curr[changed][["onset", "note", "hand", "finger"]])

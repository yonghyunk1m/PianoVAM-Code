# PianoVAM Development Progress

Branch: `260422_fix`  
Last updated: 2026-04-23

---

## 1. Dataset Download ✅

- Cloned `yonghyunk1m/PianoVAM-Code` repo
- Downloaded full PianoVAM v1.1 dataset (non-video) via `huggingface_hub`:

| Modality | Files | Size |
|---|---|---|
| Audio (.wav) | 107 | 6.3 GB |
| Handskeleton (.json) | 108 | 12 GB |
| MIDI (.mid) | 107 | 5.8 MB |
| TSV (.tsv) | 108 | 21 MB |
| metadata.json | 1 | — |

- Downloaded **32 resynced Sep 4/5 videos** (2.7 GB) — sync-corrected versions confirmed by HuggingFace v1.1 changelog
- Dataset located at `/workspace/PianoVAM_v1.0/`

---

## 2. `FingeringInterpolation/` — Nakamura HMM ✅

Implements Saitō & Nakamura 2022 (2nd-order HMM fingering completion).  
Train on PIG dataset, apply constrained Viterbi to fill `Noinfo` gaps in PianoVAM.

| File | Role |
|---|---|
| `hmm.py` | `train_hmm`, `constrained_viterbi`, `forward_backward`, `compute_entropy` |
| `pig_loader.py` | Loads PIG dataset (train/test split, per hand) |
| `pianovam_loader.py` | Decodes fingerinfo pkl → per-hand sequences; assigns Noinfo notes by pitch proximity |
| `train.py` | CLI: trains R+L HMMs on PIG → `models/hmm_R.npz`, `hmm_L.npz` |
| `interpolate.py` | CLI: runs constrained Viterbi on PianoVAM, writes updated pkl + TSV |
| `evaluate.py` | Reproduces paper table (random / specific / model-rec strategies) |

### Usage

```bash
# 1. Train on PIG
python FingeringInterpolation/train.py --pig-root /path/to/PIG

# 2. Apply to PianoVAM
python FingeringInterpolation/interpolate.py --batch \
    --dataset-root PianoVAM_v1.0 \
    --output-dir   PianoVAM_v1.0/Fingering_HMM

# 3. Evaluate on PIG test set
python FingeringInterpolation/evaluate.py --pig-root /path/to/PIG
```

> **Pending**: PIG dataset download — requires manual registration at  
> https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/register.php

---

## 3. `FingeringDetection/reextract_headless.py` ✅

Headless re-extraction of fingering for the resynced Sep 4/5 videos.  
Runs the full ASDF pipeline (MediaPipe → `handfingercorresponder` → `decide_fingering`) without Streamlit.

- Self-contained `miditotoken` avoids hardcoded paths
- `keyboardcoordinateinfo.pkl` already contains all 32 Sep 4/5 entries — no re-calibration needed

```bash
# All Sep 4/5 videos
python FingeringDetection/reextract_headless.py --all-sep45

# Single video (test)
python FingeringDetection/reextract_headless.py --video 2024-09-04_16-13-44

# Skip MediaPipe if cached handlist pkl already exists
python FingeringDetection/reextract_headless.py --all-sep45 --skip-mediapipe
```

Output:
- MediaPipe cache → `PianoVAM_v1.0/mediapipe_cache_resync/`
- Fingerinfo pkl → `PianoVAM_v1.0/fingering_pickles_resync/`

> **Pending**: Actually running it — MediaPipe on 32 videos takes several hours of compute.

---

## 4. `ManualCheck/` — Manual Verification Tool ✅

### `hard_part_selector.py`

Rule-based selector that flags notes needing human review.

| Rule | Catches |
|---|---|
| `impossible_fingering` | Finger cross w/o thumb; chord span > 15 semitones |
| `fast_jump` | ≥ 15 semitone jump in < 180 ms (hand blur in video) |
| `fast_phrase` | ≥ 4 consecutive notes with IOI < 100 ms |
| `hand_overlap` | L/R pitch regions intersect within 200 ms window |
| `rapid_alternation` | Tremolo-like L/R alternation (> 8 notes/sec) |
| `noinfo` | Note with no finger assigned |
| `noinfo_cluster` | 3+ consecutive Noinfo notes (tracking lost) |
| `stepwise_order_violation` | Finger direction wrong in stepwise motion (w/o thumb cross) |

**Results on 11 GT-annotated recordings: 23.2% flagged overall**

| Piece type | Hard % |
|---|---|
| Slow / simple (Gymnopedie, Kiss the Rain) | ~14% |
| Moderate (Scarlatti, Chopin Waltz) | ~17–21% |
| Complex (Clair de Lune, Schumann) | ~29% |
| Virtuosic (Jeux d'eau, Kapustin) | ~33–46% |

```bash
# CLI usage
python ManualCheck/hard_part_selector.py <fingering.tsv> --summary
python ManualCheck/hard_part_selector.py <fingering.tsv> --rules fast_jump,fast_phrase,hand_overlap
```

### `check_app.py` — Streamlit Review UI

Steps through hard segments, shows video at the right timestamp, supports inline hand/finger editing, and saves corrected TSV.

```bash
streamlit run ManualCheck/check_app.py
```

---

## Workflow Overview

```
1. Download dataset              ✅ Done
        ↓
2. Re-extract fingering          ⏳ Ready to run (needs compute)
   (Sep 4/5 resynced videos)
        ↓
3. Manual check — hard parts     ⏳ Ready (needs fingering data)
   (rule-based selection)
        ↓
4. HMM interpolation             ⏳ Needs PIG dataset
   (Noinfo → predicted finger)
        ↓
5. Manual check — HMM output     ⏳ Needs step 4
```

## Pending Items

| Item | Blocker |
|---|---|
| Run MediaPipe re-extraction (Sep 4/5) | Compute time (~hours) |
| PIG dataset download | Manual registration required |
| Train HMM on PIG | PIG dataset |
| Apply HMM to PianoVAM | Fingering pickles + PIG model |
| Full hard-part evaluation (all 107 recordings) | Fingering zip upload |

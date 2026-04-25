# PianoVAM Development Progress

Branch: `260422_fix`  
Last updated: 2026-04-25

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

## 2. `FingeringDetection/reextract_headless.py` ✅

Headless re-extraction of fingering for the resynced Sep 4/5 videos.  
Runs the full ASDF pipeline without Streamlit:
**MediaPipe → `handpositiondetector` → `handfingercorresponder` → `decide_fingering`**

### Pipeline steps

| Step | Function | Source |
|---|---|---|
| Hand landmark extraction | `run_mediapipe()` | `mediapipe.HandLandmarker` |
| Skeleton / floating frame detection | `modelskeleton`, `depthlist`, `detectfloatingframes` | `detection/floatinghands.py` |
| MIDI tokenization | `_miditotoken_simplified()` | custom (avoids hardcoded paths) |
| Key-hand correspondence | `handfingercorresponder()` | `detection/midicomparison.py` |
| Finger decision | `decide_fingering()` | `detection/decider.py` |

### Bug fixed: keyboard geometry was flipped

`_build_keyboard()` unpacked `ld`/`rd` corners in the wrong order, silently mirroring the keyboard left-right. All fingertips mapped to the opposite end of the keyboard, causing ~58% Noinfo before the fix vs **15.7% after**.

### Results (30 recordings extracted)

- **28/32** successfully extracted  
- **2 failures** (`21-04-38` special/blurry, `21-26-38` 2-min piece): `modelskeleton` crashes when only one hand appears in the video
- Overall Noinfo: **15.7%** across 71,489 notes
- Output: `PianoVAM_v1.0/fingering_pickles_resync/` and `PianoVAM_v1.0/Fingering_resync/`

```bash
python FingeringDetection/reextract_headless.py --all-sep45
python FingeringDetection/reextract_headless.py --video 2024-09-04_16-13-44 --skip-mediapipe
```

---

## 3. `FingeringInterpolation/` — Nakamura HMM ✅

Implements Nakamura et al. 2020 / Saitō & Nakamura 2022 (2nd-order HMM fingering completion).  
Trained on PIG dataset v1.02; applies constrained Viterbi to fill `Noinfo` gaps.

### Model (correctly matches Nakamura 2020 source code)

- **Emission**: physical keyboard position interval (dX, dY) → **93 bins** (not semitones)
  - dX = white-key columns apart, clipped to ±15
  - dY = black/white key type difference
  - Formula: `dkey = 3*(dX+15) + dY + 1`
- **2-step emission**: `outProb2[fpp, fc]` — distance between note n and note n−2
- **Timing**: fixed log-penalty **−5** when IOI < 30ms AND finger/pitch directions conflict
- **Weights**: α1=0.556, α2=0.407 (Bayesian-optimised per Table 3)
- **Transition smoothing**: λ1=0.474 (2nd→1st order interpolation)
- **Smoothing**: EPS=1e-3 (matches `SmoothInit`)

### Training data (PIG v1.02)

- Train: Miscellaneous set — 120 pieces, `#Fingering < 4` (159 sequences/hand)
- Test: Bach + Mozart + Chopin sets — 30 pieces, `#Fingering ≥ 4` (150 sequences/hand)
- PIG v1.02 downloaded from Google Drive

### Evaluation results

| Strategy | Paper R/L | Our R/L |
|---|---|---|
| R=0 (no labels) | ~67%/~68% | 30.0% / 43.3% |
| Random 40% | ~88%/~88% | 62.9% / 69.9% |
| Middle+ModelRec 50% | ~94%/~94% | 73.3% / 82.0% |

Remaining gap vs paper due to Mgen (multi-performer average) vs single-performer evaluation.

```bash
python FingeringInterpolation/train.py --pig-root /workspace/PIG
python FingeringInterpolation/evaluate.py --pig-root /workspace/PIG
python FingeringInterpolation/interpolate.py --batch \
    --dataset-root PianoVAM_v1.0 --output-dir PianoVAM_v1.0/Fingering_HMM
```

---

## 4. `ManualCheck/` — Manual Verification Tool ✅

### `hard_part_selector.py` — 7 rule-based flags

| Rule | Catches |
|---|---|
| `impossible_fingering` | Finger cross w/o thumb (IOI any); chord span exceeds anatomical limit |
| `fast_jump` | ≥15 semitone jump in <180ms (hand blurry in video) |
| `hand_overlap` | L/R pitch regions intersect within 200ms |
| `rapid_alternation` | Tremolo-like L/R alternation (>8 notes/sec) |
| `noinfo` | Note with no finger assigned |
| `noinfo_cluster` | 3+ consecutive Noinfo (tracking completely lost) |
| `stepwise_order_violation` | Finger direction wrong in stepwise motion (w/o thumb cross) |

**Chord span thresholds (semitones):**
`1-2:7, 1-3:12, 1-4:14, 1-5:16, 2-3:4, 2-4:7, 2-5:10, 3-4:4, 3-5:7, 4-5:6`

**Results on 11 GT-annotated recordings: 17.5% flagged overall**

### `check_app.py` — Streamlit Review UI

Steps through hard segments, shows video at the right timestamp, displays notes as **C4/G#4** style, supports inline hand/finger editing, saves corrected TSV.

```bash
streamlit run ManualCheck/check_app.py --server.port 8501 --server.address 0.0.0.0
# Expose via Cloudflare Quick Tunnel:
/opt/portal-aio/tunnel_manager/cloudflared tunnel --url http://localhost:8501
```

---

## Workflow Overview

```
1. Download dataset              ✅ Done
        ↓
2. Re-extract fingering          ✅ Done (28/32 videos, 15.7% Noinfo)
   (Sep 4/5 resynced videos)
        ↓
3. Manual check — hard parts     ✅ Ready (check_app.py running)
   (rule-based selection)
        ↓
4. HMM interpolation             ✅ Model trained (PIG v1.02)
   (Noinfo → predicted finger)       Apply: interpolate.py
        ↓
5. Manual check — HMM output     ⏳ After step 4
```

## Pending Items

| Item | Status |
|---|---|
| Manual verification of Sep 4/5 fingerings | In progress (check_app running) |
| Apply HMM to Sep 4/5 fingerinfo | Ready — run `interpolate.py` |
| Manual check of HMM-interpolated output | After HMM application |
| Full hard-part evaluation (all 107 recordings) | Needs full fingering zip upload |

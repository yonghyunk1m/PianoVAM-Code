from __future__ import annotations
"""
Headless fingering re-extraction for resynced Sep 4/5 videos.

Runs the full ASDF pipeline (MediaPipe → keyhandlist → fingerinfo → export)
without the Streamlit UI. Uses keyboard coordinates from keyboardcoordinateinfo.pkl.

Usage (from project root):
    # All Sep 4/5 videos
    python FingeringDetection/reextract_headless.py --all-sep45

    # Single video
    python FingeringDetection/reextract_headless.py --video 2024-09-04_16-13-44

    # With custom paths
    python FingeringDetection/reextract_headless.py --all-sep45 \\
        --video-dir  /workspace/PianoVAM_v1.0/Video \\
        --midi-dir   /workspace/PianoVAM_v1.0/MIDI \\
        --output-dir /workspace/PianoVAM_v1.0/fingering_pickles_resync
"""
import argparse
import os
import sys
import pickle
import time
from pathlib import Path

# ---- path setup ----
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "PreProcessing" / "Fingering-Export"))

# Override ASDF's hardcoded videocapture path before importing detection modules
import detection.main as _main_mod

SEP45_BASENAMES = [
    # Sep 4 - ext-train
    "2024-09-04_16-13-44", "2024-09-04_17-02-04", "2024-09-04_17-07-59",
    "2024-09-04_17-12-33", "2024-09-04_19-09-38", "2024-09-04_19-52-57",
    "2024-09-04_20-20-07", "2024-09-04_20-30-34", "2024-09-04_20-59-20",
    "2024-09-04_21-09-37", "2024-09-04_21-44-42", "2024-09-04_22-00-35",
    "2024-09-04_22-06-40", "2024-09-04_22-13-22", "2024-09-04_22-19-28",
    # Sep 4 - special(blurry)
    "2024-09-04_18-11-36", "2024-09-04_20-09-07", "2024-09-04_21-51-59",
    # Sep 5 - ext-train
    "2024-09-05_13-25-10", "2024-09-05_13-36-00", "2024-09-05_13-50-00",
    "2024-09-05_20-46-25", "2024-09-05_20-51-00", "2024-09-05_21-01-27",
    "2024-09-05_21-07-35", "2024-09-05_21-13-49", "2024-09-05_21-18-58",
    "2024-09-05_21-26-38", "2024-09-05_21-31-00", "2024-09-05_21-37-08",
    "2024-09-05_22-11-07",
    # Sep 5 - special(blurry)
    "2024-09-05_21-04-38",
]


def get_video_fps(video_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 60.0


def run_mediapipe(video_path: str, keyboard_coords: list, output_pkl_dir: str) -> tuple[str, str]:
    """
    Run MediaPipe hand landmark extraction on a video.
    Returns (handlist_pkl_path, floatingframes_pkl_path).
    """
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from tqdm import tqdm

    from detection.floatinghands import (
        handclass, modelskeleton, depthlist, faultyframes,
        detectfloatingframes, generatekeyboard,
    )

    basename = Path(video_path).stem
    conf_str = (
        f"{_main_mod.min_hand_detection_confidence*100}"
        f"{_main_mod.min_hand_presence_confidence*100}"
        f"{_main_mod.min_tracking_confidence*100}"
    )

    # Build keyboard geometry — coords order: [lu, ru, ld, rd, ...]
    lu, ru, ld, rd, blackratio, ldistortion, rdistortion, cdistortion = keyboard_coords
    keyboard = generatekeyboard(lu=lu, ru=ru, ld=ld, rd=rd,
                                 blackratio=blackratio, ldistortion=ldistortion,
                                 rdistortion=rdistortion, cdistortion=cdistortion)

    # Setup MediaPipe detector
    base_options = python.BaseOptions(model_asset_path=str(_SCRIPT_DIR / "hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=_main_mod.min_hand_detection_confidence,
        min_hand_presence_confidence=_main_mod.min_hand_presence_confidence,
        min_tracking_confidence=_main_mod.min_tracking_confidence,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate  = video.get(cv2.CAP_PROP_FPS)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    ratio  = height / width

    handlist     = []
    nohandframes = []

    print(f"  MediaPipe: {frame_count} frames @ {frame_rate:.1f} fps")
    start = time.time()

    for frame_idx in tqdm(range(frame_count), desc="  MediaPipe", unit="fr"):
        ret, frame_bgr = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / frame_rate)
        result = detector.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        if len(result.handedness) == 0:
            nohandframes.append(frame_idx)

        for j in range(len(result.handedness)):
            for lm in result.hand_landmarks[j]:
                lm.x = lm.x * 2 - 1
                lm.y = lm.y * 2 - 1

        hands = [
            handclass(
                handtype=result.handedness[j][0].category_name,
                handlandmark=result.hand_landmarks[j],
                handframe=frame_idx,
            )
            for j in range(len(result.handedness))
        ]
        handlist.append(hands)

    video.release()
    detector.close()

    elapsed = time.time() - start
    print(f"  MediaPipe done in {elapsed/60:.1f} min  ({frame_count/elapsed:.0f} fps)")

    # Post-processing
    print("  Computing skeleton models and floating frame detection...")
    lhmodel, rhmodel = modelskeleton(handlist)
    depthlist(handlist, lhmodel, rhmodel, ratio)
    faulty = faultyframes(handlist)
    floating = detectfloatingframes(handlist, frame_count, faulty, lhmodel, rhmodel, ratio=ratio)

    # Save
    os.makedirs(output_pkl_dir, exist_ok=True)
    handlist_path = os.path.join(output_pkl_dir, f"handlist_{basename}_{conf_str}.pkl")
    floating_path = os.path.join(output_pkl_dir, f"floatingframes_{basename}_{conf_str}.pkl")

    with open(handlist_path, "wb") as f:
        pickle.dump(handlist, f)
    with open(floating_path, "wb") as f:
        pickle.dump(floating, f)

    print(f"  Saved handlist → {handlist_path}")
    print(f"  Saved floating → {floating_path}")
    print(f"  No-hand frames: {len(nohandframes)}/{frame_count}")

    return handlist_path, floating_path


def _miditotoken_simplified(midi_path: str, fps: int = 60) -> list | None:
    """
    Self-contained MIDI→token list conversion (mirrors midicomparison.miditotoken 'simplified').
    Uses explicit midi_path instead of hardcoded directory.
    """
    import math
    from symusic import Score
    from miditok import REMI, TokenizerConfig

    midi = Score(midi_path)
    tempo = midi.tempos[0].qpm if midi.tempos else 120
    beatres = math.floor(fps * 60 / tempo)

    config = TokenizerConfig(
        num_velocities=16, use_chords=False, use_programs=True,
        beat_res={(0, 100): beatres},
    )
    tokenizer = REMI(config)
    tokens = tokenizer(midi)

    tokenlist = []
    tempindex = 0
    for i in range(len(tokens.tokens)):
        if "Duration" in tokens.tokens[i]:
            tokenlist.append(tokens.tokens[tempindex: i + 1])
            tempindex = i + 1

    # Simplified mode: strip Bar tokens, compute absolute frame positions
    for token in tokenlist:
        while "Bar" in token[0]:
            token.pop(0)
            token.append("Bar")
    positionindex = 0
    for i in range(len(tokenlist)):
        if not "Position" in tokenlist[i][0]:
            tokenlist[i].insert(0, tokenlist[positionindex][0])
        else:
            positionindex = i

    barcounter = -1
    for token in tokenlist:
        while "Bar" in token[-1]:
            token.pop(-1)
            barcounter += 1
        token[0] = int(token[0][9:]) + 4 * beatres * barcounter

    for token in tokenlist:
        token.pop(1)
        token[1] = int(token[1][6:]) - 21
        token.pop(2)
        token[2] = (
            token[0]
            + int(token[2].split(".")[0][9:]) * beatres
            + int(token[2].split(".")[1])
        )
    for i in range(len(tokenlist)):
        tokenlist[i].append(i)

    return tokenlist


def run_fingering_extraction(
    basename: str,
    handlist_path: str,
    floating_path: str,
    midi_path: str,
    keyboard_coords: list,
    output_pkl_dir: str,
) -> str:
    """
    Run handfingercorresponder → decide_fingering.
    Saves fingerinfo pkl. Returns fingerinfo pkl path.
    """
    from detection.midicomparison import tokentoframeinfo, handfingercorresponder
    from detection.floatinghands import handpositiondetector
    from detection.decider import decide_fingering

    with open(handlist_path, "rb") as f:
        handlist = pickle.load(f)
    with open(floating_path, "rb") as f:
        floatingframes = pickle.load(f)

    frame_count = len(handlist)
    fps = 60  # ASDF default

    keyboard = _build_keyboard(keyboard_coords)

    # Per-frame hand+finger info (which keys each fingertip touches)
    framehandfingerlist = [
        handpositiondetector(hands, floatingframes, keyboard)
        for hands in handlist
    ]

    # MIDI → token list (using self-contained version with explicit path)
    tokenlist = _miditotoken_simplified(midi_path, fps)
    if tokenlist is None or len(tokenlist) == 0:
        raise RuntimeError(f"miditotoken returned empty for {basename}")

    framemidilist  = tokentoframeinfo(tokenlist, frame_count)
    tokenlist_copy = [list(t) for t in tokenlist]

    # Key-hand correspondence
    keyhandlist = handfingercorresponder(
        framemidilist, framehandfingerlist, keyboard, tokenlist_copy,
    )

    # Decide fingering (needs a fresh tokenlist copy — decide_fingering mutates it)
    tokenlist_for_decider = _miditotoken_simplified(midi_path, fps)
    fingerinfo, undecided = decide_fingering(tokenlist_for_decider, keyhandlist)

    os.makedirs(output_pkl_dir, exist_ok=True)
    out_path = os.path.join(output_pkl_dir, f"fingerinfo_{basename}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(fingerinfo, f)

    n_noinfo = sum(1 for f in fingerinfo if f == "Noinfo" or f is None)
    print(f"  Fingering done: {len(fingerinfo)} notes, {n_noinfo} Noinfo, {len(undecided)} undecided")
    print(f"  Saved → {out_path}")

    return out_path


def _build_keyboard(coords):
    from detection.floatinghands import generatekeyboard
    # coords order matches datagenerate(): [lu, ru, ld, rd, blackratio, ldistortion, rdistortion, cdistortion]
    lu, ru, ld, rd, blackratio, ldistortion, rdistortion, cdistortion = coords
    return generatekeyboard(lu=lu, ru=ru, ld=ld, rd=rd,
                             blackratio=blackratio, ldistortion=ldistortion,
                             rdistortion=rdistortion, cdistortion=cdistortion)


def process_one(
    basename: str,
    video_dir: str,
    midi_dir: str,
    mediapipe_cache_dir: str,
    output_dir: str,
    keyboard_info: dict,
    skip_mediapipe: bool = False,
) -> dict:
    video_path = os.path.join(video_dir, basename + ".mp4")
    midi_path  = os.path.join(midi_dir,  basename + ".mid")

    if not os.path.exists(video_path):
        return {"status": "skip", "reason": "no video"}
    if not os.path.exists(midi_path):
        return {"status": "skip", "reason": "no MIDI"}
    if basename not in keyboard_info:
        return {"status": "skip", "reason": "no keyboard coords"}

    keyboard_coords = keyboard_info[basename]
    conf_str = (
        f"{_main_mod.min_hand_detection_confidence*100}"
        f"{_main_mod.min_hand_presence_confidence*100}"
        f"{_main_mod.min_tracking_confidence*100}"
    )
    handlist_path = os.path.join(mediapipe_cache_dir, f"handlist_{basename}_{conf_str}.pkl")
    floating_path = os.path.join(mediapipe_cache_dir, f"floatingframes_{basename}_{conf_str}.pkl")

    t0 = time.time()

    # Step 1: MediaPipe extraction
    if skip_mediapipe and os.path.exists(handlist_path) and os.path.exists(floating_path):
        print(f"  [skip MediaPipe] Using cached: {handlist_path}")
    else:
        print(f"  Running MediaPipe on {basename}...")
        run_mediapipe(video_path, keyboard_coords, mediapipe_cache_dir)

    # Step 2: Fingering extraction
    print(f"  Running fingering extraction...")
    try:
        fingerinfo_path = run_fingering_extraction(
            basename, handlist_path, floating_path,
            midi_path, keyboard_coords, output_dir,
        )
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    elapsed = time.time() - t0
    return {"status": "ok", "fingerinfo": fingerinfo_path, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Headless ASDF re-extraction for Sep 4/5")
    parser.add_argument("--all-sep45", action="store_true",
                        help="Process all Sep 4 & 5 recordings")
    parser.add_argument("--video", type=str,
                        help="Single recording basename (e.g. 2024-09-04_16-13-44)")
    parser.add_argument("--video-dir",  default="/workspace/PianoVAM_v1.0/Video")
    parser.add_argument("--midi-dir",   default="/workspace/PianoVAM_v1.0/MIDI")
    parser.add_argument("--cache-dir",  default="/workspace/PianoVAM_v1.0/mediapipe_cache_resync",
                        help="Where to store handlist/floatingframes pkl files")
    parser.add_argument("--output-dir", default="/workspace/PianoVAM_v1.0/fingering_pickles_resync",
                        help="Where to store fingerinfo pkl files")
    parser.add_argument("--skip-mediapipe", action="store_true",
                        help="Skip MediaPipe if cached handlist pkl exists")
    args = parser.parse_args()

    # Load keyboard coords
    kbd_path = str(_SCRIPT_DIR / "keyboardcoordinateinfo.pkl")
    with open(kbd_path, "rb") as f:
        keyboard_info = pickle.load(f)
    print(f"Loaded keyboard coords for {len(keyboard_info)} recordings.")

    # Determine which basenames to process
    if args.all_sep45:
        basenames = SEP45_BASENAMES
    elif args.video:
        basenames = [args.video]
    else:
        parser.print_help()
        sys.exit(1)

    # Filter to those with videos available
    available = [b for b in basenames
                 if os.path.exists(os.path.join(args.video_dir, b + ".mp4"))]
    missing   = [b for b in basenames if b not in available]
    if missing:
        print(f"[WARN] {len(missing)} videos not found (download first):")
        for b in missing:
            print(f"  {b}")

    print(f"\nProcessing {len(available)} recordings...\n")

    ok = fail = skip = 0
    for i, basename in enumerate(available, 1):
        print(f"[{i}/{len(available)}] {basename}")
        result = process_one(
            basename,
            video_dir=args.video_dir,
            midi_dir=args.midi_dir,
            mediapipe_cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            keyboard_info=keyboard_info,
            skip_mediapipe=args.skip_mediapipe,
        )
        if result["status"] == "ok":
            ok += 1
            print(f"  Done in {result['elapsed']/60:.1f} min\n")
        elif result["status"] == "skip":
            skip += 1
            print(f"  Skipped: {result['reason']}\n")
        else:
            fail += 1
            print(f"  FAILED: {result['reason']}\n")

    print(f"\n--- Done ---")
    print(f"  OK: {ok}, Skipped: {skip}, Failed: {fail}")
    print(f"  Fingerinfo pkl → {args.output_dir}")
    print(f"  MediaPipe cache → {args.cache_dir}")


if __name__ == "__main__":
    main()

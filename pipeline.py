#!/usr/bin/env python3
"""
PianoVAM review pipeline.

Commands:
  python pipeline.py setup          -- download MIDI + audio + TSV + Handskeleton (~19 GB)
  python pipeline.py batch [N]      -- download + prepare batch N videos (default: 1)
  python pipeline.py prep <trial>   -- (re)load one recording into the review app
  python pipeline.py clean [N]      -- delete downloaded videos for batch N
  python pipeline.py extract        -- run MediaPipe fingering extraction on Sep 4/5 videos
  python pipeline.py export-tsv     -- export fingerinfo pkls → TSV (runs after extract)
  python pipeline.py status         -- show what has been downloaded / reviewed
"""

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

import requests

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ID   = "PianoVAM/PianoVAM_v1.0"
HF_BASE      = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
HF_API       = f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/main"
BATCH_LIMIT  = 25 * 1024**3   # 25 GB per video batch (two batches = ~26 GB total)

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "PianoVAM_v1.0"
VIDEO_DIR = DATA_DIR / "Video"
MIDI_DIR  = DATA_DIR / "MIDI"
AUDIO_DIR = DATA_DIR / "Audio"
TSV_DIR   = DATA_DIR / "TSV"
PUBLIC_V  = ROOT / "annotate/public/videos"
PUBLIC_D  = ROOT / "annotate/public/data"

VERDICTS_FILE = PUBLIC_D / "human_verdicts.json"
STATE_FILE    = ROOT / ".pipeline_state.json"

NON_VIDEO_MODS = ["MIDI", "Audio", "TSV", "Handskeleton"]

SEP45_TRIALS = [
    "2024-09-04_16-13-44", "2024-09-04_17-02-04", "2024-09-04_17-07-59",
    "2024-09-04_17-12-33", "2024-09-04_18-11-36", "2024-09-04_19-09-38",
    "2024-09-04_19-52-57", "2024-09-04_20-09-07", "2024-09-04_20-20-07",
    "2024-09-04_20-30-34", "2024-09-04_20-59-20", "2024-09-04_21-09-37",
    "2024-09-04_21-44-42", "2024-09-04_21-51-59", "2024-09-04_22-00-35",
    "2024-09-04_22-06-40", "2024-09-04_22-13-22", "2024-09-04_22-19-28",
    "2024-09-05_13-25-10", "2024-09-05_13-36-00", "2024-09-05_13-50-00",
    "2024-09-05_20-46-25", "2024-09-05_20-51-00", "2024-09-05_21-01-27",
    "2024-09-05_21-04-38", "2024-09-05_21-07-35", "2024-09-05_21-13-49",
    "2024-09-05_21-18-58", "2024-09-05_21-26-38", "2024-09-05_21-31-00",
    "2024-09-05_21-37-08", "2024-09-05_22-11-07",
]
FINGERING_ENV  = "pianovam-fingering"
FINGERING_PKL  = DATA_DIR / "fingering_pickles_resync"   # fingerinfo pkl output (Sep 4/5 re-extracted)
FINGERING_TSV  = DATA_DIR / "Fingering_resync"           # exported TSV output (Sep 4/5)
FINGERING_PKL_ALL = DATA_DIR / "fingering_pickles"       # keyhandlist pkls from Google Drive (all trials)
FINGERING_TSV_ALL = DATA_DIR / "Fingering"               # exported TSV output (all trials)

# ── Helpers ───────────────────────────────────────────────────────────────────

def hf_list(subdir=""):
    url = HF_API + (f"/{subdir}" if subdir else "")
    r = requests.get(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.json()

def download(url, dest: Path, desc=""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  ↓ {desc or dest.name}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
        tmp.rename(dest)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"reviewed": [], "deleted": []}

def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2))

def get_trial(path_str):
    return Path(path_str).stem   # "Video/2024-09-04_16-13-44.mp4" → "2024-09-04_16-13-44"

# ── Commands ──────────────────────────────────────────────────────────────────

def pkl_exists(trial):
    """Return True if fingerinfo pkl exists for this trial."""
    import glob
    return bool(glob.glob(str(FINGERING_PKL / f"fingerinfo_{trial}*.pkl")))

def tsv_exists(trial):
    return (FINGERING_TSV / f"{trial}.tsv").exists()

def cmd_extract():
    """Run reextract_headless.py on Sep 4/5 videos that don't have a pkl yet."""
    missing = [t for t in SEP45_TRIALS if not pkl_exists(t)]
    if not missing:
        print("✓ All Sep 4/5 fingerinfo pkls already extracted.")
        return

    not_downloaded = [t for t in missing if not (VIDEO_DIR / f"{t}.mp4").exists()]
    if not_downloaded:
        print(f"[error] {len(not_downloaded)} Sep 4/5 videos not yet downloaded:")
        for t in not_downloaded:
            print(f"  {t}")
        sys.exit(1)

    FINGERING_PKL.mkdir(parents=True, exist_ok=True)

    print(f"Extracting fingering for {len(missing)} Sep 4/5 videos...")
    result = subprocess.run([
        "conda", "run", "-n", FINGERING_ENV, "--no-capture-output",
        "python", str(ROOT / "FingeringDetection/reextract_headless.py"),
        "--all-sep45",
        "--video-dir",  str(VIDEO_DIR),
        "--midi-dir",   str(MIDI_DIR),
        "--cache-dir",  str(DATA_DIR / "mediapipe_cache_resync"),
        "--output-dir", str(FINGERING_PKL),
    ], cwd=ROOT)

    if result.returncode != 0:
        print("[error] reextract_headless.py failed — check output above.")
        sys.exit(1)

    done = sum(1 for t in SEP45_TRIALS if pkl_exists(t))
    print(f"\n✓ Extracted {done}/{len(SEP45_TRIALS)} Sep 4/5 recordings.")
    cmd_export_tsv()


def cmd_export_tsv():
    """Export fingerinfo pkls (Sep 4/5) → TSV via export_fingering.py.

    The Sep 4/5 fingerinfo pkls live in fingering_pickles_resync/ and need a
    custom export because export_fingering.py only looks in fingering_pickles/.
    We copy them into a temp symlink so the exporter finds them, then clean up.
    """
    missing = [t for t in SEP45_TRIALS if pkl_exists(t) and not tsv_all_exists(t)]
    if not missing:
        print("✓ All Sep 4/5 TSVs already exported.")
        return

    FINGERING_TSV_ALL.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(missing)} Sep 4/5 fingerinfo pkls → TSV...")

    # Copy fingerinfo pkls into fingering_pickles/ so export_fingering.py finds them.
    # fingerinfo_*.pkl is direct (not keyhandlist), so exporter doesn't call decide_fingering.
    FINGERING_PKL_ALL.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    for t in missing:
        import glob as _glob
        srcs = _glob.glob(str(FINGERING_PKL / f"fingerinfo_{t}*.pkl"))
        for src in srcs:
            dst = FINGERING_PKL_ALL / Path(src).name
            if not dst.exists():
                _shutil.copy2(src, dst)

    files_arg = ",".join(missing)
    py = str(Path(subprocess.check_output(
        ["conda", "run", "-n", FINGERING_ENV, "which", "python"],
        text=True).strip()))
    result = subprocess.run([
        py,
        str(ROOT / "PreProcessing/Fingering-Export/export_fingering.py"),
        "--dataset-root", str(DATA_DIR),
        "--output-dir",   str(FINGERING_TSV_ALL),
        "--format", "tsv",
        "--use-tsv-ref",
        "--files", files_arg,
    ], cwd=ROOT)

    if result.returncode != 0:
        print("[warn] export_fingering.py returned non-zero — check output above.")

    done = sum(1 for t in SEP45_TRIALS if tsv_all_exists(t))
    print(f"✓ Exported {done}/{len(SEP45_TRIALS)} Sep 4/5 TSVs to {FINGERING_TSV_ALL}")


def cmd_setup():
    print("=== Downloading non-video files (MIDI, Audio, TSV) ===")
    for mod in NON_VIDEO_MODS:
        entries = hf_list(mod)
        print(f"\n[{mod}] {len(entries)} files")
        dest_dir = DATA_DIR / mod
        for e in entries:
            name = Path(e["path"]).name
            url  = f"{HF_BASE}/{e['path']}"
            download(url, dest_dir / name, name)
    print("\n✓ Setup complete.")


def cmd_batch(n):
    print(f"=== Building video batch plan ===")
    entries = sorted(hf_list("Video"), key=lambda e: e["path"])
    # split into batches by BATCH_LIMIT
    batches, cur, cur_sz = [], [], 0
    for e in entries:
        sz = e.get("size", 0)
        if cur and cur_sz + sz > BATCH_LIMIT:
            batches.append(cur); cur, cur_sz = [], 0
        cur.append(e); cur_sz += sz
    if cur:
        batches.append(cur)

    if n < 1 or n > len(batches):
        print(f"Batch {n} doesn't exist. Available: 1–{len(batches)}")
        sys.exit(1)

    batch = batches[n - 1]
    total_gb = sum(e.get("size", 0) for e in batch) / 1e9
    print(f"Batch {n}: {len(batch)} videos, {total_gb:.1f} GB\n")

    state = load_state()

    PUBLIC_V.mkdir(parents=True, exist_ok=True)
    PUBLIC_D.mkdir(parents=True, exist_ok=True)

    for e in batch:
        trial = get_trial(e["path"])
        if trial in state["deleted"]:
            print(f"  [skip] {trial} already reviewed & deleted")
            continue

        # 1. Download video
        url  = f"{HF_BASE}/{e['path']}"
        dest = VIDEO_DIR / f"{trial}.mp4"
        download(url, dest, f"{trial}.mp4  ({e.get('size',0)/1e6:.0f} MB)")

    # 2. Extract fingering for Sep 4/5 videos in this batch
    sep45_in_batch = [get_trial(e["path"]) for e in batch if get_trial(e["path"]) in SEP45_TRIALS]
    if sep45_in_batch:
        print(f"\nRunning fingering extraction for {len(sep45_in_batch)} Sep 4/5 videos...")
        cmd_extract()

    # 3. Prepare review bundles
    for e in batch:
        trial = get_trial(e["path"])
        if trial not in state["deleted"]:
            cmd_prep(trial, quiet=True)

    print(f"\n✓ Batch {n} ready.")
    print(f"  App URL: check running cloudflared URL (or http://localhost:3333)")
    print(f"  Switch recordings with:  python pipeline.py prep <trial>")
    print(f"  When done reviewing:     python pipeline.py clean {n}")


def tsv_all_exists(trial):
    return (FINGERING_TSV_ALL / f"{trial}.tsv").exists()

def cmd_prep(trial, quiet=False):
    midi  = next(MIDI_DIR.glob(f"*{trial}*"), None) if MIDI_DIR.exists() else None
    audio = next(AUDIO_DIR.glob(f"*{trial}*"), None) if AUDIO_DIR.exists() else None
    # Priority: (1) freshly re-extracted Sep 4/5 TSV, (2) Google Drive archive TSV (all trials)
    if tsv_exists(trial):
        tsv = FINGERING_TSV / f"{trial}.tsv"
    elif tsv_all_exists(trial):
        tsv = FINGERING_TSV_ALL / f"{trial}.tsv"
    else:
        tsv = None
    video = VIDEO_DIR / f"{trial}.mp4"

    if not video.exists():
        print(f"  [error] Video not downloaded: {trial}")
        return

    args = [
        sys.executable,
        str(ROOT / "annotate/prepare_review_data.py"),
        "--video", str(video),
        "--piece", trial,
    ]
    if midi:  args += ["--midi", str(midi)]
    if audio: args += ["--audio", str(audio)]
    if tsv:   args += ["--tsv", str(tsv)]

    if not quiet:
        print(f"Preparing {trial}...")
    result = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"  [warn] prepare_review_data.py failed for {trial}:")
        print(result.stderr[-500:])
    else:
        # Also save a per-trial copy so the app's trial selector can list it.
        notes_json = PUBLIC_D / "notes.json"
        trial_json = PUBLIC_D / f"{trial}.json"
        if notes_json.exists():
            import shutil
            shutil.copy2(notes_json, trial_json)
        if not quiet:
            print(f"✓ {trial} → review app  (notes.json + {trial}.json)")
        else:
            for line in result.stdout.splitlines():
                if "notes:" in line or "hard notes:" in line:
                    print(f"    {trial}: {line.strip()}")


def cmd_clean(n):
    print(f"=== Cleaning batch {n} videos ===")
    entries = sorted(hf_list("Video"), key=lambda e: e["path"])
    batches, cur, cur_sz = [], [], 0
    for e in entries:
        sz = e.get("size", 0)
        if cur and cur_sz + sz > BATCH_LIMIT:
            batches.append(cur); cur, cur_sz = [], 0
        cur.append(e); cur_sz += sz
    if cur:
        batches.append(cur)

    if n < 1 or n > len(batches):
        print(f"Batch {n} doesn't exist.")
        sys.exit(1)

    state = load_state()
    freed = 0
    for e in batches[n - 1]:
        trial = get_trial(e["path"])
        vpath = VIDEO_DIR / f"{trial}.mp4"
        vcopy = PUBLIC_V / f"{trial}.mp4"
        for p in (vpath, vcopy):
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
                print(f"  ✗ {p.name}")
        if trial not in state["deleted"]:
            state["deleted"].append(trial)
    save_state(state)
    print(f"\n✓ Freed {freed/1e9:.1f} GB")


def cmd_status():
    state = load_state()
    entries = hf_list("Video")
    total = len(entries)
    deleted = len(state["deleted"])
    downloaded = sum(1 for e in entries if (VIDEO_DIR / f"{get_trial(e['path'])}.mp4").exists())
    verdicts = {}
    if VERDICTS_FILE.exists():
        verdicts = json.loads(VERDICTS_FILE.read_text())

    print(f"Total recordings : {total}")
    print(f"Downloaded       : {downloaded}")
    print(f"Reviewed+deleted : {deleted}")
    print(f"Verdicts saved   : {len(verdicts)}")
    disk = shutil.disk_usage("/")
    print(f"Disk free        : {disk.free/1e9:.1f} GB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("setup")
    bp = sub.add_parser("batch");  bp.add_argument("n", type=int, nargs="?", default=1)
    pp = sub.add_parser("prep");   pp.add_argument("trial")
    cp = sub.add_parser("clean");  cp.add_argument("n", type=int, nargs="?", default=1)
    sub.add_parser("extract")
    sub.add_parser("export-tsv")
    sub.add_parser("status")

    args = p.parse_args()

    if args.cmd == "setup":       cmd_setup()
    elif args.cmd == "batch":     cmd_batch(args.n)
    elif args.cmd == "prep":      cmd_prep(args.trial)
    elif args.cmd == "clean":     cmd_clean(args.n)
    elif args.cmd == "extract":   cmd_extract()
    elif args.cmd == "export-tsv": cmd_export_tsv()
    elif args.cmd == "status":    cmd_status()
    else: p.print_help()

if __name__ == "__main__":
    main()

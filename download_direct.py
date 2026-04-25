#!/usr/bin/env python3
"""Direct downloader for PianoVAM dataset using correct dataset ID."""
import os
import sys
import time
import requests
from tqdm import tqdm

DATASET_ID = "PianoVAM/PianoVAM_v1"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
OUTPUT_DIR = "/workspace/PianoVAM_v1.0"
MAX_RETRIES = 5

# Modalities to skip (set to [] to download everything)
SKIP_MODALITIES = []  # e.g. ["Video"] to skip video

def get_file_list():
    resp = requests.get(API_URL, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    return [s["rfilename"] for s in resp.json().get("siblings", [])]

def download_file(url, local_path):
    if os.path.exists(local_path):
        return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    temp_path = local_path + ".part"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            os.rename(temp_path, local_path)
            return True
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  Retry {attempt}/{MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {MAX_RETRIES} attempts: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False

files = get_file_list()
# Filter out skipped modalities and special files
to_download = []
for f in files:
    if f in (".gitattributes",):
        continue
    top_dir = f.split("/")[0]
    if top_dir in SKIP_MODALITIES:
        continue
    to_download.append(f)

print(f"Files to download: {len(to_download)}")

ok, fail = 0, 0
for rfile in tqdm(to_download, desc="Downloading"):
    url = BASE_URL + rfile
    local = os.path.join(OUTPUT_DIR, rfile)
    if download_file(url, local):
        ok += 1
    else:
        fail += 1

print(f"\nDone. Success: {ok}, Failed: {fail}")

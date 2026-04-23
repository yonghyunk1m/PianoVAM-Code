"""
Paths for FingeringInterpolation.
Edit these to match your local setup.
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PIG dataset root (download from https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/)
PIG_ROOT = os.path.join(_PROJECT_ROOT, "PIG")

# PianoVAM dataset
DATASET_ROOT    = os.path.join(_PROJECT_ROOT, "PianoVAM_v1.0")
MIDI_DIR        = os.path.join(DATASET_ROOT, "MIDI")
TSV_DIR         = os.path.join(DATASET_ROOT, "TSV")
PICKLES_DIR     = os.path.join(DATASET_ROOT, "fingering_pickles")

# Trained model weights
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_R   = os.path.join(MODEL_DIR, "hmm_R.npz")
MODEL_L   = os.path.join(MODEL_DIR, "hmm_L.npz")

# Output
OUTPUT_DIR = os.path.join(DATASET_ROOT, "Fingering_HMM")

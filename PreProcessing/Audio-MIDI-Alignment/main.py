import os
import glob
import numpy as np
import librosa
import pretty_midi
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 22050
CQT_BINS_PER_OCTAVE = 12
CQT_N_BINS = 48
CQT_FMIN = librosa.midi_to_hz(36)
MIN_DB = -80.0
EPSILON = 1e-10

# Path to the SoundFont file
SF2_PATH = "YDP-GrandPiano-20160804.sf2"

# Parameters
CLIP_SEC = 30.0
CHUNK_SIZE = 1024
BAND_RADIUS_SEC = 2.5
VIEW_RANGE_SEC = 0.05  # ¡¾ 50 ms

# Directory paths
AUDIO_DIR = "./data/audio"
MIDI_DIR = "./data/midi"
OUT_DIR  = "./data/aligned_midi"

# Directories for saving images (waveforms, snippets, etc.)
IMG_DIR_WAVE   = "./data/img/wave"
IMG_DIR_SNIP   = "./data/img/snippet"

# Create directories if they don't exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR_WAVE, exist_ok=True)
os.makedirs(IMG_DIR_SNIP, exist_ok=True)

###############################################################################
# Function to compute CQT
###############################################################################
def compute_maestro_cqt(samples: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Computes the Constant-Q Transform (CQT) for the given audio samples.
    Applies a time-wise normalization to reduce amplitude variations.
    """
    cqt_mag = np.abs(
        librosa.cqt(
            samples,
            sr=sr,
            hop_length=hop_length,
            fmin=CQT_FMIN,
            n_bins=CQT_N_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE
        )
    ).astype(np.float32)

    max_val = cqt_mag.max()
    if max_val < EPSILON:
        max_val = EPSILON

    cqt_db = 20.0 * np.log10(cqt_mag / max_val + EPSILON)
    cqt_db = np.clip(cqt_db, a_min=MIN_DB, a_max=None)
    cqt_amp = 10.0 ** (cqt_db / 20.0)

    # Time-wise normalization
    normed = cqt_amp.copy()
    n_frames = normed.shape[1]
    window_radius = 2
    for t in range(n_frames):
        start_t = max(0, t - window_radius)
        end_t   = min(n_frames, t + window_radius + 1)
        local_mins = [normed[:, w].min() for w in range(start_t, end_t)]
        avg_min = np.mean(local_mins)
        if avg_min < EPSILON:
            avg_min = EPSILON
        normed[:, t] /= avg_min

    return normed.astype(np.float32)

###############################################################################
# Chunked cosine distance matrix calculation
###############################################################################
def compute_cosine_distance_matrix_chunked(X: np.ndarray, Y: np.ndarray, chunk_size=1024) -> np.ndarray:
    """
    Computes the chunked cosine distance matrix between two matrices X and Y.
    """
    N, d = X.shape
    M = Y.shape[0]
    dist_matrix = np.zeros((N, M), dtype=np.float32)

    X_norm = np.sqrt((X**2).sum(axis=1, keepdims=True))
    Y_norm = np.sqrt((Y**2).sum(axis=1, keepdims=True))
    X_norm[X_norm < 1e-15] = 1e-15
    Y_norm[Y_norm < 1e-15] = 1e-15

    Yn = Y / Y_norm

    n_chunks = (N + chunk_size - 1) // chunk_size
    for c in tqdm(range(n_chunks), desc="CosineDist", leave=False):
        i_start = c * chunk_size
        i_end   = min(i_start + chunk_size, N)
        X_block = X[i_start:i_end]
        Xn_block= X_block / X_norm[i_start:i_end]

        dot_vals = Xn_block @ Yn.T
        dist_block = 1.0 - dot_vals
        dist_matrix[i_start:i_end, :] = dist_block

    return dist_matrix

###############################################################################
# Function to estimate penalty for DTW
###############################################################################
def estimate_penalty(dist_matrix: np.ndarray, sample_size=100000) -> float:
    """
    Estimates a penalty value by randomly sampling from the distance matrix.
    """
    n, m = dist_matrix.shape
    total = n * m
    if total <= sample_size:
        return float(dist_matrix.mean())
    idx_i = np.random.randint(0, n, size=sample_size)
    idx_j = np.random.randint(0, m, size=sample_size)
    samples = dist_matrix[idx_i, idx_j]
    return float(samples.mean())

###############################################################################
# Sakoe-Chiba DTW
###############################################################################
def sakoe_chiba_dtw(dist_matrix: np.ndarray, band_radius_sec: float, sr:int,
                    hop_length:int, penalty:float):
    """
    Performs Dynamic Time Warping with Sakoe-Chiba band.
    Returns the warping path and the total cost.
    """
    sec_per_frame = hop_length / sr
    band_frames = int(round(band_radius_sec / sec_per_frame))

    n, m = dist_matrix.shape
    D = np.full((n+1, m+1), np.inf, dtype=np.float32)
    D[0,0] = 0.0
    ptr = np.zeros((n+1, m+1, 2), dtype=np.int32)

    for i in range(1, n+1):
        j_start = max(1, i - band_frames)
        j_end   = min(m, i + band_frames)
        for j in range(j_start, j_end+1):
            cost = dist_matrix[i-1, j-1]
            diag = D[i-1, j-1] + cost
            up   = D[i-1, j] + penalty
            left = D[i, j-1] + penalty

            best = diag
            src  = (i-1, j-1)
            if up < best:
                best = up
                src  = (i-1, j)
            if left < best:
                best = left
                src  = (i, j-1)

            D[i,j] = best
            ptr[i,j] = src

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        (i2, j2) = ptr[i,j]
        i, j = i2, j2
    path.reverse()

    return path, D[n,m]

###############################################################################
# MIDI trimming function
###############################################################################
def trim_midi(pm: pretty_midi.PrettyMIDI, clip_sec: float):
    """
    Trims notes in a PrettyMIDI object to the specified clip_sec.
    """
    for inst in pm.instruments:
        new_notes = []
        for note in inst.notes:
            if note.start < clip_sec:
                if note.end > clip_sec:
                    note.end = clip_sec
                new_notes.append(note)
        inst.notes = new_notes

###############################################################################
# Function to align a single (audio, MIDI) pair
###############################################################################
def align_audio_midi_pair(audio_path, midi_path, out_midi_path,
                          clip_sec=30.0,
                          sr=22050,
                          sf2_path=SF2_PATH,
                          hop_length=64,
                          do_save_plots=True):
    """
    Aligns a WAV audio file and a MIDI file for the first clip_sec seconds.
    Saves the aligned MIDI file to out_midi_path.
    Optionally saves waveform comparison images.
    """
    # Extract base filename
    basename = os.path.splitext(os.path.basename(audio_path))[0]

    # 1) Load the first clip_sec seconds of the audio
    audio_30, sr = librosa.load(audio_path, sr=sr, duration=clip_sec)

    # 2) Load the MIDI and trim to clip_sec
    pm_full = pretty_midi.PrettyMIDI(midi_path)
    pm_30   = pretty_midi.PrettyMIDI(midi_path)
    trim_midi(pm_30, clip_sec=clip_sec)

    # Synthesize the trimmed MIDI audio
    midi_audio_30 = pm_30.fluidsynth(fs=sr, sf2_path=sf2_path).astype(np.float32)
    max_samples   = int(round(clip_sec * sr))
    if len(midi_audio_30) > max_samples:
        midi_audio_30 = midi_audio_30[:max_samples]

    # 3) Zero-pad both audio signals to the same length
    max_len   = max(len(audio_30), len(midi_audio_30))
    audio_30p = np.pad(audio_30, (0, max_len - len(audio_30)), 'constant')
    midi_30p  = np.pad(midi_audio_30, (0, max_len - len(midi_audio_30)), 'constant')

    # 4) Compute CQT for both signals
    cqt_aud = compute_maestro_cqt(audio_30p, sr=sr, hop_length=hop_length)
    cqt_mid = compute_maestro_cqt(midi_30p, sr=sr, hop_length=hop_length)

    # 5) Compute distance matrix and estimate penalty
    X = cqt_mid.T
    Y = cqt_aud.T
    dist_matrix = compute_cosine_distance_matrix_chunked(X, Y, chunk_size=CHUNK_SIZE)
    penalty_val = estimate_penalty(dist_matrix, sample_size=100000)

    # 6) Perform DTW with the Sakoe-Chiba band
    path, total_cost = sakoe_chiba_dtw(
        dist_matrix,
        band_radius_sec=BAND_RADIUS_SEC,
        sr=sr,
        hop_length=hop_length,
        penalty=penalty_val
    )

    sec_pf = hop_length / sr
    warp_i = np.array([p[0] for p in path], dtype=np.float32) * sec_pf
    warp_j = np.array([p[1] for p in path], dtype=np.float32) * sec_pf

    # 7) Apply the warping to the trimmed MIDI (pm_30), then shift
    starts_30 = [note.start for inst in pm_30.instruments for note in inst.notes]
    earliest_midi_start_30 = min(starts_30) if starts_30 else 0.0

    skipped_notes = 0
    for inst in pm_30.instruments:
        for note in inst.notes:
            s_new = float(np.interp(note.start, warp_i, warp_j))
            e_new = float(np.interp(note.end,   warp_i, warp_j))
            if e_new < s_new:
                s_new, e_new = e_new, s_new
            if (e_new - s_new) < sec_pf:
                skipped_notes += 1
            note.start = s_new
            note.end   = e_new

    all_starts_30aligned = [note.start for inst in pm_30.instruments for note in inst.notes]
    earliest_midi_start_new = min(all_starts_30aligned) if all_starts_30aligned else 0.0

    shift = earliest_midi_start_new - earliest_midi_start_30
    for inst in pm_30.instruments:
        for note in inst.notes:
            note.start -= shift
            note.end   -= shift

    # Apply the same shift to the full MIDI
    for inst in pm_full.instruments:
        for note in inst.notes:
            note.start -= shift
            note.end   -= shift

    # Save final aligned MIDI
    pm_full.write(out_midi_path)

    # Optional: save plots
    if do_save_plots:
        # (a) Plot waveforms of the first clip_sec seconds of the original audio and MIDI-synthesized audio
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
        axes[0].plot(audio_30, color='blue')
        axes[0].set_title("WAV (clipped)")
        axes[1].plot(midi_audio_30, color='red')
        axes[1].set_title("MIDI Audio (trimmed)")
        plt.tight_layout()

        savepath1 = os.path.join(IMG_DIR_WAVE, f"{basename}_wave.png")
        plt.savefig(savepath1)
        plt.close(fig)

        # (b) Compare around the earliest MIDI note start (¡¾ VIEW_RANGE_SEC)
        view_start = earliest_midi_start_new - VIEW_RANGE_SEC
        view_end   = earliest_midi_start_new + VIEW_RANGE_SEC
        if view_end > 0:
            view_start = max(0, view_start)
            audio_duration = len(audio_30) / sr
            view_end = min(audio_duration, view_end)

            s_start = int(view_start * sr)
            s_end   = int(view_end   * sr)

            audio_snip = audio_30[s_start:s_end]
            midi_30_aligned_audio = pm_30.fluidsynth(fs=sr)
            midi_snip = midi_30_aligned_audio[s_start:s_end]
            tbase = np.linspace(view_start, view_end, len(audio_snip))

            fig2, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
            axs[0].plot(tbase, audio_snip, color='blue', label='WAV')
            axs[0].axvline(x=earliest_midi_start_new, color='green', linestyle='--')
            axs[0].set_ylabel("Amplitude")
            axs[0].legend()

            axs[1].plot(tbase, midi_snip, color='red', label='Aligned MIDI')
            axs[1].axvline(x=earliest_midi_start_new, color='green', linestyle='--')
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Amplitude")

            fig2.suptitle("Earliest MIDI start (¡¾ 50 ms)")
            plt.tight_layout()

            savepath2 = os.path.join(IMG_DIR_SNIP, f"{basename}_snip.png")
            plt.savefig(savepath2)
            plt.close(fig2)

###############################################################################
# Main execution: align all (audio, MIDI) pairs in the directory
###############################################################################
if __name__ == "__main__":
    wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    wav_files.sort()

    for wav_path in tqdm(wav_files, desc="Aligning pairs"):
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        midi_path = os.path.join(MIDI_DIR, basename + ".mid")
        if not os.path.isfile(midi_path):
            # Skip if no matching MIDI file
            continue

        out_midi_path = os.path.join(OUT_DIR, basename + ".mid")
        align_audio_midi_pair(
            audio_path=wav_path,
            midi_path=midi_path,
            out_midi_path=out_midi_path,
            clip_sec=CLIP_SEC,
            sr=SAMPLE_RATE,
            sf2_path=SF2_PATH,
            hop_length=64,
            do_save_plots=True
        )

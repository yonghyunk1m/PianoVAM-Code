# Python Script for Evaluating Postprocessed MIDI

import os
import pretty_midi
import numpy as np
from mir_eval.transcription import precision_recall_f1_overlap
from mir_eval.transcription_velocity import precision_recall_f1_overlap as velocity_prf
import pandas as pd
from glob import glob
from tqdm import tqdm

# key_offset: Key releasing timing
# frame_offset: Pedal-extended offset (In general, Offset refers to this.)
def load_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', comment='#', header=None,
                     names=['onset', 'key_offset', 'frame_offset', 'note', 'velocity'])
    pitches = df['note'].to_numpy()
    intervals = df[['onset', 'frame_offset']].to_numpy()
    velocities = df['velocity'].to_numpy()
    return pitches, intervals, velocities


def load_midi(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = [note for inst in midi.instruments for note in inst.notes]
    notes.sort(key=lambda n: n.start)
    pitches = np.array([n.pitch for n in notes])
    intervals = np.array([[n.start, n.end] for n in notes])
    velocities = np.array([n.velocity for n in notes])
    return pitches, intervals, velocities


def evaluate_midi_vs_tsv(midi_path, tsv_path):
    p_ref, i_ref, v_ref = load_tsv(tsv_path)
    p_est, i_est, v_est = load_midi(midi_path)

    results = {}
    # print(f"p_ref: {p_ref}")
    # print(f"p_est: {p_est}")
    if len(p_est) == 0:
        print(f"[WARN] Estimated notes are empty for {midi_path}")
        results['note/precision'] = 0.0
        results['note/recall'] = 0.0
        results['note/f1'] = 0.0
        results['note-w-offsets/precision'] = 0.0
        results['note-w-offsets/recall'] = 0.0
        results['note-w-offsets/f1'] = 0.0
        results['note-w-velocity/precision'] = 0.0
        results['note-w-velocity/recall'] = 0.0
        results['note-w-velocity/f1'] = 0.0
        return results
    
    # Note metrics without offset
    p, r, f, _ = precision_recall_f1_overlap(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    results['note/precision'] = p
    results['note/recall'] = r
    results['note/f1'] = f

    print(f"precision: {p}")
    print(f"recall: {r}")
    print(f"f1: {f}")

    # Note metrics with offset
    p, r, f, _ = precision_recall_f1_overlap(i_ref, p_ref, i_est, p_est)
    results['note-w-offsets/precision'] = p
    results['note-w-offsets/recall'] = r
    results['note-w-offsets/f1'] = f

    # Note metrics with velocity (no offset)
    p, r, f, _ = velocity_prf(i_ref, p_ref, v_ref, i_est, p_est, v_est, offset_ratio=None, velocity_tolerance=0.1)
    results['note-w-velocity/precision'] = p
    results['note-w-velocity/recall'] = r
    results['note-w-velocity/f1'] = f

    return results


def find_tsv(label_root, midi_name):
    tsv_path = os.path.join(label_root, midi_name[:-3] + '.tsv')
    return tsv_path if os.path.exists(tsv_path) else None


def batch_evaluate(pred_dir, label_root):
    midi_paths = sorted(glob(os.path.join(pred_dir, '*.mid')))
    all_results = {}

    for midi_path in tqdm(midi_paths, desc='Evaluating'):
        name = os.path.basename(midi_path).replace('.pred.mid', '')
        tsv_path = find_tsv(label_root, name)
        if not tsv_path:
            print(f"[WARNING] Missing TSV for: {name}")
            continue

        result = evaluate_midi_vs_tsv(midi_path, tsv_path)
        all_results[name] = result

    return all_results


if __name__ == '__main__':
    pred_dir = 'post-processed/midi/folder'
    label_root = 'pedal-extended/tsv/path/'

    results = batch_evaluate(pred_dir, label_root)

    # Print and save results
    df = pd.DataFrame(results).T  # Transpose for easier aggregation
    print("\n=== AVERAGE METRICS ===")
    print(df.mean())

    # Save detailed metrics
    detailed_csv_path = os.path.join(pred_dir, 'detailed_metrics.csv')
    df.to_csv(detailed_csv_path)
    print(f"[INFO] Saved detailed metrics to {detailed_csv_path}")

    # Save average metrics
    avg_csv_path = os.path.join(pred_dir, 'average_metrics.csv')
    df.mean().to_frame(name='average').T.to_csv(avg_csv_path)
    print(f"[INFO] Saved average metrics to {avg_csv_path}")

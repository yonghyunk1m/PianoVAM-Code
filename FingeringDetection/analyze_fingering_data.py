import pickle
import os
import glob
from collections import defaultdict

def analyze_file_with_logic(filepath):
    """
    Analyzes a single fingering pkl file using logic from sthanddecider
    and returns statistics.
    """
    with open(filepath, 'rb') as f:
        keyhandlist = pickle.load(f)

    if not keyhandlist:
        return 0, 0, 0

    # Pre-process to find token ranges and group data by token
    token_frames = defaultdict(list)
    notes_by_token = defaultdict(list)
    max_token_idx = -1

    for frame_idx, frame_data in enumerate(keyhandlist):
        for keyhandinfo in frame_data:
            token_idx = keyhandinfo[1]
            if token_idx > max_token_idx:
                max_token_idx = token_idx
            token_frames[token_idx].append(frame_idx)
            notes_by_token[token_idx].append(keyhandinfo)
    
    total_notes = max_token_idx + 1
    if total_notes == 0:
        return 0, 0, 0

    no_candidate_notes = 0
    multi_candidate_notes = 0

    for i in range(total_notes):
        if i not in token_frames:
            # This token does not appear in the data, count as no-candidate
            no_candidate_notes += 1
            continue

        start_frame, end_frame = min(token_frames[i]), max(token_frames[i])
        total_frame = end_frame - start_frame + 1

        fingerscore = [0] * 10
        for keyhandinfo in notes_by_token[i]:
            # keyhandinfo[3] is a list of 11, scores are at indices 1-10
            for k in range(1, 11):
                fingerscore[k - 1] += keyhandinfo[3][k]
        
        pressedfingers = []
        for j in range(10):
            if fingerscore[j] > 0:
                pressedfingers.append([j + 1, fingerscore[j]])
        
        # Filter candidates based on the threshold
        candidates = []
        for finger in pressedfingers:
            if total_frame > 0 and (finger[1] / total_frame) >= 0.5:
                candidates.append(finger)
        
        num_candidates = len(candidates)
        if num_candidates == 0:
            no_candidate_notes += 1
        elif num_candidates > 1:
            multi_candidate_notes += 1

    return total_notes, no_candidate_notes, multi_candidate_notes

def main():
    output_dir = 'PianoVAM-Code/FingeringDetection/output'
    file_pattern = os.path.join(output_dir, 'fingering_*.pkl')
    fingering_files = [f for f in glob.glob(file_pattern) if 'hand_data' not in os.path.basename(f)]

    if not fingering_files:
        print("No fingering pkl files found.")
        return

    results = []
    total_notes_all = 0
    no_candidate_notes_all = 0
    multi_candidate_notes_all = 0

    print("Analyzing files...")
    for filepath in sorted(fingering_files):
        try:
            total, no_cand, multi_cand = analyze_file_with_logic(filepath)
            
            if total > 0:
                no_cand_ratio = (no_cand / total) * 100
                multi_cand_ratio = (multi_cand / total) * 100
                results.append({
                    "File": os.path.basename(filepath),
                    "Total Notes": total,
                    "No-candidate Notes": no_cand,
                    "No-candidate Ratio (%)": f"{no_cand_ratio:.2f}",
                    "Multi-candidate Notes": multi_cand,
                    "Multi-candidate Ratio (%)": f"{multi_cand_ratio:.2f}",
                })

            total_notes_all += total
            no_candidate_notes_all += no_cand
            multi_candidate_notes_all += multi_cand
        except Exception as e:
            print(f"Could not process {os.path.basename(filepath)}: {e}")

    # Print results in a formatted table
    if results:
        # Determine column widths
        headers = list(results[0].keys())
        col_widths = {h: len(h) for h in headers}
        for row in results:
            for h in headers:
                col_widths[h] = max(col_widths[h], len(str(row[h])))
        
        # Print header
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in results:
            row_line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            print(row_line)

    if total_notes_all > 0:
        print("\n" + "="*50)
        print("Overall Summary:")
        print(f"Total notes across all files: {total_notes_all}")
        
        no_cand_ratio_all = (no_candidate_notes_all / total_notes_all) * 100
        multi_cand_ratio_all = (multi_candidate_notes_all / total_notes_all) * 100
        
        print(f"Total no-candidate notes: {no_candidate_notes_all} ({no_cand_ratio_all:.2f}%)")
        print(f"Total multi-candidate notes: {multi_candidate_notes_all} ({multi_cand_ratio_all:.2f}%)")
        print("="*50)

if __name__ == "__main__":
    main()

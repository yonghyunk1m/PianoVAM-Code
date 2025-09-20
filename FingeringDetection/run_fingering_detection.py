import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
from tqdm import tqdm
import time as timemodule
import json
import sys
import torch
import time
import random
import argparse

# GpuOptions is not available in all versions and is no longer needed with our new strategy.
# from mediapipe.tasks.python.vision import GpuOptions
import filelock
from floatinghands_torch_pure import (
    handclass,
    modelskeleton,
    depthlist,
    detectfloatingframes,
    generatekeyboard,
    handpositiondetector,
)
from midicomparison import handfingercorresponder, miditotoken, tokentoframeinfo
import glob
import multiprocessing

# Ensure the script's directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# --- Mappings ---
# Define video groups based on filenames
video_group_map = {
    "2024-02-14_19-10-09": 1,
    "2024-02-14_19-44-26": 1,
    "2024-02-14_19-55-17": 1,
    "2024-02-14_20-10-08": 1,
    "2024-02-14_20-37-55": 1,
    "2024-02-14_20-58-43": 1,
    "2024-02-14_21-30-04": 1,
    "2024-02-15_15-13-10": 1,
    "2024-02-15_20-07-54": 1,
    "2024-02-15_20-17-26": 1,
    "2024-02-15_20-28-13": 1,
    "2024-02-15_20-38-23": 1,
    "2024-02-15_20-47-59": 1,
    "2024-02-15_20-54-38": 1,
    "2024-02-15_21-01-11": 1,
    "2024-02-15_21-18-14": 1,
    "2024-02-15_21-33-36": 1,
    "2024-02-15_21-40-43": 1,
    "2024-02-15_21-57-38": 1,
    "2024-02-15_22-02-16": 1,
    "2024-02-15_22-12-41": 1,
    "2024-02-15_22-18-32": 1,
    "2024-02-15_22-57-11": 1,
    "2024-02-15_23-14-13": 1,
    "2024-02-16_14-41-07": 1,
    "2024-02-16_23-19-03": 1,
    "2024-02-17_00-12-50": 1,
    "2024-02-17_00-29-10": 1,
    "2024-02-17_21-37-57": 1,
    "2024-02-17_21-44-37": 1,
    "2024-02-17_22-12-41": 1,
    "2024-02-17_22-33-45": 1,
    "2024-02-20_18-08-35": 1,
    "2024-02-20_18-23-30": 1,
    "2024-02-22_11-58-09": 1,
    "2024-02-22_14-11-52": 1,
    "2024-03-04_03-46-36": 2,
    "2024-03-04_03-50-34": 2,
    "2024-03-04_04-14-36": 2,
    "2024-03-04_04-32-11": 2,
    "2024-03-04_04-36-26": 2,
    "2024-03-04_22-38-33": 3,
    "2024-03-11_22-23-29": 3,
    "2024-03-12_15-43-01": 3,
    "2024-03-13_11-32-42": 3,
    "2024-03-13_11-54-57": 3,
    "2024-03-15_22-54-26": 4,
    "2024-03-17_18-54-44": 5,
    "2024-03-18_04-15-06": 5,
    "2024-03-26_16-33-49": 5,
    "2024-03-27_11-49-04": 4,
    "2024-04-08_22-49-18": 4,
    "2024-04-08_23-10-23": 4,
    "2024-04-08_23-26-19": 4,
    "2024-04-08_23-42-45": 4,
    "2024-04-08_23-50-57": 4,
    "2024-04-09_03-42-36": 4,
    "2024-04-09_04-09-27": 4,
    "2024-09-02_12-45-12": 6,
    "2024-09-02_13-19-32": 6,
    "2024-09-02_14-10-41": 6,
    "2024-09-02_15-05-42": 6,
    "2024-09-02_15-15-09": 6,
    "2024-09-02_15-26-39": 6,
    "2024-09-02_18-25-31": 6,
    "2024-09-02_18-29-40": 6,
    "2024-09-02_18-42-47": 6,
    "2024-09-02_19-40-55": 6,
    "2024-09-02_20-00-16": 6,
    "2024-09-02_20-50-18": 6,
    "2024-09-02_21-04-45": 6,
    "2024-09-02_23-22-31": 6,
    "2024-09-02_23-49-01": 6,
    "2024-09-03_00-07-46": 6,
    "2024-09-03_00-44-45": 6,
    "2024-09-04_16-13-44": 6,
    "2024-09-04_17-02-04": 6,
    "2024-09-04_17-07-59": 6,
    "2024-09-04_17-12-33": 6,
    "2024-09-04_18-11-36": 6,
    "2024-09-04_19-09-38": 6,
    "2024-09-04_19-52-57": 6,
    "2024-09-04_20-09-07": 6,
    "2024-09-04_20-20-07": 6,
    "2024-09-04_20-30-34": 6,
    "2024-09-04_20-59-20": 6,
    "2024-09-04_21-09-37": 6,
    "2024-09-04_21-44-42": 6,
    "2024-09-04_21-51-59": 6,
    "2024-09-04_22-00-35": 6,
    "2024-09-04_22-06-40": 6,
    "2024-09-04_22-13-22": 6,
    "2024-09-04_22-19-28": 6,
    "2024-09-05_13-25-10": 6,
    "2024-09-05_13-36-00": 6,
    "2024-09-05_13-50-00": 6,
    "2024-09-05_20-46-25": 6,
    "2024-09-05_20-51-00": 6,
    "2024-09-05_21-01-27": 6,
    "2024-09-05_21-04-38": 6,
    "2024-09-05_21-07-35": 6,
    "2024-09-05_21-13-49": 6,
    "2024-09-05_21-18-58": 6,
    "2024-09-05_21-26-38": 6,
    "2024-09-05_21-31-00": 6,
    "2024-09-05_21-37-08": 6,
    "2024-09-05_22-11-07": 6,
}

# Define thresholds for each group
threshold_map = {
    "1": 0.8711,
    "2": 0.6642,
    "3": 0.9102,
    "4": 0.9511,
    "5": 0.9912,
    "6": 0.9386,
}


def get_video_group(video_key):
    """Finds which group a video belongs to using the new dictionary mapping."""
    # The new map directly maps video_key to a group number.
    return video_group_map.get(video_key)


DEFAULT_THRESHOLD = 0.9  # Original hardcoded value

# --- File Paths & Locking ---
progress_file = os.path.join(script_dir, "fingering_detection_progress.json")
lock_file = progress_file + ".lock"


def acquire_lock(lock_path, pid):
    """Acquires a lock using an atomic file creation, with a timeout."""
    start_time = time.time()
    while True:
        try:
            # Atomically create the lock file. If it exists, this will raise FileExistsError.
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start_time > 60:  # 1-minute timeout
                print(
                    f" G [PID: {pid}] WARNING: Lock wait timeout on {lock_path}. Attempting to remove stale lock."
                )
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                start_time = time.time()  # Reset timer
            time.sleep(
                random.uniform(0.1, 0.5)
            )  # Add random sleep to reduce contention


def release_lock(lock_path):
    """Releases the lock by deleting the lock file."""
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except OSError:
            pass  # Ignore if another process already removed it


def update_progress(video_name, status, pid, error_msg=None):
    """Atomically updates the progress JSON file."""
    acquire_lock(lock_file, pid)
    try:
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
        else:
            progress = {}

        update = {"status": status, "pid": pid}
        if error_msg:
            update["error"] = error_msg

        progress[video_name] = update

        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=4)
    finally:
        release_lock(lock_file)


def load_progress():
    """Loads the progress JSON file."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(
                f"⚠️ Error decoding JSON from {progress_file}. Starting with empty progress."
            )
            return {}
    else:
        print(
            f"⚠️ Progress file not found at {progress_file}. Starting with empty progress."
        )
        return {}


# --- Configuration ---
# These are the confidence thresholds from main_loop.py
min_hand_detection_confidence = 0.85
min_hand_presence_confidence = 0.8
min_tracking_confidence = 0.5
VisionRunningMode = mp.tasks.vision.RunningMode

# --- File Paths ---
filepath = os.path.join(script_dir, "videocapture")  # Video capture directory
midipath = os.path.join(script_dir, "midiconvert")  # MIDI file directory


# --- Core Processing Functions ---
def create_handlandmarker(
    running_mode=VisionRunningMode.VIDEO,
    min_hand_detection_confidence=0.85,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.5,
):
    """Initializes and creates the MediaPipe HandLandmarker object."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "hand_landmarker.task")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hand landmarker model not found at {model_path}")

    # --- Revert to forcing MediaPipe to always use the CPU ---
    print(f"  [MediaPipe] Configuring to run on CPU (forced for performance).")
    delegate = python.BaseOptions.Delegate.CPU

    base_options = python.BaseOptions(model_asset_path=model_path, delegate=delegate)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        # --- FIX: Use the running_mode parameter ---
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    detector = vision.HandLandmarker.create_from_options(options)
    print(
        f"🖐️ MediaPipe Hand Landmarker created (Mode: {running_mode}). Using CPU for inference."
    )
    return detector


def process_video_to_hand_data(video_path, landmarker, output_dir, worker_position=0):
    """
    STAGE 1: Process a single video to extract and save hand landmark data.
    This is a CPU-intensive task using MediaPipe.
    """
    with landmarker:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        hand_data = []

        # --- MODIFIED: Added position and leave=False for clean parallel output ---
        video_name = os.path.basename(video_path)
        desc = f"  (CPU) Vid {worker_position}: {video_name[:20]:<20}" # Padded for alignment
        for frame_idx in tqdm(
            range(num_frames), desc=desc, position=worker_position, leave=False
        ):
            ret, frame = cap.read()
            if not ret:
                break

            # --- FIX: Convert BGR (OpenCV default) to RGB (MediaPipe expected) ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # --- Use detect_for_video() for VIDEO mode ---
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # --- Final: Revert to saving the original MediaPipe object ---
            hand_data.append(detection_result)

        cap.release()

    video_key = os.path.splitext(os.path.basename(video_path))[0]
    # Ensure output_dir is an absolute path to prevent ambiguity
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)  # Create dir if it doesn't exist
    output_path = os.path.join(abs_output_dir, f"{video_key}_hand_data.pkl")

    # Save both hand_data and fps in a dictionary
    data_to_save = {"hand_data": hand_data, "fps": fps}

    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Hand landmark data and FPS saved to {output_path}")
    return output_path  # Add this return statement


def load_keyboard_for_video(video_key, pixel_points_dir, pkl_path):
    """Loads a keyboard layout for a specific video, trying JSON first, then a PKL fallback."""
    # Priority 1: Load from JSON in pixel_points dir
    json_path = os.path.join(pixel_points_dir, f"{video_key}_pixel_points.json")
    if os.path.exists(json_path):
        print(f"🎹 Found pixel points JSON: {json_path}. Loading keyboard layout.")
        try:
            # --- FIX: Use generatekeyboard from floatinghands_torch_pure directly ---
            # It already handles loading from a JSON file path.
            keyboard = generatekeyboard(json_path)
            ratio = (
                1.0  # The ratio is handled internally or irrelevant with the new method
            )

            if keyboard:
                print(
                    "✅ Keyboard layout loaded successfully from JSON via floatinghands_torch_pure."
                )
                # We need keystone points for visualization, let's load them manually for the return
                with open(json_path, "r") as f:
                    keystone_data = json.load(f).get("pixel_points")
                return keyboard, ratio, keystone_data
        except Exception as e:
            print(f"⚠️ Failed to load keyboard from JSON {json_path}: {e}")
            import traceback

            traceback.print_exc()

    # Priority 2: Fallback to the big PKL file
    if os.path.exists(pkl_path):
        print(f"🎹 Trying fallback to PKL file: {pkl_path}")
        try:
            with open(pkl_path, "rb") as f:
                keyboardcoordinateinfo = pickle.load(f)
            if video_key in keyboardcoordinateinfo:
                print("✅ Keyboard layout loaded successfully from PKL.")
                keyboard_data = keyboardcoordinateinfo[video_key]
                # Pass the raw data to generatekeyboard
                keyboard, _ = generatekeyboard(keyboard_data)
                return keyboard, 1.0, keyboard_data
        except Exception as e:
            print(f"⚠️ Failed to load keyboard from PKL {pkl_path}: {e}")

    print(f"❌ Error: Could not find or load keyboard layout for video '{video_key}'.")
    return None, None, None


def decide_fingering(prefingercorrespond, tokenlist):
    """
    Analyzes the fingering correspondence and decides the most likely finger for each note.
    This is a non-UI adaptation of the sthanddecider logic from ASDF.py.
    """
    final_fingering = [None] * len(tokenlist)
    undecided_notes = []

    for i in range(len(tokenlist)):
        total_frames_for_token = tokenlist[i][2] - tokenlist[i][0]

        finger_scores = [0] * 10  # Scores for fingers 1-10

        # Aggregate scores from all frames where this token was active
        for frame_keyhandinfo in prefingercorrespond:
            for key_info in frame_keyhandinfo:
                # key_info: [key_pitch, token_number, hand, fingercount_array]
                if key_info[1] == i:  # If this is the token we are evaluating
                    fingercount_array = key_info[3]
                    for finger_idx in range(10):
                        finger_scores[finger_idx] += fingercount_array[finger_idx]

        pressed_fingers = []
        high_candidates = []

        for finger_idx, score in enumerate(finger_scores):
            if score > 0:
                # Candidate if its score is more than half of total frames
                if score / total_frames_for_token >= 0.5:
                    pressed_fingers.append([finger_idx + 1, score])
                # Strong candidate if score is more than 80%
                if score / total_frames_for_token > 0.80:
                    high_candidates.append([finger_idx + 1, score])

        if len(pressed_fingers) == 1:
            final_fingering[i] = pressed_fingers[0][0]
        elif len(pressed_fingers) > 1:
            if len(high_candidates) == 1:
                final_fingering[i] = high_candidates[0][0]
            else:
                final_fingering[i] = "Undecided"
                undecided_notes.append(
                    {
                        "token_index": i,
                        "note": pitch_list[tokenlist[i][1]],
                        "candidates": pressed_fingers,
                    }
                )
        else:
            final_fingering[i] = "NoInfo"

    return final_fingering


def extract_fingering_from_data(
    hand_data_path,
    midi_path,
    modelskeleton_func,
    device,
    keyboard_layout,
    confidence_settings,
    ratio,
    worker_position=0,
):
    """
    STAGE 2: Process the extracted hand data to determine fingering.
    This is a GPU-intensive task using PyTorch.
    """
    video_key = os.path.basename(hand_data_path).replace("_hand_data.pkl", "")
    print(f"🚀 [Vid {worker_position}] Starting GPU processing for {video_key}...")

    # Load the dictionary containing hand_data and fps
    with open(hand_data_path, "rb") as f:
        loaded_data = pickle.load(f)

    loaded_hand_data = loaded_data["hand_data"]
    fps = loaded_data["fps"]

    handlist = []
    # --- MODIFIED: Added position and leave=False for clean parallel output ---
    desc1 = f"  (GPU) Vid {worker_position}: {video_key[:20]:<20} (1/3)"
    for i, detection_result in enumerate(
        tqdm(loaded_hand_data, desc=desc1, position=worker_position, leave=False)
    ):
        if detection_result and detection_result.hand_landmarks:
            frame_hands = []
            # Loop through each detected hand in the frame using an index
            for hand_index in range(len(detection_result.hand_landmarks)):
                # --- NEW: Convert coordinates from [0, 1] to [-1, 1] ---
                for finger_landmark in detection_result.hand_landmarks[hand_index]:
                    finger_landmark.x=finger_landmark.x*2-1 #[0,1]->[-1,1]
                    finger_landmark.y=finger_landmark.y*2-1 #[0,1]->[-1,1]

                handedness = detection_result.handedness[hand_index]

                # Extract the hand type string ('Left' or 'Right') from the handedness object
                hand_type_str = handedness[0].category_name

                # Create a handclass object with positional arguments in the correct order
                # __init__(self, handtype, handlandmark, handframe)
                hand_obj = handclass(
                    hand_type_str,
                    detection_result.hand_landmarks[hand_index],  # Use the converted landmarks
                    i,  # frame_index
                )
                frame_hands.append(hand_obj)
            handlist.append(frame_hands)
        else:
            handlist.append([])

    print(f"  [Vid {worker_position}] Creating hand skeleton model with PyTorch...")
    lhmodel, rhmodel = modelskeleton_func(handlist, device)

    print(f"  [Vid {worker_position}] Calculating depth for each hand...")
    # This `depthlist` function modifies the hand objects in `handlist` in-place and returns it.
    handlist_with_depth = depthlist(handlist, lhmodel, rhmodel, ratio, device)

    # Determine the threshold based on the video group
    video_group = get_video_group(video_key)
    threshold = threshold_map.get(video_group, 0.9)
    print(
        f"📹 Video '{video_key}' is in Group {video_group}. Using floating hand threshold: {threshold}"
    )

    print("? Detecting floating hand frames...")
    # Prepare arguments for detectfloatingframes
    frame_count = len(handlist_with_depth)
    faultyframes = (
        []
    )  # Assuming no faulty frames for now as the function isn't implemented

    floatingframelist = detectfloatingframes(
        handlist=handlist_with_depth,
        frame_count=frame_count,
        faultyframes=faultyframes,
        lhmodel=lhmodel,
        rhmodel=rhmodel,
        ratio=ratio,
        threshold=threshold,
    )

    print("? Tokenizing MIDI file...")
    midi_filename_no_ext = os.path.splitext(os.path.basename(midi_path))[0]
    miditoken = miditotoken(midi_filename_no_ext, fps, "simplified")

    # --- Start: Re-structured data flow ---
    print("? Preparing data for MIDI correlation...")

    # 1. Create framemidilist using tokentoframeinfo
    framemidilist = tokentoframeinfo(miditoken, frame_count)

    # 2. Loop through frames to create framehandfingerlist (NOW IN PARALLEL)
    # --- MODIFIED: Changed description text from (GPU) to (CPU) ---
    desc2 = f"  (CPU) Vid {worker_position}: {video_key[:20]:<20} (2/3)"

    tasks = []
    debug_counter = 0
    for i in range(frame_count):
        # Pass the debug counter only for the first few potential debug frames
        counter_to_pass = debug_counter if debug_counter < 5 else None
        tasks.append(
            (
                i,
                handlist_with_depth[i],
                floatingframelist,
                keyboard_layout,
                framemidilist[i],
                counter_to_pass,
            )
        )

    framehandfingerlist = []
    # --- FIX: Removed nested multiprocessing pool to prevent 'daemonic processes' error ---
    # The outer loop already parallelizes by video, which is more efficient.
    for task_args in tqdm(tasks, desc=desc2, position=worker_position, leave=False):
        result, returned_counter = hand_position_worker(task_args)
        framehandfingerlist.append(result)
        if returned_counter is not None and returned_counter > debug_counter:
            debug_counter = returned_counter

    # --- MODIFIED: Changed print statement ---
    print(f"  [Vid {worker_position}] Correlating hands to MIDI notes... (3/3)")
    # 3. Call handfingercorresponder with the correct arguments in the correct order
    final_fingering = handfingercorresponder(
        framemidilist, framehandfingerlist, keyboard_layout, miditoken
    )
    # --- End: Re-structured data flow ---

    return final_fingering


def hand_position_worker(args_tuple):
    """
    Worker function for parallel processing of handpositiondetector.
    Includes debugging prints for the first few active frames.
    """
    (
        i,
        handsinfo,
        floatingframelist,
        keyboard_layout,
        framemidilist_frame,
        debug_counter,
    ) = args_tuple

    # Original code expects handpositiondetector to return three separate lists
    framehands, framefingers, frametips = handpositiondetector(
        handsinfo, floatingframelist, keyboard_layout
    )

    # The handfingercorresponder expects a list containing these three lists
    return [framehands, framefingers, frametips], debug_counter


# --- NEW: Wrapper function for multiprocessing ---
def run_full_pipeline_worker(args):
    """
    Wrapper for run_full_pipeline to be used with multiprocessing.Pool.
    It unpacks the tuple of arguments and handles exceptions within the process.
    """
    try:
        return run_full_pipeline(*args)
    except Exception as e:
        # Extract video name from args for better error logging
        videoname = "Unknown"
        if args and len(args) > 0:
            videoname = os.path.basename(args[0])
        print(f"---")
        print(f"❌❌❌ A critical error occurred in the worker for video: {videoname} ❌❌❌")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"--- End of error for {videoname} ---")
        return None  # Indicate failure


# --- Main Pipeline Function ---
def run_full_pipeline(
    videoname,
    output_dir,
    midi_dir,
    pixel_points_dir,
    keyboard_pkl_path,
    video_group_map,
    threshold_map,
    confidence_settings,
    worker_position=0,
):
    """
    Runs the full CPU and GPU pipeline for a single video.
    """
    video_key = os.path.basename(videoname).replace(".mp4", "")

    progress_data = load_progress()
    video_progress = progress_data.get(video_key, {})
    status = video_progress.get("status")

    if status == "completed":
        print(f"☑️ [Vid {worker_position}] Video {video_key} already processed. Skipping.")
        return

    # --- Load Keyboard Layout ---
    keyboard, ratio, keystone_points = load_keyboard_for_video(
        video_key, pixel_points_dir, keyboard_pkl_path
    )
    if not keyboard:
        update_progress(video_key, "error", 0, "Failed to load keyboard layout.")
        return

    try:
        handlist_path = os.path.join(output_dir, f"{video_key}_hand_data.pkl")

        # --- MODIFIED: Skip Part 1 based on file existence, not status ---
        if os.path.exists(handlist_path):
            print(f"--- [Vid {worker_position}] Found existing hand data for {video_key}. Skipping CPU Pre-processing. ---")
        else:
            # --- Part 1: Hand Landmark Extraction (CPU) ---
            update_progress(video_key, "in_progress", 0, "Starting processing") # Set status before starting
            print(f"--- [Vid {worker_position}] Running CPU Pre-processing for {video_key} ---")
            landmarker = create_handlandmarker(
                running_mode=VisionRunningMode.VIDEO,
                min_hand_detection_confidence=confidence_settings[
                    "min_hand_detection_confidence"
                ],
                min_hand_presence_confidence=confidence_settings[
                    "min_hand_presence_confidence"
                ],
                min_tracking_confidence=confidence_settings["min_tracking_confidence"],
            )
            generated_path = process_video_to_hand_data(videoname, landmarker, output_dir, worker_position)
            if not generated_path:
                raise ValueError("Part 1 (Hand Landmark Extraction) failed.")
            handlist_path = generated_path # Ensure we use the generated path
            update_progress(video_key, "cpu_step_completed", 0, "Hand landmarks extracted")

        # --- Part 2: Fingering Extraction (GPU Multi-Core + CPU) ---
        print(f"--- [Vid {worker_position}] Running GPU Fingering Extraction for {video_key} ---")

        # --- MODIFIED: Dynamically assign GPU to each worker for true parallel processing ---
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                # Assign GPU in a round-robin fashion based on the worker's position
                device_id = (worker_position - 1) % num_gpus
                device = torch.device(f"cuda:{device_id}")
                print(f"--- [Vid {worker_position}] Assigned to GPU {device_id} ---")
            else:
                # Fallback in an unlikely case where cuda is available but count is 0
                device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        group = video_group_map.get(video_key, 6)
        threshold = threshold_map.get(group, 0.9387)
        
        final_fingering = extract_fingering_from_data(
            handlist_path,
            os.path.join(midi_dir, os.path.basename(videoname).replace(".mp4", ".mid")),
            modelskeleton,  # Pass the modelskeleton function itself
            device,  # Pass the dynamically assigned device
            keyboard,
            confidence_settings,
            ratio,  # Pass the ratio here
            worker_position, # Pass the worker position down
        )
        if final_fingering is None:
            raise ValueError("Part 2 (Fingering Extraction) failed.")

        # Save final result
        result_path = os.path.join(output_dir, f"fingering_{video_key}.pkl")
        with open(result_path, "wb") as f:
            pickle.dump(final_fingering, f)

        update_progress(video_key, "completed", 0, f"Success. Result at {result_path}")
        print(f"✅✅✅ [Vid {worker_position}] Successfully processed {video_key} ✅✅✅")

    except Exception as e:
        import traceback

        print(f"❌❌❌ [Vid {worker_position}] An error occurred while processing {video_key}: {e}")
        traceback.print_exc()
        update_progress(video_key, "error", 0, str(e))


def create_pytorch_model():
    model = modelskeleton()
    model.load_state_dict(
        torch.load(os.path.join(script_dir, "model_parameters_with_depth.pth"))
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Fingering detection pipeline.")

    # The script_dir is PianoVAM-Code/FingeringDetection
    default_video_dir = os.path.join(script_dir, "videocapture")
    default_midi_dir = os.path.join(script_dir, "midicapture")
    default_pixel_points_dir = os.path.join(script_dir, "pixel_points")
    default_keyboard_pkl = os.path.join(script_dir, "keyboardcoordinateinfo.pkl")
    # Set default output dir relative to the script location
    default_output_dir = os.path.join(script_dir, "output")

    parser.add_argument(
        "--video-dir",
        type=str,
        default=default_video_dir,
        help=f"Directory containing video files (default: {default_video_dir}).",
    )
    parser.add_argument(
        "--midi-dir",
        type=str,
        default=default_midi_dir,
        help=f"Directory containing MIDI files (default: {default_midi_dir}).",
    )
    parser.add_argument(
        "--pixel-points-dir",
        type=str,
        default=default_pixel_points_dir,
        help=f"Directory with keyboard layout files (default: {default_pixel_points_dir}).",
    )
    parser.add_argument(
        "--keyboard-pkl-path",
        type=str,
        default=default_keyboard_pkl,
        help=f"Path to fallback keyboard PKL file (default: {default_keyboard_pkl}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default=None,
        help="Path to a text file with a list of video files to process.",
    )
    parser.add_argument(
        "--single-video",
        type=str,
        default=None,
        help="Filename of a single video in the video-dir to process for testing.",
    )
    # Confidence arguments
    parser.add_argument("--min_hand_detection_confidence", type=float, default=0.85)
    parser.add_argument("--min_hand_presence_confidence", type=float, default=0.8)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    # --- MODIFIED: Removed the unnecessary workers-per-video argument ---
    parser.add_argument(
        "--num-parallel-videos",
        type=int,
        default=1,
        help="Number of videos to process in parallel. WARNING: If using a GPU, high values can cause memory errors.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Video and threshold mapping
    video_group_map = {
        "2024-02-14_19-10-09": 1,
        "2024-02-14_19-44-26": 1,
        "2024-02-14_19-55-17": 1,
        "2024-02-14_20-10-08": 1,
        "2024-02-14_20-37-55": 1,
        "2024-02-14_20-58-43": 1,
        "2024-02-14_21-30-04": 1,
        "2024-02-15_15-13-10": 1,
        "2024-02-15_20-07-54": 1,
        "2024-02-15_20-17-26": 1,
        "2024-02-15_20-28-13": 1,
        "2024-02-15_20-38-23": 1,
        "2024-02-15_20-47-59": 1,
        "2024-02-15_20-54-38": 1,
        "2024-02-15_21-01-11": 1,
        "2024-02-15_21-18-14": 1,
        "2024-02-15_21-33-36": 1,
        "2024-02-15_21-40-43": 1,
        "2024-02-15_21-57-38": 1,
        "2024-02-15_22-02-16": 1,
        "2024-02-15_22-12-41": 1,
        "2024-02-15_22-18-32": 1,
        "2024-02-15_22-57-11": 1,
        "2024-02-15_23-14-13": 1,
        "2024-02-16_14-41-07": 1,
        "2024-02-16_23-19-03": 1,
        "2024-02-17_00-12-50": 1,
        "2024-02-17_00-29-10": 1,
        "2024-02-17_21-37-57": 1,
        "2024-02-17_21-44-37": 1,
        "2024-02-17_22-12-41": 1,
        "2024-02-17_22-33-45": 1,
        "2024-02-20_18-08-35": 1,
        "2024-02-20_18-23-30": 1,
        "2024-02-22_11-58-09": 1,
        "2024-02-22_14-11-52": 1,
        "2024-03-04_03-46-36": 2,
        "2024-03-04_03-50-34": 2,
        "2024-03-04_04-14-36": 2,
        "2024-03-04_04-32-11": 2,
        "2024-03-04_04-36-26": 2,
        "2024-03-04_22-38-33": 3,
        "2024-03-11_22-23-29": 3,
        "2024-03-12_15-43-01": 3,
        "2024-03-13_11-32-42": 3,
        "2024-03-13_11-54-57": 3,
        "2024-03-15_22-54-26": 4,
        "2024-03-17_18-54-44": 5,
        "2024-03-18_04-15-06": 5,
        "2024-03-26_16-33-49": 5,
        "2024-03-27_11-49-04": 4,
        "2024-04-08_22-49-18": 4,
        "2024-04-08_23-10-23": 4,
        "2024-04-08_23-26-19": 4,
        "2024-04-08_23-42-45": 4,
        "2024-04-08_23-50-57": 4,
        "2024-04-09_03-42-36": 4,
        "2024-04-09_04-09-27": 4,
        "2024-09-02_12-45-12": 6,
        "2024-09-02_13-19-32": 6,
        "2024-09-02_14-10-41": 6,
        "2024-09-02_15-05-42": 6,
        "2024-09-02_15-15-09": 6,
        "2024-09-02_15-26-39": 6,
        "2024-09-02_18-25-31": 6,
        "2024-09-02_18-29-40": 6,
        "2024-09-02_18-42-47": 6,
        "2024-09-02_19-40-55": 6,
        "2024-09-02_20-00-16": 6,
        "2024-09-02_20-50-18": 6,
        "2024-09-02_21-04-45": 6,
        "2024-09-02_23-22-31": 6,
        "2024-09-02_23-49-01": 6,
        "2024-09-03_00-07-46": 6,
        "2024-09-03_00-44-45": 6,
        "2024-09-04_16-13-44": 6,
        "2024-09-04_17-02-04": 6,
        "2024-09-04_17-07-59": 6,
        "2024-09-04_17-12-33": 6,
        "2024-09-04_18-11-36": 6,
        "2024-09-04_19-09-38": 6,
        "2024-09-04_19-52-57": 6,
        "2024-09-04_20-09-07": 6,
        "2024-09-04_20-20-07": 6,
        "2024-09-04_20-30-34": 6,
        "2024-09-04_20-59-20": 6,
        "2024-09-04_21-09-37": 6,
        "2024-09-04_21-44-42": 6,
        "2024-09-04_21-51-59": 6,
        "2024-09-04_22-00-35": 6,
        "2024-09-04_22-06-40": 6,
        "2024-09-04_22-13-22": 6,
        "2024-09-04_22-19-28": 6,
        "2024-09-05_13-25-10": 6,
        "2024-09-05_13-36-00": 6,
        "2024-09-05_13-50-00": 6,
        "2024-09-05_20-46-25": 6,
        "2024-09-05_20-51-00": 6,
        "2024-09-05_21-01-27": 6,
        "2024-09-05_21-04-38": 6,
        "2024-09-05_21-07-35": 6,
        "2024-09-05_21-13-49": 6,
        "2024-09-05_21-18-58": 6,
        "2024-09-05_21-26-38": 6,
        "2024-09-05_21-31-00": 6,
        "2024-09-05_21-37-08": 6,
        "2024-09-05_22-11-07": 6,
    }
    threshold_map = {1: 0.8711, 2: 0.6642, 3: 0.9102, 4: 0.9511, 5: 0.9912, 6: 0.9387}

    confidence_settings = {
        "min_hand_detection_confidence": args.min_hand_detection_confidence,
        "min_hand_presence_confidence": args.min_hand_presence_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
    }

    all_video_files = []
    if args.single_video:
        # Search for the single video file recursively in the video directory
        all_video_files = glob.glob(
            os.path.join(args.video_dir, "**", args.single_video), recursive=True
        )
        if all_video_files:
            # Take the first match
            video_path = all_video_files[0]
            all_video_files = [video_path]  # Ensure only one file is processed
            print(f"🔬 Running in single video test mode for: {video_path}")
        else:
            print(
                f"❌ Single video file '{args.single_video}' not found in {args.video_dir} or its subdirectories. Exiting."
            )
            return
    elif args.file_list:
        try:
            with open(args.file_list, "r") as f:
                all_video_files = [
                    line.strip() for line in f if line.strip().endswith(".mp4")
                ]
            print(f"📄 Loaded {len(all_video_files)} videos from {args.file_list}")
        except FileNotFoundError:
            print(f"❌ File list not found at {args.file_list}. Exiting.")
            return
    else:
        # Find all .mp4 files recursively
        all_video_files = glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True)
        print(f"📂 Found {len(all_video_files)} videos in {args.video_dir} (recursive)")

    if not all_video_files:
        print("No video files found to process.")
        return

    all_video_files.sort()

    # --- MODIFIED: Replaced sequential loop with parallel processing logic ---

    # Create a list of argument tuples for each video processing task
    tasks = []
    for i, videoname in enumerate(all_video_files):
        # Position cycles from 1 to num_parallel_videos
        worker_pos = (i % args.num_parallel_videos) + 1
        task_args = (
            videoname,
            args.output_dir,
            args.midi_dir,
            args.pixel_points_dir,
            args.keyboard_pkl_path,
            video_group_map,
            threshold_map,
            confidence_settings,
            worker_pos, # Pass the position to the worker
        )
        tasks.append(task_args)

    # Process videos, either sequentially or in parallel based on user input
    if args.num_parallel_videos <= 1:
        print("⚙️ Running in sequential mode (1 parallel video). To run faster, increase --num-parallel-videos.")
        # --- MODIFIED: Add position=0 to main tqdm ---
        for task_args in tqdm(tasks, desc="Total Video Progress", position=0):
            run_full_pipeline_worker(task_args)
    else:
        print(f"⚙️ Running in parallel with {args.num_parallel_videos} video workers.")
        
        # Use a Pool to manage worker processes
        with multiprocessing.Pool(processes=args.num_parallel_videos) as pool:
            # Use imap_unordered to process tasks in parallel and show progress as tasks complete
            # --- MODIFIED: Add position=0 to main tqdm ---
            for _ in tqdm(pool.imap_unordered(run_full_pipeline_worker, tasks), total=len(tasks), desc="Total Video Progress", position=0):
                pass


if __name__ == "__main__":
    # --- FIX: Set the multiprocessing start method to 'spawn' for CUDA compatibility ---
    # This must be done inside the __main__ block before any other multiprocessing code.
    if "spawn" != multiprocessing.get_start_method(allow_none=True):
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            # It might have been already set by another library.
            pass
    main()

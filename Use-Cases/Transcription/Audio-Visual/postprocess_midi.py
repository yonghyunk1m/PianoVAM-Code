# AudioVisual Method described in 5.2 Audio-Visual Piano Transcription

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import mido

VISION_THRESHOLD = 2
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

fingertips_indices = [4, 8, 12, 16, 20, 25, 29, 33, 37, 41]

white_to_midi = {0:21, 1:23, 2:24, 3:26, 4:28, 5:29, 6:31,
                 7:33, 8:35, 9:36, 10:38, 11:40, 12:41, 13:43,
                 14:45, 15:47, 16:48, 17:50, 18:52, 19:53, 20:55,
                 21:57, 22:59, 23:60, 24:62, 25:64, 26:65, 27:67,
                 28:69, 29:71, 30:72, 31:74, 32:76, 33:77, 34:79,
                 35:81, 36:83, 37:84, 38:86, 39:88, 40:89, 41:91,
                 42:93, 43:95, 44:96, 45:98, 46:100, 47:101, 48:103,
                 49:105, 50:107, 51:108}

def crop_keyboard(img, points, hand_landmarks):
    Point_LT, Point_RT, Point_RB, Point_LB = points
    rectangle = np.array([Point_LT, Point_RT, Point_RB, Point_LB], np.int32).reshape((-1, 1, 2))

    WIDTH = 1024
    HEIGHT = int(WIDTH / 8.147)
    dst_pts = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    M = cv2.getPerspectiveTransform(np.float32(points), dst_pts)

    hand_landmarks_img_coords = None
    if hand_landmarks is not None:
        hand_landmarks_img_coords = cv2.perspectiveTransform(
            hand_landmarks.reshape(-1, 1, 2) * np.array([img.shape[1], img.shape[0]]), M
        ).reshape(-1, 2)

    return cv2.warpPerspective(img, M, (WIDTH, HEIGHT)), hand_landmarks_img_coords

def detect_final_key_candidates(hand_landmarks):
    final_key_candidates = set()
    if hand_landmarks is not None:
        for idx, landmark in enumerate(hand_landmarks):
            x, y = int(landmark[0]), int(landmark[1])
            if idx in fingertips_indices:
                key_index = max(0, min(51, round(52 * x / 1024)))
                start = max(0, key_index - VISION_THRESHOLD)
                end = min(51, key_index + VISION_THRESHOLD)
                key_candidates = list(range(white_to_midi[start], white_to_midi[end] + 1))
                final_key_candidates.update(key_candidates)
    return final_key_candidates

def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results.multi_hand_landmarks if results.multi_hand_landmarks else None

def find_key_candidates(video_path, target_frame_time):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, target_frame_time * 1000)
    success, frame = cap.read()
    cap.release()
    if not success:
        print("Error: Could not read frame from video.")
        return

    root_dir = 'pianovam/path'
    record_time = os.path.basename(video_path)[:-4]
    metadata = json.load(open(os.path.join(root_dir, 'metadata.json')))
    points = [
        [metadata[key]['Point_LT'], metadata[key]['Point_RT'], metadata[key]['Point_RB'], metadata[key]['Point_LB']]
        for key in metadata if metadata[key]['record_time'] == record_time
    ][0]
    points = [list(map(int, point.split(', '))) for point in points]

    hand_landmarks = get_hand_landmarks(frame)
    landmarks_array = None
    if hand_landmarks and len(hand_landmarks) == 2:
        first = [[lm.x, lm.y] for lm in hand_landmarks[0].landmark]
        second = [[lm.x, lm.y] for lm in hand_landmarks[1].landmark]
        landmarks_array = np.array(first + second)

    frame, hand_landmarks = crop_keyboard(frame, points, landmarks_array)
    return detect_final_key_candidates(hand_landmarks)

# Test splits of PianoVAM
dates = [
    "2024-02-14_19-55-17", "2024-02-14_20-10-08", "2024-02-15_20-17-26", "2024-02-15_20-47-59",
    "2024-09-02_14-10-41", "2024-09-02_18-42-47", "2024-09-02_21-04-45",
    "2024-09-03_00-44-45", "2024-09-04_17-02-04", "2024-09-04_17-07-59", "2024-09-04_21-09-37",
    "2024-09-05_13-36-00", "2024-09-05_21-26-38", "2024-09-05_22-11-07"
]

for date in dates:
    midi_file_path = f"audio-only/AMT/prediction/midi/path/{date}.mid"
    midi = mido.MidiFile(midi_file_path)

    midi_events = []
    abs_time = 0
    for track in midi.tracks:
        for msg in track:
            abs_time += mido.tick2second(msg.time, midi.ticks_per_beat, mido.bpm2tempo(120))
            midi_events.append((abs_time, msg))

    midi_events.sort(key=lambda x: x[0])

    new_midi = mido.MidiFile()
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)

    bpm = 120
    last_time = 0.0
    deleted_midi_events = []
    video_path = f"/video/path/{date}.mp4"
  
    active_notes = {}
    buffer = []

    for time, msg in midi_events:
        delta_time = time - last_time
        delta_ticks = int(mido.second2tick(delta_time, midi.ticks_per_beat, mido.bpm2tempo(bpm)))

        advance_time = True

        if msg.type == 'note_on' and msg.velocity > 0:
            candidates = find_key_candidates(video_path, time)
            if candidates!=set() and msg.note not in candidates:
                print(f"candidates: {candidates}")
                print(f"msg.note: {msg.note}")
                deleted_midi_events.append(f"time: {time:.2f}s, msg: {msg}")
                print(f"Delete Midi Event: time: {time:.2f}s, msg: {msg}")
                active_notes[msg.note] = None
                advance_time = False
                continue
            else:
                active_notes[msg.note] = True
                buffer.append((delta_ticks, msg))

        elif msg.type == 'note_off':
            if msg.note in active_notes:
                if active_notes[msg.note] is None:
                    deleted_midi_events.append(f"time: {time:.2f}s, msg: {msg}")
                    advance_time = False
                    continue
                else:
                    buffer.append((delta_ticks, msg))
                    del active_notes[msg.note]
            else:
                buffer.append((delta_ticks, msg))
        else:
            buffer.append((delta_ticks, msg))
        if advance_time:
            last_time = time

    for dt, m in buffer:
        new_track.append(m.copy(time=dt))

    print(f"Deleted {len(deleted_midi_events)} events.")

    new_midi_file_path = f'post-processed/midi/path/{date}.mid'
    new_midi.save(new_midi_file_path)

    deleted_log_path = f'deleted/midi-events/logging/path/{date}.txt'
    os.makedirs(os.path.dirname(deleted_log_path), exist_ok=True)
    with open(deleted_log_path, 'w') as f:
        for line in deleted_midi_events:
            f.write(line + '\n')

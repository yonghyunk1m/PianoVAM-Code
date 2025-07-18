# FingeringDetection: Automated System for Detecting Fingering
import streamlit as st
import sys, os
import mediapipe as mp
from midicomparison import *
from fingergt import *
from main import (
    filepath,
    min_hand_detection_confidence,
    min_hand_presence_confidence,
    min_tracking_confidence,
    datagenerate,
)
from streamlit_image_coordinates import streamlit_image_coordinates
import mido
import pickle
import pretty_midi
import cv2
import numpy as np
import stroll    # https://github.com/exeex/midi-visualization with some modifications in order to use in streamlit environment

import subprocess
import json
from tqdm.auto import tqdm
from stqdm import stqdm
import glob

def load_predefined_keypoints(file_path):
    """
    Load predefined keypoints from pixel_points folder and convert to our keystone format
    
    Actual JSON format (20 points):
    - Points 0-1: B0C1_upper, B0C1_lower (edge between B0 and C1)
    - Points 2-3: B1C2_upper, B1C2_lower (edge between B1 and C2)
    - Points 4-5: B2C3_upper, B2C3_lower (edge between B2 and C3)
    - Points 6-7: E4F4_upper, E4F4_lower (edge between E4 and F4)
    - Points 8-9: B5C6_upper, B5C6_lower (edge between B5 and C6)
    - Points 10-11: B6C7_upper, B6C7_lower (edge between B6 and C7)
    - Points 12-13: B7C8_upper, B7C8_lower (edge between B7 and C8)
    - Points 14-19: Meeting points (F1F#1G1, F2F#2G2, F3F#3G3, C5C#5D5, F6F#6G6, F7F#7G7)
    """
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            points = data['pixel_points']
            video_name = data.get('video_name', 'Unknown')
        elif file_path.endswith('.npy'):
            points = np.load(file_path).tolist()
            video_name = os.path.basename(file_path).replace('_pixel_points.npy', '')
        else:
            raise ValueError("Unsupported file format")
        
        if len(points) != 20:
            raise ValueError(f"Expected 20 points, got {len(points)}")
        
        # Get actual video resolution (instead of assuming 1920x1080)
        video_resolution = data.get('video_resolution', [1920, 1080])
        if isinstance(video_resolution, dict):
            video_width = video_resolution.get('width', 1920)
            video_height = video_resolution.get('height', 1080)
        else:
            video_width, video_height = video_resolution
        
        # Convert to normalized coordinates using actual video resolution
        normalized_points = []
        for x, y in points:
            norm_x = x / video_width
            norm_y = y / video_height
            normalized_points.append([norm_x, norm_y])
        
        # Direct mapping to edge format (no conversion needed)
        edge_coords = {}
        
        # Edge points (7 pairs of upper/lower)
        edge_names = ["B0C1", "B1C2", "B2C3", "E4F4", "B5C6", "B6C7", "B7C8"]
        for i, name in enumerate(edge_names):
            upper_idx = i * 2
            lower_idx = i * 2 + 1
            edge_coords[f"{name}_upper"] = normalized_points[upper_idx]
            edge_coords[f"{name}_lower"] = normalized_points[lower_idx]
        
        # Meeting points (6 points starting from index 14)
        meeting_names = ["F1F#1G1", "F2F#2G2", "F3F#3G3", "C5C#5D5", "F6F#6G6", "F7F#7G7"]
        for i, name in enumerate(meeting_names):
            middle_idx = 14 + i
            edge_coords[f"{name}_middle"] = normalized_points[middle_idx]
        
        print(f"🔧 Loaded {len(edge_coords)} edge coordinates from {video_name}")
        print(f"   📐 Video resolution: {video_width}x{video_height}")
        print(f"   🔗 Edge points: {edge_names}")
        print(f"   🎵 Meeting points: {meeting_names}")
        
        return edge_coords, video_name
        
    except Exception as e:
        st.error(f"Error loading predefined keypoints: {e}")
        return None, None

def get_available_predefined_files():
    """Get list of available predefined keypoint files"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pixel_points_dir = os.path.join(script_dir, "pixel_points")
    
    # Check if pixel_points directory exists
    if not os.path.exists(pixel_points_dir):
        return []
    
    # Find all JSON files matching the pattern
    pattern = os.path.join(pixel_points_dir, "*_pixel_points.json")
    json_files = glob.glob(pattern)
    return sorted(json_files, reverse=True)  # Most recent first

# Define edge points for 88-key piano (actual JSON format)
EDGE_POINTS = [
    ("B0C1", 1, "Edge between B0 and C1"),
    ("B1C2", 8, "Edge between B1 and C2"),
    ("B2C3", 15, "Edge between B2 and C3"),
    ("E4F4", 22, "Edge between E4 and F4"),
    ("B5C6", 29, "Edge between B5 and C6"),
    ("B6C7", 36, "Edge between B6 and C7"),
    ("B7C8", 43, "Edge between B7 and C8")
]

# Define meeting points for black keys
MEETING_POINTS = [
    ("F1F#1G1", 5, "F1-F#1-G1 meeting point"),
    ("F2F#2G2", 12, "F2-F#2-G2 meeting point"),
    ("F3F#3G3", 19, "F3-F#3-G3 meeting point"),
    ("C5C#5D5", 26, "C5-C#5-D5 meeting point"),
    ("F6F#6G6", 33, "F6-F#6-G6 meeting point"),
    ("F7F#7G7", 40, "F7-F#7-G7 meeting point")
]

st.set_page_config(layout="wide")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
mididirectory = os.path.join(script_dir, "midiconvert") + "/"
videodirectory = os.path.join(script_dir, "videocapture") + "/"

def delete_smart_tempo(midiname):
    if not "_singletempo.mid" in midiname:
        midi_data = pretty_midi.PrettyMIDI(midiname, initial_tempo=120)
        midi_data.write(midiname[:-4] + "_singletempo.mid")


def intro():
    st.write("# FingeringDetection: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    st.markdown(
        """
        **FingeringDetection : Automated System for Detecting Fingering** is a semi-automatic assistant to label fingering from video. 
        The algorithm only asks confusing fingering for us, so you can either answer the correct fingering from the video or just skip to answer if it is hard to determine the correct fingering even for us.)\\
        
        #### Prerequisites
        - Top-view video
        - Performance MIDI which is recorded from above video
        
        #### Data format for PianoVAM
        - Audio: 16kHz wav format
        - MIDI: mid and Logic Project file
        - Video: Top-View 60fps 720*1280 Video (Full 88 keyboard must be shown.)

        **👈 Select a menu from the dropdown on the left** to create your own
        QR code or starting the record!
    """
    )

    st.write("Settings (three dots at upperright side) - use wide mode")

def initialize_state():
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'responses' not in st.session_state:
        st.session_state.responses = []

# 버튼 입력 함수
def button_input(undecidedtokeninfolist, fps, videoname, newmidiname):
    if len(undecidedtokeninfolist) == 0:
        st.write("No fingers to choose!")
        return ["Complete"]
    buttons=[]
    for tokeninfo in undecidedtokeninfolist:
        buttons.append(tokeninfo[2])
    
    # 버튼 클릭 핸들러
    def button_click(button_name):
        if st.session_state.index < len(buttons):
            st.session_state.history.append(st.session_state.index)
            st.session_state.responses.append([button_name[0],undecidedtokeninfolist[st.session_state.index][0]])   #Finger, token numebr
            st.session_state.index += 1
        st.rerun()

    # Undo 핸들러
    def undo():
        if st.session_state.history:
            st.session_state.index = st.session_state.history.pop()
            st.session_state.responses.pop()
        st.rerun()

    # Reset 핸들러
    def reset():
        st.session_state.index = 0
        st.session_state.history = []
        st.session_state.responses = []
        st.rerun()

    # Complete 핸들러
    def complete():
        st.session_state.responses.append("Complete")
        return st.session_state.responses
    
    # 현재 반복 단계에 따라 버튼을 표시
    if st.session_state.index < len(buttons):
        st.write(
                f"#### Choose the actual finger which pressed {undecidedtokeninfolist[st.session_state.index][1][1]}({pitch_list.index(undecidedtokeninfolist[st.session_state.index][1][1])}) at frame {undecidedtokeninfolist[st.session_state.index][1][0]} or time {str(int(undecidedtokeninfolist[st.session_state.index][1][0]/fps//60)).zfill(2)}:{str(math.floor(undecidedtokeninfolist[st.session_state.index][1][0]/fps%60)).zfill(2)}:"
                + format(
                    math.floor(undecidedtokeninfolist[st.session_state.index][1][0] / fps % 60 * 1000) / 1000
                    - math.floor(undecidedtokeninfolist[st.session_state.index][1][0] / fps % 60),
                    ".2f",
                    )[2:]
                + " : "
                )
    
        col1, col2 = st.columns(2)
        starttime = undecidedtokeninfolist[st.session_state.index][1][0] / fps

        with col1:
            print("preparing video")
            video_file = open(videodirectory + videoname, "rb")
            st.video(video_file, start_time=starttime)
            print("completed preparing video")
        with col2:
            print("preparing midi")
            midi_file_path = mididirectory + newmidiname
            rednoteindex=filter_midi_notes(midi_file_path, undecidedtokeninfolist[st.session_state.index][0])
            mid=stroll.MidiFile(f"{mididirectory}trimmed{undecidedtokeninfolist[st.session_state.index][0]}.mid")
            mid.draw_roll(rednoteidx=rednoteindex)
            os.remove(f"{mididirectory}trimmed{undecidedtokeninfolist[st.session_state.index][0]}.mid")
            print("completed preparing midi")
        st.write(f"Decided {st.session_state.index + 1} of {len(buttons)} undecided fingerings")
        st.write(f"Total frame: {undecidedtokeninfolist[st.session_state.index][3]}")

        # 버튼 표시
        
        user_input=st.text_input("If there are no right candidates, type the finger number from 1 to 10, 1-5 and 6-10 are left/right thumb~little finger.",key=f"{st.session_state.index}-input")
        if st.button('User input',key=f"{st.session_state.index}-inputbutton"):
            st.session_state.history.append(st.session_state.index)
            st.session_state.responses.append([int(user_input),undecidedtokeninfolist[st.session_state.index][0]])   #Finger, token numebr
            st.session_state.index += 1
            st.rerun()
        strval=''

        for button_name in buttons[st.session_state.index]:
            str_button_name = ''
            if len(button_name) > 0:
                for idx, val in enumerate(button_name):
                    if val <= 5:
                        strval=f"L{val}"
                    elif val >= 6:
                        strval=f"R{val-5}"
                    str_button_name += (strval if idx != len(button_name) -1 else str(val)) + (' frames' if idx == len(button_name) -1 else ': ')
            if st.button(str_button_name, key=f"{st.session_state.index}-{button_name[0]}"):
                button_click(button_name)
                break
    else:
        st.write("Completed all steps")

    # Complete 버튼 표시 + txt file 생성
    if st.session_state.index >= len(buttons):
        if st.button("Complete"):
            responses = complete()
            st.write(f"Responses: {responses}")
            with open("./fingering.txt", "r") as fingering_textfile:
                fingering_textlist=fingering_textfile.split("\n")
            with open("./completefingering.txt", "w") as complete_textfile:
                complete_textfile.write("Token number 1~5: Left hand, 6~10: Right hand (Both from thumb finger to little finger) \n")
                human_label_count = 0 
                for i in range(len(fingering_textlist)):
                    if responses[human_label_count][1]==int(fingering_textlist[i].split(",")[0]):
                        fingering_textfile.write(f"Tokennumber={i}, Finger={responses[human_label_count][0]}, \n")
                    else:
                        fingering_textfile.write(f"Tokennumber={i}, Finger={fingering_textlist[i].split(',')[1]}, \n")
                        


    # Undo 버튼 표시
    if st.session_state.history:
        if st.button("Undo"):
            undo()

    # Reset 버튼 표시
    if st.session_state.index >= len(buttons):
        if st.button("Reset"):
            reset()

    # 현재 상태 및 응답 출력
    st.write(f"Current Index: {st.session_state.index}")
    st.write(f"Responses: {st.session_state.responses}")


def sthanddecider(tokenlist, keyhandlist):
    if 'index' not in st.session_state:
        st.session_state.index = 0

    pressedfingerlist = [None for k in range(len(tokenlist))]
    undecidedtokeninfolist = []

    correct=0
    undecided=0
    noinfo=0
    for i in range(len(tokenlist)):
        totalframe=tokenlist[i][2]-tokenlist[i][0]
        tokenlist[i].pop(2)
        tokenlist[i][1] = pitch_list[tokenlist[i][1]]
        lefthandcounter = 0
        righthandcounter = 0
        noinfocounter = 0
        lhindex = []
        rhindex = []
        fingerindex = [[],[],[],[],[],[],[],[],[],[],0]  # Ten fingers & Noinfo counter
        fingerscore = [0,0,0,0,0, 0,0,0,0,0]
        for j in range(len(keyhandlist)):
            framekeyhandinfo = keyhandlist[j]
            for keyhandinfo in framekeyhandinfo:
                if keyhandinfo[1] == i:
                    if keyhandinfo[2] == "Left":
                        lefthandcounter += 1
                        lhindex.append(j)
                        for k in range(1, 11):
                            if keyhandinfo[3][k] != 0:
                                fingerindex[k - 1].append(j)
                                fingerscore[k - 1] += keyhandinfo[3][k]
                        if keyhandinfo[3] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                            fingerindex[10] += 1
                    elif keyhandinfo[2] == "Right":
                        righthandcounter += 1
                        rhindex.append(j)
                        for k in range(1, 11):
                            if keyhandinfo[3][k] != 0:
                                fingerindex[k - 1].append(j)
                                fingerscore[k - 1] += keyhandinfo[3][k]
                        if keyhandinfo[3] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                            fingerindex[10] += 1
                    elif keyhandinfo[2] == "Noinfo":
                        noinfocounter += 1
                        fingerindex[10] += 1
        counterlist = [
            lefthandcounter,
            righthandcounter,
            noinfocounter,
            (lefthandcounter + righthandcounter + noinfocounter) / 2,
        ]
        if counterlist.index(max(counterlist)) == 0:
            tokenlist[i].append("Left")

        elif counterlist.index(max(counterlist)) == 1:
            tokenlist[i].append("Right")

        else:
            tokenlist[i].append("Noinfo")
        print(
            f"Tokennumber:{i},   Tokenpitch={tokenlist[i][1]},    lefthandcounter={len(lhindex)},     righthandcounter={len(rhindex)}, noinfocounter ={noinfocounter}, fingercount={fingerscore}, fingernoinfo={fingerindex[10]}"
        )
        pressedfingers = []
        highcandidates = []
        totalfingercount = 0
        c = 0
        for j in range(10):
            if fingerscore[j] != 0:
                pressedfingers.append([j + 1,fingerscore[j]])
                totalfingercount += fingerscore[j]

        while c < len(pressedfingers):
            finger=pressedfingers[c]
            if finger[1]/totalframe <0.5: # We choose a finger as a candidate if its score is more than a half of total frames.
                pressedfingers.pop(c)
                c -= 1
            if finger[1]/totalframe >0.80: # If the fingering score is more than 80% of total frames, we let the finger "strong candidate".
                highcandidates.append(finger)
            c += 1

        gt=jeuxdeau_150
        if c > 1:   # If there are multiple candidates:
            if len(highcandidates) == 1:
                pressedfingerlist[i] = highcandidates[0][0]
                if i <= 149:
                    if pressedfingerlist[i] == gt[i]:
                        correct += 1
                    else: print(f"tokennumber: {i}, gt: {gt[i]}, pred:{pressedfingerlist[i]}")
            else:
                undecidedtokeninfolist.append([i,tokenlist[i],pressedfingers,totalframe])  #Token number, token info, finger candidate, frame count of the token
                undecided += 1
                print(f"tokennumber: {i}, undecided")
        elif c == 1:
            pressedfingerlist[i] = pressedfingers[0][0]
            if i <= 149:
                if pressedfingerlist[i] == gt[i]:
                    correct += 1
                else: print(f"tokennumber: {i}, gt: {gt[i]}, pred:{pressedfingerlist[i]}")

        elif c == 0:
            pressedfingerlist[i] = "Noinfo"
            noinfo += 1
            undecidedtokeninfolist.append([i,tokenlist[i],[],totalframe])
            print(f"tokennumber: {i}, noinfo")
    
        
        if i==149:
            st.write(f'150 Accuracy: {correct/(150-noinfo-undecided)}, 150 noinfo: {noinfo}, 150 undecided: {undecided}')
        if i==len(tokenlist)-1:
            st.write(f'total tokens: {len(tokenlist)}, total noinfo: {noinfo/len(tokenlist)}, total undecided: {undecided/len(tokenlist)}')
    return pressedfingerlist, undecidedtokeninfolist
def decider(pressedfingerlist,undecidedtokeninfolist, fps, videoname, newmidiname):
    decision=button_input(undecidedtokeninfolist, fps, videoname, newmidiname)
    if decision:   #Exclude decidion=None
        if len(decision) == len(undecidedtokeninfolist)+1: # +1: complete
            for j in range(len(pressedfingerlist)):
                if pressedfingerlist[j] == "Noinfo" and decision[0][1]==j:
                    pressedfingerlist[j] = decision[0]
                    decision.pop(0)

    return pressedfingerlist  # Output: [Pitch(0~95(C0~ G8)), token number, hand]

def videodata():
    st.sidebar.success("Select the menu above.")
    st.write('make mediapipe data from video')
    files = os.listdir(filepath)
    newfiles = []
    for file in files:
        if ".mp4" in file:
            newfiles.append(file)
    newfiles.sort()
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )
    if st.button('Generate mediapipe data'):
        datagenerate(selected_option)
        st.write(f'Generated data of {filepath + selected_option}')

    
    

def preprocess():
    st.write('delete smart tempo which is automatically recorded by logic')
    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if not "_singletempo" in str(file):
            newfiles.append(file)
    newfiles.sort()
    st.write("Settings (three dots at upperright side) - use wide mode")
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )
    if st.button('Delete smart tempo'):
        delete_smart_tempo(os.path.join(mididirectory, selected_option))
        st.write(f'Changed {os.path.join(mididirectory, selected_option)}.')

def get_video_fps(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = json.loads(result.stdout.decode())
    fps = eval(data['streams'][0]['r_frame_rate'])  # Convert string to fraction (e.g., '30000/1001')
    return fps

def prefinger():
    st.write("# FingeringDetection: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    
    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if "_singletempo" in str(file):
            newfiles.append(file)
    newfiles.sort()
    st.write("Settings (three dots at upperright side) - use wide mode")
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )

    # Change the tempo to the desired BPM
    
    # Save the modified MIDI file

    newmidiname = selected_option
    videoname = '_'.join(selected_option.split("_")[:-1]) + ".mp4"

    st.write("Selected MIDI:", selected_option)

    if st.button("Precorrespond fingering"):
        st.write(
            'fingering pre-labeling started'
        )
        video = cv2.VideoCapture(os.path.join(filepath, videoname))
        if not video.isOpened():
            print("Error: Failed to open video.")
        frame_rate = get_video_fps(os.path.join(filepath, videoname))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_count)
        dirname = (os.path.join(filepath,
                videoname[:-4]
                + "_"
                + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            )
        )
        with open(
            dirname
            + "/floatingframes_"
            + videoname[:-4]
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl",
            "rb",
        ) as f:
            floatingframes = pickle.load(f)

        with open(
            dirname
            + "/handlist_"
            + videoname[:-4]
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl",
            "rb",
        ) as f:
            handlist = pickle.load(f)
        
        with open(
            os.path.join(script_dir, "keyboardcoordinateinfo.pkl"),
            "rb",
        ) as f:
            keyboardcoordinateinfo = pickle.load(f)
        # Get keyboard data for this video
        video_keyboard_data = keyboardcoordinateinfo[videoname[:-4]]
        keyboard = generatekeyboard(video_keyboard_data)
        handfingerpositionlist = []
        tokenlist = miditotoken(newmidiname[:-4], frame_rate, "simplified")
        iter=0
        pbar = tqdm(total=len(handlist))
        for _ in stqdm(range(len(handlist)), desc="Detecting finger position information from frame images..."):
            if iter<len(handlist):
                handsinfo=handlist[iter]
                handfingerposition = handpositiondetector(
                    handsinfo, floatingframes, keyboard
                )
                handfingerpositionlist.append(handfingerposition)
                iter += 1 
            else:
                break
        pbar.close()
        prefingercorrespond =handfingercorresponder(
                tokentoframeinfo(tokenlist, frame_count), handfingerpositionlist, keyboard, tokenlist
            )
        
        fingerinfo,undecidedfingerlist = sthanddecider(
            tokenlist,
            prefingercorrespond,
        )
        
        with open("./fingering.txt", "w") as fingering_textfile:            
            for i in range(len(fingerinfo)):
                fingering_textfile.write(f"{i},{fingerinfo[i]}, \n")



        with open(
            dirname
            + "/fingerinfo_"
            + videoname
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(fingerinfo, f)

        with open(
            dirname
            + "/undecidedfingerlist_"
            + videoname
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(undecidedfingerlist, f)
        st.write("Prefinger info saved")

def keyboardcoordinate():
    st.sidebar.success("Select the menu above.")
    st.write("# 🎹 Keyboard Keystone Points Collection")
    st.write("Collect keystone points across the piano keyboard for accurate interpolation")
    
    # Initialize session state for keystone points
    if 'keystone_coords' not in st.session_state:
        st.session_state.keystone_coords = {}
    if 'current_keystone_index' not in st.session_state:
        st.session_state.current_keystone_index = 0
    if 'collecting_upper' not in st.session_state:
        st.session_state.collecting_upper = True
    if 'current_frame_number' not in st.session_state:
        st.session_state.current_frame_number = 0
    
    files = os.listdir(filepath)
    newfiles = []
    for file in files:
        if ".mp4" in file:
            newfiles.append(file)
    newfiles.sort()
    selected_option = st.selectbox(
        "Select video files:",
        newfiles,
    )

    # Preview existing keyboard configuration for this video
    st.write("---")
    st.write("### 🔍 Preview Existing Configuration")
    
    # Check if this video has existing keyboard configuration
    try:
        with open(os.path.join(script_dir, "keyboardcoordinateinfo.pkl"), "rb") as f:
            keyboardcoordinateinfo = pickle.load(f)
        
        video_key = selected_option[:-4]
        if video_key in keyboardcoordinateinfo and video_key != "Status":
            keyboard_data = keyboardcoordinateinfo[video_key]
            
            # Determine the format
            if isinstance(keyboard_data, list) and len(keyboard_data) == 8:
                format_type = "Old Format (4-corner + distortion)"
                format_details = "lu, ru, ld, rd, blackratio, ldistortion, rdistortion, cdistortion"
            elif isinstance(keyboard_data, dict) and 'keystone_points' in keyboard_data:
                format_type = "New Format (keystone points)"
                format_details = f"{len(keyboard_data['keystone_points'])} keystone points"
            else:
                format_type = "Unknown Format"
                format_details = str(type(keyboard_data))
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"📋 **Existing configuration found**\n- Format: {format_type}\n- Details: {format_details}")
            
            with col2:
                if st.button("👁️ Preview Existing"):
                    try:
                        from floatinghands_torch_pure import generatekeyboard, draw_keyboard_on_image
                        import mediapipe as mp
                        
                        # Generate keyboard using existing configuration (backward compatible)
                        keyboard = generatekeyboard(keyboard_data)
                        
                        # Get video frame for overlay
                        video = cv2.VideoCapture(os.path.join(filepath, selected_option))
                        ret, frame = video.read()
                        video.release()
                        
                        if ret:
                            # Draw keyboard on image
                            img_np = np.array(frame)
                            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                            keyboard_image = draw_keyboard_on_image(img.numpy_view(), keyboard)
                            keyboard_image_rgb = cv2.cvtColor(keyboard_image, cv2.COLOR_BGR2RGB)
                            
                            # Display result
                            st.image(keyboard_image_rgb, caption=f"🎹 Existing Keyboard Configuration for {video_key}", use_column_width=True)
                            
                            # Show stats
                            st.success(f"""
                            ✅ **Existing Configuration Preview**
                            - **Format**: {format_type}
                            - **Keys generated**: {len(keyboard)} keys
                            - **Status**: Working correctly with backward compatibility
                            """)
                        else:
                            st.error("❌ Could not read video frame")
                            
                    except Exception as e:
                        st.error(f"❌ Preview failed: {e}")
                        st.exception(e)
        else:
            st.warning(f"⚠️ No existing keyboard configuration found for **{video_key}**")
            st.info("💡 You can collect new keystone points below.")
    
    except FileNotFoundError:
        st.warning("⚠️ No keyboardcoordinateinfo.pkl file found")
        st.info("💡 You can collect new keystone points below.")
    except Exception as e:
        st.error(f"❌ Error checking existing configuration: {e}")

    # Add predefined keypoints loading option
    st.write("---")
    st.write("### 🔄 Load Predefined Keypoints")
    st.info("💡 You can load previously collected keypoints from the pixel_points folder, or manually collect new ones below.")
    
    # Get available predefined files
    predefined_files = get_available_predefined_files()
    
    if predefined_files:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            selected_predefined = st.selectbox(
                "📁 Select predefined keypoint file:",
                ["None (Manual Collection)"] + [os.path.basename(f) for f in predefined_files],
                help="Choose a predefined keypoint file or select 'None' to collect manually"
            )
        
        with col2:
            if st.button("📥 Load", disabled=(selected_predefined == "None (Manual Collection)")):
                if selected_predefined != "None (Manual Collection)":
                    # Find the full path
                    file_path = None
                    for f in predefined_files:
                        if os.path.basename(f) == selected_predefined:
                            file_path = f
                            break
                    
                    if file_path:
                        keystone_coords, video_name = load_predefined_keypoints(file_path)
                        if keystone_coords:
                            st.session_state.keystone_coords = keystone_coords
                            st.success(f"✅ Loaded {len(keystone_coords)} keystone points from {video_name}")
                            st.balloons()
                            st.rerun()
        
        with col3:
            if st.button("🗑️ Clear"):
                st.session_state.keystone_coords = {}
                st.session_state.current_keystone_index = 0
                st.success("✅ Cleared all keystone points")
                st.rerun()
        
        # Show current keystone status
        if st.session_state.keystone_coords:
            st.write("### 📋 Current Keystone Points")
            keystone_summary = {}
            for key in st.session_state.keystone_coords:
                if '_' in key:
                    base_name = key.rsplit('_', 1)[0]
                    if base_name not in keystone_summary:
                        keystone_summary[base_name] = []
                    keystone_summary[base_name].append(key.rsplit('_', 1)[1])
            
            cols = st.columns(4)
            for i, (base_name, point_types) in enumerate(keystone_summary.items()):
                with cols[i % 4]:
                    st.write(f"**{base_name}**: {', '.join(point_types)}")
    else:
        st.warning("📂 No predefined keypoint files found in pixel_points folder")
    
    st.write("---")

    # Open video and get frame information
    video = cv2.VideoCapture(os.path.join(filepath, selected_option))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set frame position and read frame
    video.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_number)
    ret, image = video.read()
    video.release()
    
    if ret:
        cv2.imwrite("tmp.jpg", image)
    else:
        st.error("❌ Could not read video frame")
        return
    # Frame information and navigation controls
    st.write("### 📹 Frame Navigation")
    col_frame1, col_frame2, col_frame3, col_frame4 = st.columns([2, 1, 1, 1])
    
    with col_frame1:
        st.info(f"📹 Current frame: {st.session_state.current_frame_number + 1} / {total_frames}")
    
    with col_frame2:
        if st.button("🔄 Random", help="Switch to a random frame if hands are blocking the keyboard"):
            import random
            st.session_state.current_frame_number = random.randint(0, total_frames - 1)
            st.rerun()
    
    with col_frame3:
        if st.button("⬅️ -100", help="Go back 100 frames"):
            st.session_state.current_frame_number = max(0, st.session_state.current_frame_number - 100)
            st.rerun()
    
    with col_frame4:
        if st.button("➡️ +100", help="Go forward 100 frames"):
            st.session_state.current_frame_number = min(total_frames - 1, st.session_state.current_frame_number + 100)
            st.rerun()
    
    # Frame slider for precise navigation
    new_frame = st.slider(
        "🎬 Navigate to specific frame:", 
        min_value=0, 
        max_value=total_frames - 1, 
        value=st.session_state.current_frame_number,
        help="Drag to navigate to a specific frame"
    )
    
    if new_frame != st.session_state.current_frame_number:
        st.session_state.current_frame_number = new_frame
        st.rerun()
    
    value = streamlit_image_coordinates(
        "tmp.jpg",
        key="local4",
        use_column_width="always",
        click_and_drag=True,
    )
    
    # Calculate total points: Edge points need upper+lower, meeting points need only middle
    edge_points = [k for k in EDGE_POINTS]  # All edge points
    meeting_points = [k for k in MEETING_POINTS]  # All meeting points
    expected_total = len(edge_points) * 2 + len(meeting_points)  # Edge points: upper+lower, meeting points: middle only
    
    collected_points = len(st.session_state.keystone_coords)
    progress = collected_points / expected_total
    
    st.progress(progress)
    st.write(f"📊 Progress: {collected_points}/{expected_total} points collected")
    
    # Current collection instruction
    all_points = EDGE_POINTS + MEETING_POINTS
    if st.session_state.current_keystone_index < len(all_points):
        current_point = all_points[st.session_state.current_keystone_index]
        point_name, point_index, description = current_point

        # Determine if this is a meeting point or edge point
        is_meeting_point = point_name in [p[0] for p in MEETING_POINTS]
        

        if is_meeting_point:
            # Meeting points only need middle position
            position = "MIDDLE"
            st.write(f"### 🎯 Current Target: {point_name} - {position}")
            st.write(f"**{description}** - Click the meeting point")
        else:
            # Edge points need upper and lower
            position = "UPPER" if st.session_state.collecting_upper else "LOWER"
            st.write(f"### 🎯 Current Target: {point_name} - {position}")
            st.write(f"**{description}** - Click the **{position.lower()}** point")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if value:
                st.write(f"📍 Current position: ({value['x1']}, {value['y1']})")
        
        with col2:
            if st.button(f"✅ Collect {point_name} {position}", key=f"collect_{point_name}_{position}"):
                if value:
                    if is_meeting_point:
                        coord_key = f"{point_name}_middle"
                    else:
                        coord_key = f"{point_name}_{position.lower()}"
                    
                    normalized_coord = [value["x1"]/value['width'], value['y1']/value['height']]
                    st.session_state.keystone_coords[coord_key] = normalized_coord
                    
                    # Move to next point
                    if is_meeting_point:
                        # Meeting point collected, move to next point
                        st.session_state.current_keystone_index += 1
                        st.session_state.collecting_upper = True  # Reset for next edge point
                    else:
                        # Edge point: toggle between upper and lower
                        if st.session_state.collecting_upper:
                            st.session_state.collecting_upper = False
                        else:
                            st.session_state.collecting_upper = True
                            st.session_state.current_keystone_index += 1
                    
                    st.rerun()
                else:
                    st.error("Please click on the image first!")
    
    # Display collected points
    if st.session_state.keystone_coords:
        st.write("### 📌 Collected Edge Points:")
        for coord_key, coord in st.session_state.keystone_coords.items():
            st.write(f"• {coord_key}: ({coord[0]:.4f}, {coord[1]:.4f})")
    
    # Keyboard visualization - works with any amount of data
    st.write("### 🎹 Keyboard Preview:")
    
    # Determine what data we have available for preview
    collected_keystones = len(st.session_state.keystone_coords)
    has_predefined_file = False
    predefined_file_name = None
    
    # Check for predefined files for this video
    video_key = selected_option[:-4]
    predefined_files = get_available_predefined_files()
    
    # Look for a matching predefined file
    for file_path in predefined_files:
        if video_key.lower() in os.path.basename(file_path).lower():
            has_predefined_file = True
            predefined_file_name = os.path.basename(file_path)
            break
    
    # Show available preview options
    if collected_keystones > 0 and collected_keystones >= expected_total:
        st.write("✅ **All keystone points collected** - Preview using direct interpolation method:")
        preview_method = "new_complete"
    elif collected_keystones > 0:
        st.write(f"🔄 **{collected_keystones}/{expected_total} keystone points collected** - Preview with current points:")
        preview_method = "new_partial"
    elif has_predefined_file:
        st.write(f"📁 **Using predefined keystone points** - {predefined_file_name}:")
        preview_method = "predefined"
    else:
        st.write("⚠️ **No data available** - Collect keystone points or load predefined configuration:")
        preview_method = "none"
    
    if preview_method != "none":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if preview_method == "new_complete":
                st.info("""
                💡 **Edge-Based Interpolation Method:**
                - White key edges: Linear interpolation between edge points
                - Black key meeting points: Use exact collected meeting point coordinates  
                - Other black keys: Interpolate between edge points
                - Example: Point between E4-F4 uses E4F4 edge coordinates directly
                """)
            elif preview_method == "new_partial":
                st.warning(f"""
                ⚠️ **Partial Preview (Not all points collected)**
                - Collected: {collected_keystones}/{expected_total} points
                - Missing points will be estimated or cause preview to fail
                - Continue collecting for accurate preview
                """)
            elif preview_method == "predefined":
                st.info("""
                📁 **Predefined Edge Points**
                - Uses saved edge points from file
                - Direct interpolation between edge points
                - All edge points available from previous collection
                """)
        
        with col2:
            if st.button("🔄 Refresh Preview"):
                try:
                    from floatinghands_torch_pure import generatekeyboard, draw_keyboard_on_image
                    import mediapipe as mp
                    
                    keystone_points = None
                    data_source = None
                    
                    # Priority 1: Use current session state keystone points
                    if hasattr(st.session_state, 'keystone_coords') and st.session_state.keystone_coords:
                        keystone_points = st.session_state.keystone_coords
                        data_source = "Current Session"
                        st.info(f"🎯 Using edge points from current session ({len(keystone_points)} points)")
                    
                    # Priority 2: Try to load from predefined files for this video
                    if keystone_points is None:
                        video_key = selected_option[:-4]
                        predefined_files = get_available_predefined_files()
                        
                        # Look for a matching predefined file
                        matching_file = None
                        for file_path in predefined_files:
                            if video_key.lower() in os.path.basename(file_path).lower():
                                matching_file = file_path
                                break
                        
                        if matching_file:
                            keystone_coords, video_name = load_predefined_keypoints(matching_file)
                            if keystone_coords:
                                keystone_points = keystone_coords
                                data_source = f"Predefined File: {os.path.basename(matching_file)}"
                                st.info(f"📁 Using edge points from predefined file ({len(keystone_points)} points)")
                    
                    # Priority 3: Check if user has collected points but not in expected format
                    if keystone_points is None:
                        st.error("""
                        ❌ **No Edge Points Available**
                        
                        No edge points found for this video.
                        
                        **To fix this:**
                        1. Collect edge points using the interface above, OR
                        2. Load predefined edge points from a file, OR  
                        3. Make sure you have saved edge points for this video
                        
                        The preview requires edge points to generate the keyboard layout.
                        """)
                        return
                    
                    # Validate edge points format
                    if not isinstance(keystone_points, dict):
                        st.error("❌ **Invalid edge points format** - Expected dictionary format")
                        return
                    
                    # Generate keyboard using keystone points
                    keyboard_result = generatekeyboard(keystone_points)
                    
                    print(f"🔍 ASDF - keyboard_result type: {type(keyboard_result)}")
                    print(f"🔍 ASDF - keyboard_result length: {len(keyboard_result) if hasattr(keyboard_result, '__len__') else 'N/A'}")
                    
                    # Handle new return format (keyboard, black_key_data)
                    if isinstance(keyboard_result, tuple) and len(keyboard_result) == 2:
                        keyboard, black_key_data = keyboard_result
                        print(f"🔍 ASDF - Using new format: keyboard={len(keyboard)}, black_key_data={black_key_data is not None}")
                    else:
                        # Fallback for old format
                        keyboard = keyboard_result
                        black_key_data = None
                        print(f"🔍 ASDF - Using old format: keyboard={len(keyboard)}, black_key_data=None")
                    
                    # Load video frame
                    video = cv2.VideoCapture(os.path.join(filepath, selected_option))
                    ret, frame = video.read()
                    video.release()
                    
                    if ret:
                        # Convert to MediaPipe format and draw keyboard with points
                        img_np = np.array(frame)
                        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                        
                        # Use enhanced drawing function that shows all intermediate points
                        from floatinghands_torch_pure import draw_keyboard_with_points
                        keyboard_image = draw_keyboard_with_points(img.numpy_view(), keyboard, keystone_points, black_key_data, show_intermediate_points=True)
                        keyboard_image_rgb = cv2.cvtColor(keyboard_image, cv2.COLOR_BGR2RGB)
                            
                        # Display result
                        st.image(keyboard_image_rgb, caption="🎹 Keyboard Preview with Colored Key Polygons", use_column_width=True)
                        
                        # Add legend for the colored lines
                        st.write("#### 🎨 Key Polygon Legend:")
                        st.write("🔴 **Red Lines**: White key polygons")
                        st.write("- Each white key is outlined with red lines")
                        st.write("🟢 **Green Lines**: Black key polygons") 
                        st.write("- Each black key is outlined with green lines")
                        st.write("📐 **Polygon Structure**: Each key shows its complete boundary")
                        
                        # Show detailed keyboard structure
                        st.write("### 📊 Keyboard Structure Analysis:")
                        
                        # Analyze the generated keyboard
                        total_keys = len(keyboard)
                        white_keys = 0
                        black_keys = 0
                        
                        # Count key types and analyze structure
                        key_analysis = []
                        for i, key_polygon in enumerate(keyboard):
                            num_points = len(key_polygon)
                            if num_points == 5:  # Simple white key (rectangle)
                                white_keys += 1
                                key_type = "White Key (Simple)"
                            elif num_points == 7:  # White key with one black key neighbor
                                white_keys += 1
                                key_type = "White Key (1 Black Neighbor)"
                            elif num_points == 9:  # White key with two black key neighbors
                                white_keys += 1
                                key_type = "White Key (2 Black Neighbors)"
                            else:
                                black_keys += 1
                                key_type = f"Black Key ({num_points} points)"
                            
                            key_analysis.append({
                                'index': i,
                                'type': key_type,
                                'points': num_points,
                                'coordinates': key_polygon[:3]  # Show first 3 points for brevity
                            })
                        
                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Keys", total_keys)
                        with col2:
                            st.metric("White Keys", white_keys)
                        with col3:
                            st.metric("Black Keys", black_keys)
                        
                        # Show detailed breakdown for first few keys
                        st.write("#### 🔍 Key Structure Details (First 10 keys):")
                        for i, key_info in enumerate(key_analysis[:10]):
                            with st.expander(f"Key {i}: {key_info['type']} ({key_info['points']} points)"):
                                st.write(f"**All Coordinates:**")
                                for j, point in enumerate(key_info['coordinates']):
                                    st.write(f"  Point {j}: ({point[0]:.4f}, {point[1]:.4f})")
                                if len(key_info['coordinates']) < key_info['points']:
                                    st.write(f"  ... and {key_info['points'] - len(key_info['coordinates'])} more points")
                        
                        # Show raw toppoints and bottompoints analysis
                        st.write("#### 📐 Raw Points Analysis:")
                        
                        # Import the function to get raw points
                        from floatinghands_torch_pure import generate_keyboard_from_keystone_points
                        
                        try:
                            # Get the raw toppoints and bottompoints
                            bottompoints, toppoints = generate_keyboard_from_keystone_points(keystone_points)
                            
                            st.write(f"**Generated {len(toppoints)} toppoints and {len(bottompoints)} bottompoints**")
                            
                            # Show first 10 toppoints and bottompoints
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Toppoints (first 10):**")
                                for i in range(min(10, len(toppoints))):
                                    st.write(f"  {i}: ({toppoints[i][0]:.4f}, {toppoints[i][1]:.4f})")
                            
                            with col2:
                                st.write("**Bottompoints (first 10):**")
                                for i in range(min(10, len(bottompoints))):
                                    st.write(f"  {i}: ({bottompoints[i][0]:.4f}, {bottompoints[i][1]:.4f})")
                            
                            # Show black key analysis if available
                            st.write("#### 🎹 Black Key Analysis:")
                            
                            # Import black key generation function
                            from floatinghands_torch_pure import create_interpolation_functions
                            
                            try:
                                upper_coords, lower_coords = create_interpolation_functions(keystone_points)
                                st.write(f"**Black key interpolation coordinates:**")
                                st.write(f"  Upper coords: {len(upper_coords)} points")
                                st.write(f"  Lower coords: {len(lower_coords)} points")
                                
                                # Show first few black key interpolation points
                                st.write("**First 5 upper coordinates:**")
                                for i, (pos, x, y) in enumerate(upper_coords[:5]):
                                    st.write(f"  Position {pos}: ({x:.4f}, {y:.4f})")
                                
                                st.write("**First 5 lower coordinates:**")
                                for i, (pos, x, y) in enumerate(lower_coords[:5]):
                                    st.write(f"  Position {pos}: ({x:.4f}, {y:.4f})")
                                
                                # Show the actual black key points generated from the offset formulas
                                st.write("#### 🔧 Black Key Offset Points:")
                                
                                # Import the black key generation function to get the actual points
                                from floatinghands_torch_pure import generatekeyboard
                                
                                # We need to access the internal black key generation
                                # Let's create a simple version to show the offset points
                                black_key_indices = [n for n in range(1, 52) if ((n) % 7) in [0, 1, 3, 4, 6]]
                                
                                st.write(f"**Black key positions:** {black_key_indices[:10]}... (total: {len(black_key_indices)})")
                                
                                # Show the offset formulas and generated points for first 3 black keys
                                for i, black_idx in enumerate(black_key_indices[:3]):
                                    st.write(f"**Black Key {i+1} (Position {black_idx}):**")
                                    
                                    # Find interpolation weights for this black key
                                    interval_start_idx = 0
                                    interval_end_idx = 1
                                    
                                    for j in range(len(upper_coords) - 1):
                                        if upper_coords[j][0] <= black_idx <= upper_coords[j + 1][0]:
                                            interval_start_idx = j
                                            interval_end_idx = j + 1
                                            break
                                    
                                    # Calculate interpolation weights
                                    start_pos = upper_coords[interval_start_idx][0]
                                    end_pos = upper_coords[interval_end_idx][0]
                                    
                                    if end_pos == start_pos:
                                        weight_end = 0
                                        weight_start = 1
                                    else:
                                        weight_end = (black_idx - start_pos) / (end_pos - start_pos)
                                        weight_start = 1 - weight_end
                                    
                                    # Interpolate corner positions (lu, ru, ld, rd)
                                    lu = [
                                        weight_start * upper_coords[interval_start_idx][1] + weight_end * upper_coords[interval_end_idx][1],
                                        weight_start * upper_coords[interval_start_idx][2] + weight_end * upper_coords[interval_end_idx][2]
                                    ]
                                    ru = [
                                        weight_start * upper_coords[interval_start_idx][1] + weight_end * upper_coords[interval_end_idx][1],
                                        weight_start * upper_coords[interval_start_idx][2] + weight_end * upper_coords[interval_end_idx][2]
                                    ]
                                    ld = [
                                        weight_start * lower_coords[interval_start_idx][1] + weight_end * lower_coords[interval_end_idx][1],
                                        weight_start * lower_coords[interval_start_idx][2] + weight_end * lower_coords[interval_end_idx][2]
                                    ]
                                    rd = [
                                        weight_start * lower_coords[interval_start_idx][1] + weight_end * lower_coords[interval_end_idx][1],
                                        weight_start * lower_coords[interval_start_idx][2] + weight_end * lower_coords[interval_end_idx][2]
                                    ]
                                    
                                    st.write(f"  **Corner points:**")
                                    st.write(f"    lu: ({lu[0]:.4f}, {lu[1]:.4f})")
                                    st.write(f"    ru: ({ru[0]:.4f}, {ru[1]:.4f})")
                                    st.write(f"    ld: ({ld[0]:.4f}, {ld[1]:.4f})")
                                    st.write(f"    rd: ({rd[0]:.4f}, {rd[1]:.4f})")
                                    
                                    # Show top black points (6 points with 1/12 offsets)
                                    st.write(f"  **Top black points (6 points with 1/12 offsets):**")
                                    top_offsets = [1/4, -1/4, 1/4+1/12, -1/4+1/12, 1/4-1/12, -1/4-1/12]
                                    for j, offset in enumerate(top_offsets):
                                        top_x = lu[0] * (52 - black_idx + offset) / 52 + ru[0] * (black_idx - offset) / 52
                                        top_y = lu[1] * (52 - black_idx + offset) / 52 + ru[1] * (black_idx - offset) / 52 - 0.01
                                        st.write(f"    Point {j+1} (offset {offset:.3f}): ({top_x:.4f}, {top_y:.4f})")
                                    
                                    # Show bottom black points (7 points with 1/12 offsets)
                                    st.write(f"  **Bottom black points (7 points with 1/12 offsets):**")
                                    bottom_offsets = [1/4, 0, -1/4, 1/4+1/12, -1/4+1/12, 1/4-1/12, -1/4-1/12]
                                    blackratio = 0.6  # Default value
                                    for j, offset in enumerate(bottom_offsets):
                                        bottom_x = lu[0] * (52 - black_idx + offset) / 52 + ru[0] * (black_idx - offset) / 52
                                        bottom_y = (lu[1] * (52 - black_idx + offset) / 52 + ru[1] * (black_idx - offset) / 52) * (1 - blackratio) + (ld[1] * (52 - black_idx + offset) / 52 + rd[1] * (black_idx - offset) / 52) * blackratio
                                        st.write(f"    Point {j+1} (offset {offset:.3f}): ({bottom_x:.4f}, {bottom_y:.4f})")
                                    
                                    st.write("---")
                                    
                            except Exception as e:
                                st.warning(f"Could not analyze black key coordinates: {e}")
                            
                            # Show black key structure analysis
                            st.write("#### 🎹 Black Key Structure:")
                            
                            try:
                                # Find black keys in the keyboard
                                black_key_indices = [n for n in range(1, 52) if ((n) % 7) in [0, 1, 3, 4, 6]]
                                st.write(f"**Black key positions:** {black_key_indices[:10]}... (total: {len(black_key_indices)})")
                                
                                # Show black key structure for first few black keys
                                st.write("**Black key structure (first 5):**")
                                for i, black_idx in enumerate(black_key_indices[:5]):
                                    # Find the corresponding key in the keyboard
                                    if black_idx < len(keyboard):
                                        key_polygon = keyboard[black_idx]
                                        st.write(f"  Black key at position {black_idx}: {len(key_polygon)} points")
                                        for j, point in enumerate(key_polygon[:3]):  # Show first 3 points
                                            st.write(f"    Point {j}: ({point[0]:.4f}, {point[1]:.4f})")
                                        if len(key_polygon) > 3:
                                            st.write(f"    ... and {len(key_polygon) - 3} more points")
                                    
                            except Exception as e:
                                st.warning(f"Could not analyze black key structure: {e}")
                                
                        except Exception as e:
                            st.warning(f"Could not analyze raw points: {e}")
                        
                        # Show success message
                        st.success(f"""
                        ✅ **Preview Generated Successfully**
                        - **Keys generated**: {total_keys} (88 piano keys)
                        - **White keys**: {white_keys}
                        - **Black keys**: {black_keys}
                        - **Video**: {video_key}
                        - **Method**: Edge-based interpolation
                        """)

                    else:
                        st.error(f"❌ No keyboard configuration found for {video_key}")
                        
                except Exception as e:
                    st.error(f"❌ Preview generation failed: {e}")
                    st.exception(e)
    
    # Save section
    st.write("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if collected_keystones > 0:
            progress_pct = int((collected_keystones / expected_total) * 100)
            st.metric("Collection Progress", f"{collected_keystones}/{expected_total}", f"{progress_pct}%")
    
    with col2:
        if has_predefined_file:
            st.metric("Predefined File", "Available", predefined_file_name)

    with col3:
        if collected_keystones >= expected_total:
            st.metric("Status", "Ready", "Complete")
        else:
            remaining = expected_total - collected_keystones
            st.metric("Remaining", f"{remaining}", "points")

    # Control buttons
    st.write("### 🎛️ Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All"):
            st.session_state.keystone_coords = {}
            st.session_state.current_keystone_index = 0
            st.session_state.collecting_upper = True
            st.success("✅ Reset all edge points")
            st.rerun()
    
    with col2:
        if st.button("↶ Undo Last"):
            if st.session_state.keystone_coords:
                # Remove last added coordinate
                last_key = list(st.session_state.keystone_coords.keys())[-1]
                del st.session_state.keystone_coords[last_key]
                
                # Update collection state
                if st.session_state.collecting_upper:
                    st.session_state.collecting_upper = False
                    if st.session_state.current_keystone_index > 0:
                        st.session_state.current_keystone_index -= 1
                else:
                    st.session_state.collecting_upper = True
                
                st.success(f"✅ Removed {last_key}")
                st.rerun()
            else:
                st.warning("⚠️ No points to undo")
    
    with col3:
        if collected_keystones >= expected_total:
            if st.button("💾 Save Edge Points"):
                # Convert edge coordinates to pixel coordinates for saving
                # Assuming 1920x1080 video resolution (standard)
                video_width, video_height = 1920, 1080
                pixel_points = []
                
                # Convert normalized coordinates back to pixel coordinates
                # Order: Edge points (upper/lower pairs), then meeting points (middle only)
                edge_names = ["B0C1", "B1C2", "B2C3", "E4F4", "B5C6", "B6C7", "B7C8"]
                meeting_names = ["F1F#1G1", "F2F#2G2", "F3F#3G3", "C5C#5D5", "F6F#6G6", "F7F#7G7"]
                
                # Add edge point upper/lower pairs
                for edge in edge_names:
                    upper_key = f"{edge}_upper"
                    lower_key = f"{edge}_lower"
                    if upper_key in st.session_state.keystone_coords and lower_key in st.session_state.keystone_coords:
                        # Upper point
                        upper_coord = st.session_state.keystone_coords[upper_key]
                        pixel_points.append([int(upper_coord[0] * video_width), int(upper_coord[1] * video_height)])
                        # Lower point
                        lower_coord = st.session_state.keystone_coords[lower_key]
                        pixel_points.append([int(lower_coord[0] * video_width), int(lower_coord[1] * video_height)])
                
                # Add meeting point middle points
                for meeting in meeting_names:
                    middle_key = f"{meeting}_middle"
                    if middle_key in st.session_state.keystone_coords:
                        middle_coord = st.session_state.keystone_coords[middle_key]
                        pixel_points.append([int(middle_coord[0] * video_width), int(middle_coord[1] * video_height)])
                
                # Create save data structure
                save_data = {
                    'video_name': selected_option[:-4],
                    'pixel_points': pixel_points,
                    'collection_method': 'edge_based_interpolation',
                    'video_resolution': [video_width, video_height],
                    'edge_coords': st.session_state.keystone_coords,  # Keep normalized coords too
                    'collection_timestamp': __import__('datetime').datetime.now().isoformat()
                }
                
                # Ensure pixel_points directory exists
                pixel_points_dir = os.path.join(script_dir, "pixel_points")
                os.makedirs(pixel_points_dir, exist_ok=True)
                
                # Save to JSON file
                output_file = os.path.join(pixel_points_dir, f"{selected_option[:-4]}_pixel_points.json")
                with open(output_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                st.success(f"✅ Edge points saved to {os.path.basename(output_file)}")
                st.info(f"📁 Saved to: {output_file}")
                st.balloons()
        else:
            st.button("💾 Save Keystone Points", disabled=True, help=f"Collect all {expected_total} points first")

    if value: 
        st.write(f"🖱️ Mouse position: ({value['x1']}, {value['y1']})")
            
def keyboarddistortion():

    st.sidebar.success("Select the menu above.")
    st.write('make mediapipe data from video')
    files = os.listdir(filepath)
    newfiles = []
    for file in files:
        if ".mp4" in file:
            newfiles.append(file)
    newfiles.sort()

    selected_option = st.selectbox(
        "Select video files:",
        newfiles,
    )



    with open(
            os.path.join(script_dir, "keyboardcoordinateinfo.pkl"),
            "rb",
        ) as f:
            keyboardcoordinateinfo = pickle.load(f)
    if "blackratio" not in st.session_state:
        st.session_state["blackratio"] = keyboardcoordinateinfo[selected_option[:-4]][4]
    if "ldistortion" not in st.session_state:
        st.session_state["ldistortion"] = keyboardcoordinateinfo[selected_option[:-4]][5]
    if "rdistortion" not in st.session_state:
        st.session_state["rdistortion"] = keyboardcoordinateinfo[selected_option[:-4]][6]
    if "cdistortion" not in st.session_state:
        st.session_state["cdistortion"] = keyboardcoordinateinfo[selected_option[:-4]][7]

    video = cv2.VideoCapture(os.path.join(filepath , selected_option))
    ret, image = video.read()
    img_np = np.array(image)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    # For keyboarddistortion, we still use the old format for backward compatibility
    keyboard_data = [
        keyboardcoordinateinfo[selected_option[:-4]][0],  # lu
        keyboardcoordinateinfo[selected_option[:-4]][1],  # ru  
        keyboardcoordinateinfo[selected_option[:-4]][2],  # ld
        keyboardcoordinateinfo[selected_option[:-4]][3],  # rd
        st.session_state["blackratio"],
        st.session_state["ldistortion"],
        st.session_state["rdistortion"], 
        st.session_state["cdistortion"]
    ]
    keyboard = generatekeyboard(keyboard_data)
    keyboard_image = cv2.cvtColor(draw_keyboard_on_image(img.numpy_view(), keyboard), cv2.COLOR_BGR2RGB)
    col1, col2=st.columns(2)
    with col1:
        st.image(keyboard_image)
    with col2:
        st.session_state["blackratio"]= st.slider("What is ratio of length of black keys and white keys?", 0.0, 1.0, st.session_state["blackratio"],step=0.05)
        st.session_state["cdistortion"]= st.slider("At the point between E4 and F4, how much the virtual keyboard differs from originial point?", -0.5, 0.5, st.session_state["cdistortion"]*50)/50
        st.session_state["ldistortion"]= st.slider("How distorted is the left side of image?", -0.3, 0.3, st.session_state["ldistortion"]*2000)/2000
        st.session_state["rdistortion"]= st.slider("How distorted is the right side of image?", -0.3, 0.3, st.session_state["rdistortion"]*2000)/2000
    if st.button("Save keyboard"):
        with open(
                os.path.join(script_dir, "keyboardcoordinateinfo.pkl"),
                "wb",
            ) as f:
                keyboardcoordinateinfo[selected_option[:-4]]=[
                    keyboardcoordinateinfo[selected_option[:-4]][0],
                    keyboardcoordinateinfo[selected_option[:-4]][1],
                    keyboardcoordinateinfo[selected_option[:-4]][2],
                    keyboardcoordinateinfo[selected_option[:-4]][3],
                    st.session_state["blackratio"],
                    st.session_state["ldistortion"],
                    st.session_state["rdistortion"],
                    st.session_state["cdistortion"]
                ]
                pickle.dump(keyboardcoordinateinfo, f, pickle.HIGHEST_PROTOCOL) #lu, ru, ld, rd, blackratio, ldstortion, rdistortion, cdistortion
        st.write("Saved keyboard.")
    if st.button("Reload image"):
        st.rerun()


def label():
    initialize_state()
    st.write("# FingeringDetection: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    files = os.listdir(mididirectory)

    newfiles = []
    for file in files:
        if "_singletempo" in str(file):
            newfiles.append(file)
    newfiles.sort()
    st.write("Settings (three dots at upperright side) - use wide mode")
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )


    # Change the tempo to the desired BPM
    
    # Save the modified MIDI file

    if "_singletempo.mid" in selected_option:
        newmidiname = selected_option
    else:
        newmidiname = selected_option[:-4] + "_singletempo.mid"

    if "_singletempo" in selected_option:
        videoname = '_'.join(selected_option.split("_")[:-1]) + ".mp4"
    else:
        videoname = selected_option[:-4] + ".mp4"

    st.write("Selected MIDI:", selected_option)
    frame_rate = get_video_fps(os.path.join(filepath, videoname))

    dirname = (
            os.path.join(
                filepath,
                videoname[:-4]
                + "_"
                + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            )
    )
    
    with open(
            os.path.join(
                dirname,
                "fingerinfo_"
                + videoname
                + "_"
                + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
                + ".pkl"
            ),
            "rb",
    ) as f: 
        fingerinfo = pickle.load(f)

    with open(
        os.path.join(
            dirname,
            "undecidedfingerlist_"
            + videoname
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl"
        ),
            "rb",
    ) as f: 
        undecidedfingerlist = pickle.load(f)
    fingerinfo = decider(
        fingerinfo, undecidedfingerlist, frame_rate, videoname, newmidiname
    )
    st.write(fingerinfo)


def filter_midi_notes(input_midi, target_note_index):
    # Load the MIDI file
    
    mid = mido.MidiFile(input_midi)
    target_note_new_index = -1

    # Create a new MIDI file to store the filtered notes
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        all_notes = []

        # Collect all note_on messages with their corresponding index
        for msg in track:
            if msg.type == 'note_on':
                all_notes.append(msg)

        # Determine the range of notes to keep
        start_index = max(0, target_note_index - 15)
        end_index = min(len(all_notes) - 1, target_note_index + 15)

        # Create a set of indices to keep
        indices_to_keep = range(start_index, end_index + 1)

        active_notes = []

        current_note_index = 0
        new_note_index = 0
        for msg in track:
            
            if msg.type == 'note_on' and msg.velocity > 0:  # Note_off is sometimes denoted as Note_on & Note.velocity=0
                if current_note_index in indices_to_keep:
                    new_track.append(msg)
                    active_notes.append(msg.note)
                    if current_note_index == target_note_index:
                        target_note_new_index = new_note_index
                    new_note_index += 1
                current_note_index += 1
            elif msg.type == 'note_on' and msg.velocity == 0 or msg.type == 'note_off':
                if msg.note in active_notes:
                    if msg.type =='note_on' and msg.velocity == 0:
                        new_msg=mido.Message('note_off', channel=msg.channel, note=msg.note, velocity=0, time=msg.time)
                        new_track.append(new_msg)
                    elif msg.type == 'note_off': new_track.append(msg)
                    active_notes.remove(msg.note)
            else:
                if current_note_index in indices_to_keep:
                    new_track.append(msg)
        for msg in new_track:
            print(msg)
        # Ensure all active notes are properly closed
        if not active_notes:
            assert("Some notes are not closed")

        new_mid.tracks.append(new_track)
    if target_note_new_index == -1:
        assert("Target note not found")
    new_mid.save(f"{mididirectory}trimmed{target_note_index}.mid")
    
    return target_note_new_index

def annotate():
    initialize_state()

    st.write("# FingeringDetection: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if "_singletempo" in file:
            newfiles.append(file)
    newfiles.sort()
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )
    if "_singletempo.mid" in selected_option:
        newmidiname = selected_option
    else:
        newmidiname = selected_option[:-4] + "_singletempo.mid"

    if "_singletempo" in selected_option:
        videoname = '_'.join(selected_option.split("_")[:-1]) + ".mp4"
    else:
        videoname = selected_option[:-4] + ".mp4"
    st.write("Selected MIDI:", selected_option)
    
    frame_rate = get_video_fps(os.path.join(filepath, videoname))
    tokeninfolist=miditotoken(newmidiname[:-4], frame_rate, "simplified")

    def button_click():
        if st.session_state.index < 150:
            st.session_state.history.append(st.session_state.index)
            st.session_state.responses+=user_input.split(',')   #Finger, token numebr
            st.session_state.index += len(user_input.split(','))
        st.rerun()

    def undo():
        if st.session_state.history:
            st.session_state.index -=1
            st.session_state.responses.pop()
        st.rerun()

    def reset():
        st.session_state.index = 0
        st.session_state.history = []
        st.session_state.responses = []
        st.rerun()

    def complete():
        st.session_state.responses.append("Complete")
        return st.session_state.responses
    
    def notecount(filename):
        # Open your MIDI file
        midi_file = mido.MidiFile(filename)

        # Initialize a variable to count the notes
        note_count = 0

        # Iterate through all the messages in the MIDI file
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'note_on':  # 'note_on' message indicates a note is being pressed
                    note_count += 1

        # Print the total number of notes
        return note_count

    if st.session_state.index < min(150, notecount(os.path.join(mididirectory,newmidiname))):
        st.write(
                f"#### Choose the actual finger which pressed {tokeninfolist[st.session_state.index][1]}({pitch_list[tokeninfolist[st.session_state.index][1]]}) at frame {tokeninfolist[st.session_state.index][0]} or time {str(int(tokeninfolist[st.session_state.index][0]/frame_rate//60)).zfill(2)}:{str(math.floor(tokeninfolist[st.session_state.index][0]/frame_rate%60)).zfill(2)}:"
                + format(
                    math.floor(tokeninfolist[st.session_state.index][0] / frame_rate % 60 * 1000) / 1000
                    - math.floor(tokeninfolist[st.session_state.index][0] / frame_rate % 60),
                    ".2f",
                    )[2:]
                + " : "
                )
    
        col1, col2 = st.columns(2)
        starttime = math.floor(tokeninfolist[st.session_state.index][0] / frame_rate)
        print(starttime)
        with col1:
            print("preparing video")
            video_file = open(videodirectory + videoname, "rb")
            st.video(video_file, start_time=starttime)
            print("completed preparing video")
        with col2:
            print("preparing midi")
            midi_file_path = mididirectory + newmidiname
            rednoteindex=filter_midi_notes(midi_file_path, st.session_state.index)
            mid=stroll.MidiFile(f"{mididirectory}trimmed{st.session_state.index}.mid")
            mid.draw_roll(rednoteidx=rednoteindex)
            os.remove(f"{mididirectory}trimmed{st.session_state.index}.mid")
            print("completed preparing midi")
        st.write(f"Decided {st.session_state.index} of 150")

        upcominglist=[]
        for i in range(10):
            if st.session_state.index+i<min(150, notecount(os.path.join(mididirectory,newmidiname))):
                upcominglist.append(pitch_list[tokeninfolist[st.session_state.index+i][1]])
        st.write(f"Present note and upcoming notes:{upcominglist}")

        user_input=st.text_input("Enter finger number from 1 to 10, comma between multiple fingers with no blank space")
        if st.button('next'):
            button_click()
    else:
        st.write("Completed all steps")

    if st.session_state.index >= min(150, notecount(os.path.join(mididirectory,newmidiname))):
        if st.button("Complete"):
            responses = complete()
            st.write(f"Responses: {responses}")
            file=open(os.path.join(script_dir, f'{newmidiname[:-16]}.txt'), 'a')   #_singletempo.mid
            for response in responses:
                w=file.write(f'{response}, ')
            file.close()

    if st.session_state.history:
        if st.button("Undo"):
            undo()

    if st.session_state.index >= 150:
        if st.button("Reset"):
            reset()

    st.write(f"Current Index: {st.session_state.index}")
    st.write(f"Responses: {st.session_state.responses}")



page_names_to_funcs = {
    "Intro": intro,
    "Delete smart tempo": preprocess,
    "Generate mediapipe data" : videodata, 
    "Pre-finger labeling": prefinger,   
    "Label": label,
    "Groundtruth annotation": annotate,
    "Keyboard keystone collection": keyboardcoordinate,
}

demo_name = st.sidebar.selectbox("**MENU** 🍽️", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

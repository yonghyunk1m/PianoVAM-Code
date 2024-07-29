# ASDF: Automated System for Detecting Fingering
import streamlit as st
import fortepyan as ff
import sys, os
from midicomparison import *
from main import (
    filepath,
    min_hand_detection_confidence,
    min_hand_presence_confidence,
    min_tracking_confidence,
    keyboard,
    datagenerate,
)
import mido
import pickle
import pretty_midi
import cv2
import stroll    # https://github.com/exeex/midi-visualization with some modifications in order to use in streamlit environment

miditest300=[
    9, 10, 7, 5, 1, 2, 8, 6, 7, 5, 
    8, 2, 7, 7, 6, 10, 1, 5, 9, 9,
    10, 5, 1, 6, 2, 7, 7, 8, 5, 1,
    2, 7, 2, 5, 7, 5, 6, 10, 8, 5,
    7, 6, 1, 2, 9, 8, 1, 5, 6, 8,

    7, 7, 8, 6, 1, 2, 6, 7, 5, 1, 
    2, 8, 10, 1, 8, 5, 6, 8, 1, 2, 
    8, 6, 5, 1, 2, 10, 1, 10, 1, 9, 
    2, 8, 6, 5, 2, 1, 5, 1, 2, 8,
    6, 9, 5, 1, 8, 7, 6, 1, 9, 5,

    8, 9, 6, 5, 10, 1, 8, 1, 5, 8, 
    6, 7, 5, 1, 6, 8, 7, 1, 6, 8, 
    7, 1, 7, 6, 5, 1, 8, 1, 6, 1,
    10, 5, 6, 6, 7, 10, 1, 5, 5, 7, 
    9, 6, 10, 1, 2, 5, 5, 5, 1, 5,

    1, 9, 7, 6, 10, 2, 9, 1, 7, 10, 
    6, 2, 6, 8, 10, 5, 1, 2, 7, 6,
    8, 10, 2, 1, 5, 1, 5, 2, 1, 5,
    6, 1, 5, 7, 5, 1, 6, 9, 1, 10, 
    1, 10, 6, 5, 7, 1, 10, 2, 1, 5,

    6, 2, 5, 7, 6, 1, 2, 1, 5, 8,
    6, 5, 1, 10, 2, 1, 10, 2, 6, 8,
    10, 1, 5, 7, 2, 7, 6, 5, 1, 2,
    5, 1, 7, 6, 2, 6, 10, 8, 5, 1, 
    2, 6, 5, 1, 9, 7, 6, 5, 1, 6, 

    9, 7, 2, 7, 10, 9, 5, 6, 1, 1,
    5, 10, 7, 6, 9, 5, 1, 7, 9, 10,
    5, 1, 6, 5, 1, 10, 6, 2, 7, 10,
    6, 5, 1, 10, 6, 5, 1, 6, 10, 10,
    6, 2, 5, 10, 6, 6, 1, 5, 7, 6
]


st.set_page_config(layout="wide")

mididirectory = "./midiconvert/"
videodirectory = "./videocapture/"

def delete_smart_tempo(midiname):
    if not "_singletempo.mid" in midiname:
        midi_data = pretty_midi.PrettyMIDI(midiname, initial_tempo=120)
        midi_data.write(midiname[:-4] + "_singletempo.mid")


def intro():
    st.write("# ASDF: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    st.markdown(
        """
        **ASDF : Automated System for Detecting Fingering** is a semi-automatic assistant to label fingering from video. 
        The algorithm only asks confusing fingering for us, so you can either answer the correct fingering from the video or just skip to answer if it is hard to determine the correct fingering even for us.)\\
        
        #### Prerequisites
        - Top-view video
        - Performance MIDI which is recorded from above video
        
        #### Data format for PianoVAM
        - Audio: 16kHz wav format
        - MIDI: mid and Logic Project file
        - Video: Top-View 30fps 720*1280 Video (Full 88 keyboard must be shown.)

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
    initialize_state()

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
        print(buttons[st.session_state.index])
        strval=''
        for button_name in buttons[st.session_state.index]:
            str_button_name = ''
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

    # Complete 버튼 표시
    if st.session_state.index >= len(buttons):
        if st.button("Complete"):
            responses = complete()
            st.write(f"Responses: {responses}")

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


def sthanddecider(tokenlist, keyhandlist, fps, videoname, newmidiname):
    if 'index' not in st.session_state:
        st.session_state.index = 0

    tokenhandlist = []
    pressedfingerlist = [None for k in range(len(tokenlist))]
    undecidedtokeninfolist = []

    #for evaluation
    correct=0
    undecided=0
    noinfo=0
    for i in range(len(tokenlist)):
        # tokenlist[i].pop(0)  #Total Start Position (frame)
        # tokenlist[i].pop(1)  # End position
        # tokenlist[i][0]=pitch_list[tokenlist[i][0]+12]
        totalframe=tokenlist[i][2]-tokenlist[i][0]
        tokenlist[i].pop(2)  # End position
        tokenlist[i][1] = pitch_list[tokenlist[i][1]]
        lefthandcounter = 0
        righthandcounter = 0
        noinfocounter = 0
        lhindex = []
        rhindex = []
        fingerindex = [[],[],[],[],[],[],[],[],[],[],0]  # 10개의 손가락과 Noinfo counter
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
        # if lefthandcounter>0 and righthandcounter>0 or noinfocounter>0:
        # print(f"Tokennumber:{i},   Tokenpitch={pitch_list[tokenlist[i][1]]},    lefthandcounter={lhindex},     righthandcounter={rhindex}, noinfocounter ={noinfocounter}")
        print(
            f"Tokennumber:{i},   Tokenpitch={tokenlist[i][1]},    lefthandcounter={len(lhindex)},     righthandcounter={len(rhindex)}, noinfocounter ={noinfocounter}, fingercount={fingerscore}, fingernoinfo={fingerindex[10]}"
        )
        pressedfingers = []
        totalfingercount = 0
        c = 0
        for j in range(10):
            if fingerscore[j] != 0:
                pressedfingers.append([j + 1,fingerscore[j]])
                totalfingercount += fingerscore[j]

        while c < len(pressedfingers):
            finger=pressedfingers[c]
            if finger[1]/totalframe <0.4: # 어떤 손가락이 총 frame의 50프로를 넘지 않으면 후보에서 삭제.
                pressedfingers.pop(c)
                c -= 1
            c += 1
        if c > 1:   # 후보 손가락이 두개 이상일때
            undecidedtokeninfolist.append([i,tokenlist[i],pressedfingers,totalframe])  #Token number, token info, finger candidate, frame count of the token
            undecided += 1
        elif c == 1:
            pressedfingerlist[i] = pressedfingers[0][0]
            if i <= 299:
                if pressedfingerlist[i] == miditest300[i]:
                    correct += 1
                else: print(f"tokennumber: {i}, gt: {miditest300[i]}, pred:{pressedfingerlist[i]}")
        elif c == 0:
            pressedfingerlist[i] = "Noinfo"
            noinfo += 1
    
        
        if i==299:
            st.write(f'Accuracy: {correct/(300-noinfo-undecided)}, total noinfo: {noinfo}, total undecided: {undecided}')

    decision=button_input(undecidedtokeninfolist, fps, videoname, newmidiname)
    if decision:   #decidion=None 인 경우 제외
        if len(decision) == len(undecidedtokeninfolist)+1: # +1: complete
            for j in range(len(pressedfingerlist)):
                if pressedfingerlist[j] == "Noinfo" and decision[0][1]==j:
                    pressedfingerlist[j] = decision[0]
                    decision.pop(0)

    return tokenhandlist,pressedfingerlist  # Output: [Pitch(0~95(가상건반은 C0~ G8까지 있음)), token number, hand]

def videodata():
    st.sidebar.success("Select the menu above.")
    st.write('make mediapipe data from video')
    files = os.listdir(filepath)
    newfiles = []
    for file in files:
        if ".mp4" in file:
            newfiles.append(file)

    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )
    if st.button('Generate mediapipe data'):
        datagenerate(selected_option)
        st.write(f'Generated data of {filepath + selected_option}')

    
    

def preprocess():
    st.write('delete smart tempo which is automatically recorded by logic')
    mididirectory = "./midiconvert/"
    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if not "_" in str(file):
            newfiles.append(file)
    st.write("Settings (three dots at upperright side) - use wide mode")
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )
    if st.button('Delete smart tempo'):
        delete_smart_tempo(mididirectory + selected_option)
        st.write(f'Changed {mididirectory + selected_option}.')

def prefinger():
    st.write("# ASDF: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    
    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if "_singletempo" in str(file):
            newfiles.append(file)
    st.write("Settings (three dots at upperright side) - use wide mode")
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )

    # Change the tempo to the desired BPM
    
    # Save the modified MIDI file

    newmidiname = selected_option
    videoname = selected_option.split("_")[0] + ".mp4"

    st.write("Selected MIDI:", selected_option)

    if st.button("Precorrespond fingering"):
        st.write(
            'fingering pre-labeling started'
        )
        video = cv2.VideoCapture(filepath + videoname)
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        dirname = (
            filepath
            + videoname[:-4]
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
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

        handfingerpositionlist = []
        for handsinfo in handlist:
            handfingerposition = handpositiondetector(
                handsinfo, floatingframes, keyboard
            )
            handfingerpositionlist.append(handfingerposition)
        tokenlist = miditotoken(newmidiname[:-4], frame_rate, "simplified")
        prefingercorrespond =handfingercorresponder(
                tokentoframeinfo(tokenlist, frame_count), handfingerpositionlist, keyboard, tokenlist
            )
        
        with open(
            dirname
            + "/prefingercorrespond_"
            + videoname
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(prefingercorrespond, f)

        st.write("Prefinger info saved")


            
    


def label():
    st.write("# ASDF: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    
    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if not "_" in files:
            newfiles.append(file)
    
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

    if "_" in selected_option:
        videoname = selected_option.split("_")[0] + ".mp4"
    else:
        videoname = selected_option[:-4] + ".mp4"

    st.write("Selected MIDI:", selected_option)

    video = cv2.VideoCapture(filepath + videoname)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    tokenlist = miditotoken(newmidiname[:-4], frame_rate, "simplified")

    dirname = (
            filepath
            + videoname[:-4]
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        )
    
    with open(
            dirname
            + "/prefingercorrespond_"
            + videoname
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl",
            "rb",
    ) as f: 
        prefingercorrespond = pickle.load(f)

    handinfo, fingerinfo = sthanddecider(
        tokenlist,
        prefingercorrespond,
        frame_rate,
        videoname,
        newmidiname,
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
            
            if msg.type == 'note_on' and msg.velocity > 0:  # Note_off가 Note_on + Note.velocity=0으로 표시되는 경우가 있음. ???? ㅋㅋ...
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
     

page_names_to_funcs = {
    "Intro": intro,
    "Delete smart tempo": preprocess,
    "Generate mediapipe data" : videodata, 
    "Pre-finger labeling": prefinger,   
    "Label": label,
}

demo_name = st.sidebar.selectbox("**MENU** 🍽️", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

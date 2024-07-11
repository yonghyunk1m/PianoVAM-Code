# ASDF: Automated System for Detecting Fingering
import streamlit as st
import fortepyan as ff
import sys, os
from streamlit_pianoroll import from_fortepyan
from midicomparison import *
from main import (
    filepath,
    min_hand_detection_confidence,
    min_hand_presence_confidence,
    min_tracking_confidence,
    keyboard,
)
import mido
import pickle
import pretty_midi
import cv2

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

def initialize_state():
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'responses' not in st.session_state:
        st.session_state.responses = []

# 버튼 입력 함수
def button_input(undecidedtokeninfolist, fps, videoname, newmidiname):
    buttons=[]
    for tokeninfo in undecidedtokeninfolist:
        buttons.append(tokeninfo[2])
    initialize_state()

    # 버튼 클릭 핸들러
    def button_click(button_name):
        if st.session_state.index < len(buttons):
            st.session_state.history.append(st.session_state.index)
            st.session_state.responses.append(button_name[0])
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
    st.write(
            f"#### Choose the actual finger which pressed {undecidedtokeninfolist[st.session_state.index][1][1]} at frame {undecidedtokeninfolist[st.session_state.index][1][0]} or time {str(int(undecidedtokeninfolist[st.session_state.index][1][0]/fps//60)).zfill(2)}:{str(math.floor(undecidedtokeninfolist[st.session_state.index][1][0]/fps%60)).zfill(2)}:"
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
        video_file = open(videodirectory + videoname, "rb")
        st.video(video_file, start_time=starttime)

    with col2:
        midi_file_path = mididirectory + newmidiname
        rednoteindex=filter_midi_notes(midi_file_path, undecidedtokeninfolist[st.session_state.index][0])

        piece = ff.MidiPiece.from_file(
            f"{mididirectory}trimmed{undecidedtokeninfolist[st.session_state.index][0]}.mid"
        )  # Midifile의 apply_sustain을 False로 함.
        from_fortepyan(piece=piece, show_bird_view=False)

        os.remove(f"{mididirectory}trimmed{undecidedtokeninfolist[st.session_state.index][0]}.mid")

    if st.session_state.index < len(buttons):
        st.write(f"Decided {st.session_state.index + 1} of {len(buttons)} undecided fingerings")

        # 버튼 표시
        print(buttons[st.session_state.index])
        for button_name in buttons[st.session_state.index]:
            str_button_name = ''
            for idx, val in enumerate(button_name):
                str_button_name += str(val) + (' frames' if idx == len(button_name) -1 else ': ')
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

    for i in range(len(tokenlist)):
        # tokenlist[i].pop(0)  #Total Start Position (frame)
        # tokenlist[i].pop(1)  # End position
        # tokenlist[i][0]=pitch_list[tokenlist[i][0]+12]
        tokenlist[i].pop(2)  # End position
        tokenlist[i][1] = pitch_list[tokenlist[i][1] + 12]
        lefthandcounter = 0
        righthandcounter = 0
        noinfocounter = 0
        lhindex = []
        rhindex = []
        fingerindex = [[],[],[],[],[],[],[],[],[],[],0]  # 10개의 손가락과 Noinfo counter
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
                        if keyhandinfo[3] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                            fingerindex[10] += 1
                    elif keyhandinfo[2] == "Right":
                        righthandcounter += 1
                        rhindex.append(j)
                        for k in range(1, 11):
                            if keyhandinfo[3][k] != 0:
                                fingerindex[k - 1].append(j)
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
        fingercount = [len(fingerindex[i]) for i in range(0, 10)]
        # if lefthandcounter>0 and righthandcounter>0 or noinfocounter>0:
        # print(f"Tokennumber:{i},   Tokenpitch={pitch_list[tokenlist[i][1]]},    lefthandcounter={lhindex},     righthandcounter={rhindex}, noinfocounter ={noinfocounter}")
        print(
            f"Tokennumber:{i},   Tokenpitch={tokenlist[i][1]},    lefthandcounter={len(lhindex)},     righthandcounter={len(rhindex)}, noinfocounter ={noinfocounter}, fingercount={fingercount}, fingernoinfo={fingerindex[10]}"
        )
        pressedfingers = []
        
        totalfingercount = 0
        c = 0
        for j in range(10):
            if fingercount[j] != 0:
                pressedfingers.append([j + 1,fingercount[j]])
                totalfingercount += fingercount[j]
        if len(pressedfingers) == 0:
            pressedfingerlist[i] = "Noinfo"
        if len(pressedfingers) == 1:

            pressedfingerlist[i] = pressedfingers[0][0]
        if len(pressedfingers) >= 2:
            for finger in pressedfingers:
                if finger[1]/totalfingercount >0.75: # 어떤 손가락이 전체의 75프로를 넘으면 그냥 그 손가락으로 간주.
                    pressedfingerlist[i]=finger[0]
                    c = 1
            if c ==0:
                undecidedtokeninfolist.append([i,tokenlist[i],pressedfingers])  #Token number, token info, finger candidate

    decision=button_input(undecidedtokeninfolist, fps, videoname, newmidiname)

    return tokenhandlist,pressedfingerlist  # Output: [Pitch(0~95(가상건반은 C0~ G8까지 있음)), token number, hand]

def preprocess():
    st.write('delete smart tempo which is automatically recorded by logic')
    mididirectory = "./midiconvert/"
    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if not "_" in files:
            newfiles.append(file)
    st.write("Settings (three dots at upperright side) - use wide mode")
    selected_option = st.selectbox(
        "Select groundtruth MIDI files and **wait few seconds until the midi file is loaded** :",
        newfiles,
    )
    if st.button('Delete smart tempo'):
        delete_smart_tempo(mididirectory + selected_option)
        st.write(f'Changed {mididirectory + selected_option}.')

def label():
    st.write("# ASDF: Automated System for Detecting Fingering")
    st.sidebar.success("Select the menu above.")

    
    files = os.listdir(mididirectory)
    newfiles = []
    for file in files:
        if not "_" in files:
            newfiles.append(file)
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

    if "_" in selected_option:
        videoname = selected_option.split("_")[0] + ".mp4"
    else:
        videoname = selected_option[:-4] + ".mp4"

    st.write("Selected MIDI:", selected_option)

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
    handinfo, fingerinfo = sthanddecider(
        tokenlist,
        handfingercorresponder(
            tokentoframeinfo(tokenlist, frame_count), handfingerpositionlist
        ),
        frame_rate,
        videoname,
        newmidiname,
    )

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
        start_index = max(0, target_note_index - 20)
        end_index = min(len(all_notes) - 1, target_note_index + 20)

        # Create a set of indices to keep
        indices_to_keep = range(start_index,end_index + 1)

        # Add messages to the new track while maintaining timing
        current_note_index = 0
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                if current_note_index in indices_to_keep:
                    new_track.append(msg)
                    if current_note_index == target_note_index:
                        target_note_new_index = len(new_track) - 1
                if msg.type == 'note_on':
                    current_note_index += 1
            else:
                new_track.append(msg)

        new_mid.tracks.append(new_track)

    new_mid.save(f"{mididirectory}trimmed{target_note_index}.mid")
    return(target_note_new_index)
     

page_names_to_funcs = {
    "Intro": intro,
    "Preprocess": preprocess,    
    "Label": label,
}

demo_name = st.sidebar.selectbox("**MENU** 🍽️", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

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
import stroll    # https://github.com/exeex/midi-visualization with some modifications in order to use in streamlit environment
import dill
import subprocess
import json
from tqdm.auto import tqdm
from stqdm import stqdm

st.set_page_config(layout="wide")

mididirectory = "./FingeringDetection/midiconvert/"
videodirectory = "./FingeringDetection/videocapture/"

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
                        fingering_textfile.write(f"Tokennumber={i}, Finger={fingering_textlist[i].split(",")[1]}, \n")
                        


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
            "./FingeringDetection/"
            + "keyboardcoordinateinfo"
            + ".pkl",
            "rb",
        ) as f:
            keyboardcoordinateinfo = pickle.load(f)
        keyboard = generatekeyboard(
            lu=keyboardcoordinateinfo[videoname[:-4]][0],
            ru=keyboardcoordinateinfo[videoname[:-4]][1],
            ld=keyboardcoordinateinfo[videoname[:-4]][2],
            rd=keyboardcoordinateinfo[videoname[:-4]][3],
            blackratio=keyboardcoordinateinfo[videoname[:-4]][4],
            ldistortion=keyboardcoordinateinfo[videoname[:-4]][5],
            rdistortion=keyboardcoordinateinfo[videoname[:-4]][6],
            cdistortion=keyboardcoordinateinfo[videoname[:-4]][7],
        )
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

    video = cv2.VideoCapture(os.path.join(filepath, selected_option))
    ret, image = video.read()
    cv2.imwrite("tmp.jpg", image)
    value = streamlit_image_coordinates(
        "tmp.jpg",
        key="local4",
        use_column_width="always",
        click_and_drag=True,
    )
    st.write("Click the image of leftupper, leftunder, rightupper, and rightunder side of the keyboard and click the button.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("leftupper"):
            lu=[value["x1"]/value['width'],value['y1']/value['height']]
            with open(
                    "./FingeringDetection/lu.pkl",
                    "wb",
                ) as f:
                    pickle.dump(lu, f)
        if st.button("leftunder"):
            ld=[value["x1"]/value['width'],value['y1']/value['height']]
            with open(
                    "./FingeringDetection/ld.pkl",
                    "wb",
                ) as f:
                    pickle.dump(ld, f)
    with col2:
        if st.button("rightupper"):
            ru=[value["x1"]/value['width'],value['y1']/value['height']]
            with open(
                    "./FingeringDetection/ru.pkl",
                    "wb",
                ) as f:
                    pickle.dump(ru, f)
        if st.button("rightunder"):
            rd=[value["x1"]/value['width'],value['y1']/value['height']]
            with open(
                    "./FingeringDetection/rd.pkl",
                    "wb",
                ) as f:
                    pickle.dump(rd, f)
    if st.button("Complete"):
        with open(
                "./FingeringDetection/lu.pkl",
                "rb",
            ) as f:
            lu=pickle.load(f)
        with open(
                "./FingeringDetection/ld.pkl",
                "rb",
            ) as f:
            ld=pickle.load(f)
        with open(
                "./FingeringDetection/ru.pkl",
                "rb",
            ) as f:
            ru=pickle.load(f)
        with open(
                "./FingeringDetection/rd.pkl",
                "rb",
            ) as f:
            rd=pickle.load(f)
        if not "keyboardcoordinateinfo.pkl" in os.listdir("./FingeringDetection"):
            with open(
                    "./FingeringDetection/"
                    "keyboardcoordinateinfo"
                    + ".pkl",
                    "wb",
                ) as f:
                    keyboardcoordinateinfo={"Status":"Generated"}
                    pickle.dump(keyboardcoordinateinfo, f)
        with open(
                "./FingeringDetection/"
                "keyboardcoordinateinfo"
                + ".pkl",
                "rb",
            ) as f:
                keyboardcoordinateinfo = pickle.load(f)
                keyboardcoordinateinfo[selected_option[:-4]]=[lu,ru,ld,rd,0.5,0.0,0.0,0.0]
                print(keyboardcoordinateinfo)
        with open(
                "./FingeringDetection/"
                "keyboardcoordinateinfo"
                + ".pkl",
                "wb",
            ) as f:
                print(keyboardcoordinateinfo)
                pickle.dump(keyboardcoordinateinfo, f, pickle.HIGHEST_PROTOCOL)

    if value: st.write(value["x1"], value["y1"])
            
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
            "./FingeringDetection/"
            + "keyboardcoordinateinfo"
            + ".pkl",
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
    keyboard=generatekeyboard(
        lu = keyboardcoordinateinfo[selected_option[:-4]][0],
        ru = keyboardcoordinateinfo[selected_option[:-4]][1],
        ld = keyboardcoordinateinfo[selected_option[:-4]][2],
        rd = keyboardcoordinateinfo[selected_option[:-4]][3],
        blackratio = st.session_state["blackratio"],
        ldistortion = st.session_state["ldistortion"],
        rdistortion = st.session_state["rdistortion"],
        cdistortion = st.session_state["cdistortion"],
        
    )
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
                "./FingeringDetection/"
                "keyboardcoordinateinfo"
                + ".pkl",
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
            file=open(f'./FingeringDetection/{newmidiname[:-16]}.txt', 'a')   #_singletempo.mid
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
    "Keyboard detection": keyboardcoordinate,
    "Keyboard distortion": keyboarddistortion
}

demo_name = st.sidebar.selectbox("**MENU** 🍽️", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

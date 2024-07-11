import cv2
import numpy as np
import os
import time

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from floatinghands import *
from midicomparison import *
import pickle
from tqdm.auto import tqdm
# Note: file order is main>evaluate>midicomparison>floatinghands

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
min_hand_detection_confidence = 0.85
min_hand_presence_confidence = 0.8
min_tracking_confidence = 0.5
VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=min_hand_detection_confidence,
    min_hand_presence_confidence=min_hand_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
)
detector = vision.HandLandmarker.create_from_options(options)

distortion = 0.001
keyboard = generatekeyboard(
    lu=[0.0614, 1 - 0.6750],
    ru=[0.9593, 1 - 0.6824],
    ld=[0.0614, 1 - 0.4796],
    rd=[0.9593, 1 - 0.4796],
    blackratio=0.65,
    distortion=distortion,
)  # miditest
# keyboard=generatekeyboard(lu=[0.0651,0.4287],ru=[0.9380, 0.4148],ld=[0.0661,0.6],rd=[0.9380,0.5916], blackratio=0.65, distortion=distortion)  #Handcrossing3
# keyboard=generatekeyboard(lu=[0.0557,1-0.359],ru=[1-193/1920, 1-385/1080],ld=[0.0505,1-0.207],rd=[1-185/1920,1-218/1080], blackratio=0.5, distortion=distortion) #sonatinetest
# keyboard=generatekeyboard(lu=[100/1920,1-538/1080],ru=[1-159/1920, 1-538/1080],ld=[100/1920,1-361/1080],rd=[1-154/1920,1-358/1080], blackratio=0.56, distortion=distortion) #BWV846
# keyboard=generatekeyboard(lu=[0,516/1080],ru=[1, 516/1080],ld=[0,755/1080],rd=[1,755/1080], blackratio=0.66, distortion=distortion) #Rousseau
filepath = "./videocapture/"
videoname = "miditestcropped"
midiname = videoname
video = cv2.VideoCapture(filepath + videoname + ".mp4")
dirname = (
    filepath
    + videoname
    + "_"
    + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = video.get(cv2.CAP_PROP_FPS)


def datagenerate():
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for file in os.scandir(dirname):
        os.remove(file.path)

    count = 0
    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)
    pbar = tqdm(total=frame_count)
    while video.isOpened():
        if count < frame_count:
            ret, image = video.read()
            cv2.imwrite(dirname + "/frame%d.jpg" % count, image)
            count += 1
            pbar.update(1)
        else:
            break
    pbar.close()
    video.release()

    # STEP 3: Load the input image.
    file_list = os.listdir(dirname)
    for files in file_list:
        if files[-4:] != ".jpg":
            file_list.remove(files)
    file_count = len(file_list)

    nohandframelist = []
    handlist = []
    handtypelist = []
    tempimglist = []

    pbar2 = tqdm(total=file_count)
    for frame in range(file_count):
        img = Image.open(dirname + "/frame%d.jpg" % frame)
        img_np = np.array(img)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        time = frame * int(1000 / frame_rate)
        detection_result = detector.detect_for_video(image, timestamp_ms=time)
        tempimglist.append([image, detection_result])
        if len(detection_result.handedness) == 0:
            nohandframelist.append(frame)
        handsinfo = [
            handclass(
                handtype=detection_result.handedness[j][0].category_name,
                handlandmark=detection_result.hand_landmarks[j],
                handframe=frame,
            )
            for j in range(len(detection_result.handedness))
        ]
        handlist.append(handsinfo)
        handtypelist.append([hand.handtype for hand in handsinfo])
        pbar2.update(1)
    faultyframe = faultyframes(handlist)
    pbar2.close()
    print("Calculating faulty frames...")
    floatingframes = detectfloatingframes(handlist, frame_count, faultyframe)
    with open(
        dirname
        + "/floatingframes_"
        + videoname
        + "_"
        + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(floatingframes, f)

    with open(
        dirname
        + "/handlist_"
        + videoname
        + "_"
        + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(handlist, f)

    print("Data generated and saved")
    print(
        f"faulty frames:{len(faultyframe)}, nohand frames:{len(nohandframelist)} floating hands:{len(floatingframes)}/{len(handlist)}"
    )
    for frame in range(file_count):
        image, detection_result = tempimglist[frame]
        new_file_name = f"frame_annotated{frame}.jpg"
        if frame in nohandframelist:
            new_file_name = new_file_name[:-4] + "_NoDetect" + ".jpg"
        annotated_image = draw_landmarks_and_floatedness_on_image(
            image.numpy_view(), detection_result, frame, floatingframes
        )
        cv2.imwrite(
            dirname + "/" + new_file_name,
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        )
    image, detection_result = tempimglist[max(0, file_count - 100)]
    keyboard_image = draw_keyboard_on_image(image.numpy_view(), keyboard)
    cv2.imwrite(
        dirname + "/" + "keyboard.jpg", cv2.cvtColor(keyboard_image, cv2.COLOR_BGR2RGB)
    )
    print("Image generated and saved")


start = time.time()

##############################
# datagenerate() ##############
##############################


datatime = time.time()
print(f"Data generation time: {datatime-start: .5f} sec")


def handdetokenizer(originaltokenlist, tokenhandinfo):
    noinfocounter = 0
    outputtoken = []
    for i in range(len(originaltokenlist)):
        if tokenhandinfo[i][-2] == "Right":
            for j in range(len(originaltokenlist[i])):
                if "Program" in originaltokenlist[i][j]:
                    originaltokenlist[i][j] = "Program_0"
        elif tokenhandinfo[i][-2] == "Left":
            for j in range(len(originaltokenlist[i])):
                if "Program" in originaltokenlist[i][j]:
                    originaltokenlist[i][j] = "Program_1"
        elif tokenhandinfo[i][-2] == "noinfo":
            for j in range(len(originaltokenlist[i])):
                if "Program" in originaltokenlist[i][j]:
                    originaltokenlist[i][j] = "Program_2"
                    noinfocounter += 1
    for token in originaltokenlist:
        for attribute in token:
            outputtoken.append(attribute)
    print(f"noinfocount: {noinfocounter}")
    return outputtoken


def midiprocess():
    with open(
        dirname
        + "/floatingframes_"
        + videoname
        + "_"
        + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + ".pkl",
        "rb",
    ) as f:
        floatingframes = pickle.load(f)

    with open(
        dirname
        + "/handlist_"
        + videoname
        + "_"
        + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + ".pkl",
        "rb",
    ) as f:
        handlist = pickle.load(f)

    handfingerpositionlist = []
    for handsinfo in handlist:
        handfingerposition = handpositiondetector(handsinfo, floatingframes, keyboard)
        handfingerpositionlist.append(handfingerposition)
    tokenlist = miditotoken(midiname, frame_rate, "simplified")
    originaltokenlist = miditotoken(midiname, frame_rate, "highres")

    tokenhandfingerinfo = handdecider(
        tokenlist,
        handfingercorresponder(
            tokentoframeinfo(tokenlist, frame_count), handfingerpositionlist
        ),
        frame_rate,
    )
    # Detokenizer

    tokenizer, beatres, tokens = generatetokenizer(midiname, frame_rate, highres=True)
    converted_back_midi = tokenizer(
        handdetokenizer(originaltokenlist, tokenhandfingerinfo)
    )
    converted_back_midi.dump_midi(
        dirname
        + "/"
        + f"{midiname}{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + "converted_back_midi.mid"
    )
    print("midi converted")


#########################
# midiprocess() ##########
#########################

miditime = time.time()
print(f"Midi generation time: {miditime-datatime: .5f} sec")

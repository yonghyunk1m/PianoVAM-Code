import cv2
import numpy as np
import os
from psutil import Process
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from floatinghands_torch_pure import *
from midicomparison import *
import pickle
from tqdm.auto import tqdm
from stqdm import stqdm
import time as timemodule
# Note: file order is main>evaluate>midicomparison>floatinghands

# STEP 2: Create an HandLandmarker object with GPU support.
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def check_gpu_support():
    """GPU 지원 여부를 확인합니다."""
    try:
        # OpenCV GPU 지원 확인 (안전하게 처리)
        try:
            has_opencv_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except AttributeError:
            # OpenCV가 CUDA 지원 없이 컴파일된 경우
            has_opencv_gpu = False
            print("⚠️  OpenCV가 CUDA 지원 없이 컴파일되었습니다.")
        
        # PyTorch GPU 지원 확인
        try:
            import torch
            has_torch_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
            if has_torch_gpu:
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            else:
                gpu_count = 0
                gpu_names = []
        except ImportError:
            has_torch_gpu = False
            gpu_count = 0
            gpu_names = []
        
        # TensorFlow GPU 지원 확인 (선택적)
        try:
            import tensorflow as tf
            has_tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            has_tf_gpu = False
        
        print(f"🔍 GPU 지원 상태:")
        print(f"   OpenCV CUDA: {'✅ 사용 가능' if has_opencv_gpu else '❌ 사용 불가'}")
        print(f"   PyTorch GPU: {'✅ 사용 가능' if has_torch_gpu else '❌ 사용 불가'}")
        if has_torch_gpu:
            print(f"   🎮 GPU 장치: {gpu_count}개")
            for i, name in enumerate(gpu_names):
                print(f"      GPU {i}: {name}")
        print(f"   TensorFlow GPU: {'✅ 사용 가능' if has_tf_gpu else '❌ 사용 불가'}")
        
        # PyTorch GPU가 있으면 충분함 (main processing은 PyTorch로 수행)
        if has_torch_gpu:
            print("✅ PyTorch GPU 가속이 사용 가능합니다!")
            print("   (OpenCV CUDA 없이도 메인 처리는 GPU로 가속됩니다)")
        elif not has_opencv_gpu and not has_tf_gpu:
            print("💡 GPU 가속을 위해 PyTorch with CUDA 또는 OpenCV-CUDA를 설치하세요.")
            
        return has_torch_gpu or has_opencv_gpu or has_tf_gpu
            
    except Exception as e:
        print(f"⚠️ GPU 지원 확인 중 오류: {e}")
        return False

def create_handlandmarker_with_gpu():
    """GPU 지원을 포함한 HandLandmarker 객체를 생성합니다."""
    hand_landmarker_path = os.path.join(script_dir, "hand_landmarker.task")
    
    # 1. GPU delegate 먼저 시도
    try:
        if hasattr(python.BaseOptions, 'Delegate') and hasattr(python.BaseOptions.Delegate, 'GPU'):
            base_options = python.BaseOptions(
                model_asset_path=hand_landmarker_path,
                delegate=python.BaseOptions.Delegate.GPU
            )
            print("🚀 GPU delegate 설정 시도...")
            
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            
            # GPU HandLandmarker 생성 시도
            detector = vision.HandLandmarker.create_from_options(options)
            print("✅ GPU 가속 HandLandmarker 초기화 완료")
            return detector
            
        else:
            print("⚠️ GPU delegate 미지원")
            raise Exception("GPU delegate not supported")
            
    except Exception as e:
        print(f"⚠️ GPU HandLandmarker 초기화 실패: {e}")
        print("🔄 CPU 모드로 폴백...")
    
    # 2. CPU 모드로 폴백
    try:
        base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        detector = vision.HandLandmarker.create_from_options(options)
        print("✅ CPU HandLandmarker 초기화 완료")
        return detector
        
    except Exception as e:
        print(f"❌ CPU HandLandmarker 초기화도 실패: {e}")
        raise e

# Create paths relative to script location
min_hand_detection_confidence = 0.85
min_hand_presence_confidence = 0.8
min_tracking_confidence = 0.5
VisionRunningMode = mp.tasks.vision.RunningMode

# GPU 지원 확인
print("🚀 MediaPipe HandLandmarker 초기화...")
check_gpu_support()

# HandLandmarker 생성 (GPU 지원 포함)
detector = create_handlandmarker_with_gpu()
print("✅ MediaPipe HandLandmarker 초기화 완료")

filepath = os.path.join(script_dir, 'videocapture') #Video capture directory relative to script

def datagenerate(videoname):
    start = timemodule.time()
    video = cv2.VideoCapture(
        os.path.join(filepath,
                     videoname,
        )
    )
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    ratio=height/width
    dirname = os.path.join(
        filepath,
        videoname[:-4]+'_'+f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}",
    )

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    
    # Load keyboard data using new JSON format
    video_name = videoname[:-4]  # Remove .mp4 extension
    json_file_path = os.path.join(script_dir, "pixel_points", f"{video_name}_pixel_points.json")
    
    if os.path.exists(json_file_path):
        print(f"📁 Loading keyboard data from JSON: {json_file_path}")
        keyboard, black_key_data = load_keyboard_from_json_pixel_points(json_file_path)
        if keyboard is None:
            print("❌ Failed to load keyboard data from JSON")
            exit(1)
        print(f"✅ Keyboard loaded successfully with {len(keyboard)} keys")
    else:
        print(f"❌ JSON file not found: {json_file_path}")
        print("💡 Please ensure the JSON file exists in the pixel_points directory")
        print(f"   Expected format: {video_name}_pixel_points.json")
        exit(1)
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
        for _ in stqdm(range(frame_count), desc="Generating Frame images from video..."):
            if count < frame_count:
                ret, image = video.read()
                cv2.imwrite(r"{}".format(dirname + "/frame%d.jpg" % count), image)
                count += 1
                pbar.update(1)
            del ret, image
        if count >= frame_count:
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
    frame=0
    for _ in stqdm(range(file_count), desc="Generating hand information from framewise images..."):
        with Image.open(dirname + "/frame%d.jpg" % frame) as img:
            img_np = np.array(img)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
            time = frame * int(1000 / frame_rate)
            detection_result = detector.detect_for_video(image, timestamp_ms=time)
            # tempimglist.append([image, detection_result])                          # 메모리 관리를 위해 좀 느리더라도 뒤에서 image는 다시 로드해준다.
            tempimglist.append(detection_result)
            if len(detection_result.handedness) == 0:
                nohandframelist.append(frame)

            for j in range(len(detection_result.handedness)):
                for finger_landmark in detection_result.hand_landmarks[j]:
                    finger_landmark.x=finger_landmark.x*2-1 #[0,1]->[-1,1]
                    finger_landmark.y=finger_landmark.y*2-1 #[0,1]->[-1,1]

            handsinfo = [
                handclass(
                    handtype=detection_result.handedness[j][0].category_name,
                    handlandmark=detection_result.hand_landmarks[j],
                    handframe=frame,
                )
                for j in range(len(detection_result.handedness))
            ]                                                               # handclass : ( handtype = "Left" or "Right" / hand_landmarks = (Normalized_landmark(x,y,z,visibiliry,presence)) handframe = frame : int )
            handlist.append(handsinfo)
            handtypelist.append([hand.handtype for hand in handsinfo])
            if frame == max(0, file_count - 100):
                # Use enhanced drawing function with colored polygon lines
                keyboard_image = draw_keyboard_with_points(
                    image.numpy_view(), 
                    keyboard, 
                    None,  # keystone_data not needed for drawing
                    black_key_data, 
                    show_intermediate_points=True
                )
                cv2.imwrite(
                    dirname + "/" + "keyboard.jpg", cv2.cvtColor(keyboard_image, cv2.COLOR_BGR2RGB)
                )
            pbar2.update(1)
            frame += 1

    lhmodel,rhmodel=modelskeleton(handlist)
    depthlist(handlist,lhmodel,rhmodel,ratio)

    faultyframe = faultyframes(handlist)
    pbar2.close()
    print("Calculating faulty frames...")
    floatingframes = detectfloatingframes(handlist, frame_count, faultyframe, lhmodel, rhmodel, ratio=ratio)
    with open(
        dirname
        + "/floatingframes_"
        + videoname[:-4]
        + "_"
        + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(floatingframes, f)

    if os.path.exists(        
        dirname
        + "/handlist_"
        + videoname[:-4]
        + "_"
        + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + ".pkl",
    ):
        os.remove(
            dirname
            + "/handlist_"
            + videoname[:-4]
            + "_"
            + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            + ".pkl",
        )
    with open(
        dirname
        + "/handlist_"
        + videoname[:-4]
        + "_"
        + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
        + ".pkl",
        "wb",
    ) as f:
        pickle.dump(handlist, f)

    print("Data generated and saved")
    print(
        f"faulty frames:{len(faultyframe)}, nohand frames:{len(nohandframelist)}"
    )
    pbar3 = tqdm(total=file_count)
    frame=0
    for _ in stqdm(range(file_count), desc="Generating new images with hand information..."):
        #image, detection_result = tempimglist[frame]
        with Image.open(dirname + "/frame%d.jpg" % frame) as img:
            img_np = np.array(img)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
            detection_result = tempimglist[frame]
            
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
        pbar3.update(1)
        frame += 1
    pbar3.close()
    
    print("Image generated and saved")
    datatime = timemodule.time()
    print(f"Data generation time: {datatime-start: .5f} sec")




##############################
#datagenerate() ##############
##############################



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

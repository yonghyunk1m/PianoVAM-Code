import cv2
import numpy as np
import os
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from floatinghands import *
from midicomparison import *
import pickle
from tqdm.auto import tqdm
from stqdm import stqdm
import time as timemodule
import json
# Note: file order is main>evaluate>midicomparison>floatinghands

# STEP 2: Create an HandLandmarker object with GPU support.
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def check_gpu_support():
    """GPU 지원 여부를 확인합니다."""
    try:
        # OpenCV GPU 지원 확인
        has_opencv_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        # TensorFlow GPU 지원 확인 (선택적)
        try:
            import tensorflow as tf
            has_tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            has_tf_gpu = False
        
        print(f"🔍 GPU 지원 상태:")
        print(f"   OpenCV CUDA: {'✅ 사용 가능' if has_opencv_gpu else '❌ 사용 불가'}")
        print(f"   TensorFlow GPU: {'✅ 사용 가능' if has_tf_gpu else '❌ 사용 불가'}")
        
        if not has_opencv_gpu and not has_tf_gpu:
            print("💡 GPU 가속을 위해 CUDA 드라이버와 OpenCV-CUDA 또는 TensorFlow-GPU를 설치하세요.")
            
        return has_opencv_gpu or has_tf_gpu
            
    except Exception as e:
        print(f"⚠️ GPU 지원 확인 중 오류: {e}")
        return False

def create_handlandmarker_with_gpu():
    """GPU 지원을 포함한 HandLandmarker 객체를 생성합니다."""
    hand_landmarker_path = os.path.join(script_dir, "hand_landmarker.task")
    
    # GPU delegate 설정 시도
    base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)
    
    try:
        # GPU delegate 시도
        if hasattr(python.BaseOptions, 'Delegate') and hasattr(python.BaseOptions.Delegate, 'GPU'):
            base_options = python.BaseOptions(
                model_asset_path=hand_landmarker_path,
                delegate=python.BaseOptions.Delegate.GPU
            )
            print("🚀 GPU delegate 설정 완료 - GPU 가속 활성화")
        else:
            print("⚠️ GPU delegate 미지원 - CPU 사용")
    except Exception as e:
        print(f"⚠️ GPU delegate 설정 실패, CPU 사용: {e}")
    
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    return vision.HandLandmarker.create_from_options(options)

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
    with open(
        os.path.join(script_dir, "keyboardcoordinateinfo.pkl"),
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
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # 기존 파일들 정리
    for file in os.scandir(dirname):
        os.remove(file.path)

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)

    # 비디오에서 직접 프레임을 읽어서 MediaPipe 처리
    nohandframelist = []
    handlist = []
    handtypelist = []
    
    pbar2 = tqdm(total=frame_count)
    frame = 0
    
    for _ in stqdm(range(frame_count), desc="Generating hand information from video frames..."):
        if frame < frame_count:
            ret, cv_image = video.read()
            if not ret:
                break
                
            # OpenCV 이미지를 MediaPipe 형식으로 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            time = frame * int(1000 / frame_rate)
            detection_result = detector.detect_for_video(image, timestamp_ms=time)
            
            if len(detection_result.handedness) == 0:
                nohandframelist.append(frame)

            for j in range(len(detection_result.handedness)):
                for finger_landmark in detection_result.hand_landmarks[j]:
                    finger_landmark.x = finger_landmark.x * 2 - 1  # [0,1] -> [-1,1]
                    finger_landmark.y = finger_landmark.y * 2 - 1  # [0,1] -> [-1,1]

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
            
            # 키보드 시각화 이미지 생성 (디버깅 용도)
            if frame == max(0, frame_count - 100):
                keyboard_image = draw_keyboard_on_image(image.numpy_view(), keyboard)
                cv2.imwrite(
                    dirname + "/" + "keyboard.jpg", cv2.cvtColor(keyboard_image, cv2.COLOR_BGR2RGB)
                )
            
            pbar2.update(1)
            frame += 1
    
    video.release()

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
    
    # 처리 완료 후 불필요한 파일들 정리
    print("Cleaning up temporary files...")
    cleanup_count = 0
    for file in os.scandir(dirname):
        # .pkl 파일과 keyboard.jpg는 보존, 나머지는 삭제
        if file.name.endswith('.pkl') or file.name == 'keyboard.jpg':
            continue
        else:
            try:
                os.remove(file.path)
                cleanup_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {file.name}: {e}")
    
    if cleanup_count > 0:
        print(f"Cleaned up {cleanup_count} temporary files")
    
    datatime = timemodule.time()
    print(f"Data generation time: {datatime-start: .5f} sec")




##############################
#datagenerate() ##############
##############################

# 모든 비디오 파일을 자동으로 처리하는 루프
def load_progress_tracker():
    """처리 진행 상황을 JSON 파일에서 로드합니다."""
    progress_file = os.path.join(script_dir, "video_processing_progress.json")
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_progress_tracker(progress_data):
    """처리 진행 상황을 JSON 파일에 저장합니다."""
    progress_file = os.path.join(script_dir, "video_processing_progress.json")
    
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

def initialize_progress_tracker(video_files):
    """모든 비디오 파일의 초기 처리 상태를 설정합니다."""
    progress_data = load_progress_tracker()
    
    # 새로운 파일들을 False로 초기화
    for video_file in video_files:
        if video_file not in progress_data:
            progress_data[video_file] = False
    
    # JSON에 있지만 실제로는 없는 파일들 제거
    existing_files = set(video_files)
    progress_data = {k: v for k, v in progress_data.items() if k in existing_files}
    
    save_progress_tracker(progress_data)
    return progress_data

def print_progress_summary(progress_data):
    """처리 진행 상황 요약을 출력합니다."""
    total = len(progress_data)
    completed = sum(1 for status in progress_data.values() if status)
    remaining = total - completed
    
    print(f"📊 처리 진행 상황:")
    print(f"   전체: {total}개")
    print(f"   완료: {completed}개 ({completed/total*100:.1f}%)")
    print(f"   남은: {remaining}개 ({remaining/total*100:.1f}%)")
    
    return total, completed, remaining

def process_all_videos():
    """videocapture 폴더의 모든 mp4 파일을 처리합니다."""
    
    # 비디오 파일 목록 가져오기
    video_files = []
    if os.path.exists(filepath):
        for file in os.listdir(filepath):
            if file.lower().endswith('.mp4'):
                video_files.append(file)
    
    if not video_files:
        print("❌ videocapture 폴더에 mp4 파일이 없습니다.")
        return
    
    # 파일명으로 정렬
    video_files.sort()
    
    print(f"🎥 총 {len(video_files)}개의 비디오 파일을 발견했습니다.")
    print(f"📁 비디오 경로: {filepath}")
    
    # 진행 상황 추적기 초기화
    progress_data = initialize_progress_tracker(video_files)
    
    # 진행 상황 요약 출력
    total, completed, remaining = print_progress_summary(progress_data)
    
    if remaining == 0:
        print("🎉 모든 비디오 파일이 이미 처리되었습니다!")
        return
    
    # 키보드 정보 파일 확인
    keyboard_info_path = os.path.join(script_dir, "keyboardcoordinateinfo.pkl")
    if not os.path.exists(keyboard_info_path):
        print("❌ keyboardcoordinateinfo.pkl 파일이 없습니다.")
        print("   먼저 키보드 좌표 정보를 설정해주세요.")
        return
    
    # 미완료 파일들만 처리
    unprocessed_files = [f for f, status in progress_data.items() if not status]
    
    print(f"\n🔄 {len(unprocessed_files)}개의 미완료 파일을 처리합니다.")
    
    processed_count = 0
    failed_files = []
    
    for i, video_file in enumerate(unprocessed_files, 1):
        print(f"\n{'='*60}")
        print(f"🔄 [{i}/{len(unprocessed_files)}] 처리 중: {video_file}")
        print(f"   (전체 진행률: {completed + processed_count}/{total})")
        print(f"{'='*60}")
        
        try:
            # 해당하는 MIDI 파일이 있는지 확인 (선택사항)
            midi_path = os.path.join(script_dir, 'midiconvert', video_file[:-4] + '.mid')
            if os.path.exists(midi_path):
                print(f"✅ 대응되는 MIDI 파일 발견: {video_file[:-4]}.mid")
            else:
                print(f"⚠️  대응되는 MIDI 파일 없음: {video_file[:-4]}.mid")
            
            # 실제 출력 디렉토리 확인 (이중 체크)
            output_dir = os.path.join(
                filepath,
                video_file[:-4] + '_' + f"{min_hand_detection_confidence*100}{min_hand_presence_confidence*100}{min_tracking_confidence*100}"
            )
            
            if os.path.exists(output_dir):
                print(f"✅ 출력 디렉토리가 이미 존재합니다. 완료로 표시합니다.")
                progress_data[video_file] = True
                save_progress_tracker(progress_data)
                processed_count += 1
                continue
                
            # 데이터 생성 실행
            datagenerate(video_file)
            
            # 성공 시 진행 상황 업데이트
            progress_data[video_file] = True
            save_progress_tracker(progress_data)
            processed_count += 1
            
            print(f"✅ [{i}/{len(unprocessed_files)}] 완료: {video_file}")
            print(f"   진행 상황이 저장되었습니다.")
            
        except Exception as e:
            print(f"❌ [{i}/{len(unprocessed_files)}] 실패: {video_file}")
            print(f"   오류: {str(e)}")
            failed_files.append(video_file)
            # 실패해도 진행 상황은 저장 (False 상태 유지)
            continue
    
    # 최종 결과 출력
    print(f"\n{'='*60}")
    print(f"🎯 배치 처리 완료!")
    print(f"   이번 실행에서 성공: {processed_count}개")
    print(f"   이번 실행에서 실패: {len(failed_files)}개")
    
    # 최종 진행 상황
    final_total, final_completed, final_remaining = print_progress_summary(progress_data)
    
    if failed_files:
        print(f"\n❌ 실패한 파일들:")
        for file in failed_files:
            print(f"   - {file}")
            
    if final_remaining == 0:
        print(f"\n🎉 모든 비디오 파일 처리가 완료되었습니다!")
    else:
        print(f"\n📝 다음에 처리할 파일들:")
        for video_file, status in progress_data.items():
            if not status:
                print(f"   - {video_file}")
    
    print(f"{'='*60}")

def show_progress_status():
    """현재 처리 진행 상황만 표시합니다."""
    video_files = []
    if os.path.exists(filepath):
        for file in os.listdir(filepath):
            if file.lower().endswith('.mp4'):
                video_files.append(file)
    
    if not video_files:
        print("❌ videocapture 폴더에 mp4 파일이 없습니다.")
        return
    
    video_files.sort()
    progress_data = initialize_progress_tracker(video_files)
    
    print(f"📊 비디오 처리 상태 현황:")
    print(f"{'='*60}")
    
    total, completed, remaining = print_progress_summary(progress_data)
    
    print(f"\n✅ 완료된 파일들:")
    completed_files = [f for f, status in progress_data.items() if status]
    for file in completed_files:
        print(f"   - {file}")
    
    print(f"\n⏳ 대기중인 파일들:")
    pending_files = [f for f, status in progress_data.items() if not status]
    for file in pending_files:
        print(f"   - {file}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    print("🚀 PianoVAM 손 인식 데이터 생성 도구")
    print(f"📂 작업 디렉토리: {script_dir}")
    
    # 사용자 옵션
    print("\n선택하세요:")
    print("1. 모든 비디오 파일 처리")
    print("2. 현재 진행 상황만 확인")
    
    try:
        choice = input("번호를 입력하세요 (1 또는 2): ").strip()
        
        if choice == "1":
            # 모든 비디오 파일 처리 실행
            process_all_videos()
        elif choice == "2":
            # 진행 상황만 표시
            show_progress_status()
        else:
            print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        print("💾 진행 상황은 JSON 파일에 저장되어 있어 나중에 이어서 할 수 있습니다.")



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

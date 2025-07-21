#!/usr/bin/env python3
"""
PyTorch 기반 Floating Hand Detection
- PyTorch GPU 가속 알고리즘만 사용
- 실시간 성능 모니터링
- 상세 시각화 비디오 생성
"""

import os
import pickle
import time
import numpy as np
import cv2
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import psutil
import threading
from datetime import datetime, timedelta

# 진행률 표시용 (선택적)
try:
    from stqdm import stqdm
    HAS_STQDM = True
except ImportError:
    HAS_STQDM = False



# GPU 메모리 모니터링을 위한 import
try:
    import torch
    TORCH_AVAILABLE = True
    print("? PyTorch 사용 가능")
    if torch.cuda.is_available():
        print(f"? CUDA GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        print("??  CPU 모드로 실행")
except ImportError:
    print("? PyTorch가 설치되지 않았습니다.")
    exit(1)

# PyTorch 버전 floating hands 모듈 import
try:
    import floatinghands_torch_pure as pytorch_version
    print("? PyTorch Floating Hands 모듈 로드 성공")
except ImportError as e:
    print(f"? PyTorch Floating Hands 모듈 로드 실패: {e}")
    exit(1)

# MIDI 데이터 처리용 모듈 import (임계값 계산용)
try:
    import midicomparison
    print("? MIDI 비교 모듈 로드 성공")
except ImportError as e:
    print(f"??  MIDI 비교 모듈 로드 실패: {e}")
    print("   MIDI 기반 임계값 계산은 사용할 수 없습니다.")
    midicomparison = None

# main_loop import는 나중에 필요할 때만 하도록 지연
main_loop_module = None

def import_main_loop():
    """main_loop 모듈을 지연 import합니다"""
    global main_loop_module
    if main_loop_module is None:
        try:
            import main_loop as main_loop_module_temp
            main_loop_module = main_loop_module_temp
        except Exception as e:
            print(f"? main_loop 모듈 로드 실패: {e}")
            raise
    return main_loop_module

# MediaPipe 그리기 유틸리티 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_hand_landmarks_with_floating(image, hand_landmarks, handedness, floating_status, depth_info, threshold_value=0.9):
    """Hand landmarks와 floating 상태를 시각적으로 표시합니다"""
    if not hand_landmarks:
        return image
    
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    for idx, landmarks in enumerate(hand_landmarks):
        # 손 종류 확인
        if idx < len(handedness):
            hand_type = handedness[idx].classification[0].category_name
            
            # 깊이 정보 가져오기
            depth_value = depth_info.get(f'{hand_type}_depth', 0.0)
            is_floating = floating_status.get(f'{hand_type}_floating', False)
            
            # 임계값 설정
            if isinstance(threshold_value, dict):
                threshold = threshold_value.get(hand_type, 0.9)
            else:
                threshold = threshold_value
            
            # 색상 설정 (floating 상태에 따라)
            if is_floating:
                connection_color = (0, 255, 0)  # 초록색 (floating)
                landmark_color = (0, 255, 0)
                status_text = "FLOATING"
                status_color = (0, 255, 0)
            else:
                connection_color = (255, 0, 0)  # 빨간색 (not floating)
                landmark_color = (255, 0, 0)
                status_text = "PLAYING"
                status_color = (255, 0, 0)
            
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2)
            )
            
            # 텍스트 정보 표시
            text_y_offset = 30 if hand_type == "Left" else 60
            
            # 손 종류 표시
            cv2.putText(annotated_image, f"{hand_type} Hand", 
                       (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 상태 표시
            cv2.putText(annotated_image, f"Status: {status_text}", 
                       (10, text_y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # 깊이 값 표시
            cv2.putText(annotated_image, f"Depth: {depth_value:.3f}", 
                       (10, text_y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 임계값 표시
            cv2.putText(annotated_image, f"Threshold: {threshold:.2f}", 
                       (10, text_y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    return annotated_image

def draw_header_info(image, frame_num, floating_status, depth_info, total_frames, processing_time=None):
    """비디오 헤더에 전체 정보를 표시합니다"""
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    # 헤더 배경
    header_height = 100
    cv2.rectangle(annotated_image, (0, 0), (width, header_height), (0, 0, 0), -1)
    
    # 제목
    cv2.putText(annotated_image, "PyTorch Floating Hand Detection", 
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 프레임 정보
    cv2.putText(annotated_image, f"Frame: {frame_num}/{total_frames}", 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 처리 시간 (있는 경우)
    if processing_time:
        cv2.putText(annotated_image, f"Processing: {processing_time:.2f}s", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 우측에 전체 floating 상태 요약
    right_x = width - 300
    cv2.putText(annotated_image, "Floating Status:", 
               (right_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    left_floating = floating_status.get('Left_floating', False)
    right_floating = floating_status.get('Right_floating', False)
    
    left_color = (0, 255, 0) if left_floating else (255, 0, 0)
    right_color = (0, 255, 0) if right_floating else (255, 0, 0)
    
    cv2.putText(annotated_image, f"Left: {'FLOATING' if left_floating else 'PLAYING'}", 
               (right_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
    cv2.putText(annotated_image, f"Right: {'FLOATING' if right_floating else 'PLAYING'}", 
               (right_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
    
    return annotated_image

def create_keyboard_overlay(image, keyboard_coords=None):
    """키보드 오버레이를 추가합니다 (선택사항)"""
    if keyboard_coords is None:
        return image
    
    annotated_image = image.copy()
    # 키보드 영역 표시 로직 (필요시 구현)
    return annotated_image

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.current_step = None
        self.step_start_time = None
        self.step_progress = 0
        self.step_total = 0
        self.total_steps = 0
        self.current_step_num = 0
        
    def start_monitoring(self, total_steps=0):
        """모니터링 시작"""
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step_num = 0
        print(f"? 성능 모니터링 시작")
        self._print_system_info()
        
    def stop_monitoring(self):
        """모니터링 종료"""
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n? 전체 처리 완료: {self._format_time(total_time)}")
        
    def start_step(self, step_name: str, total_items=0):
        """새 단계 시작"""
        self.current_step_num += 1
        self.current_step = step_name
        self.step_start_time = time.time()
        self.step_progress = 0
        self.step_total = total_items
        
        progress_text = f"[{self.current_step_num}/{self.total_steps}]" if self.total_steps > 0 else ""
        print(f"\n? {progress_text} {step_name}")
        
    def log_operation(self, operation_name: str, details: str = "", show_gpu=False):
        """작업 로깅"""
        print(f"   ??  {operation_name}: {details}")
        if show_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"      ? GPU 메모리: {gpu_memory:.2f}GB")
            
    def log_batch_progress(self, batch_num: int, total_batches: int, batch_size: int, 
                          items_processed: int, operation: str = ""):
        """배치 처리 진행 상황 로깅"""
        progress = (batch_num / total_batches) * 100
        print(f"   ? 배치 {batch_num}/{total_batches} ({progress:.1f}%) - "
              f"{items_processed:,}개 처리됨 {operation}")
              
    def log_algorithm_start(self, algorithm_name: str, method: str, precision: str, 
                           processing_type: str, optimizations: list = None):
        """알고리즘 시작 로깅"""
        print(f"? 알고리즘: {algorithm_name}")
        print(f"   ? 해법: {method}")
        print(f"   ? 정밀도: {precision}")
        print(f"   ? 처리: {processing_type}")
        if optimizations:
            print(f"   ? 최적화:")
            for opt in optimizations:
                print(f"      ? {opt}")
                
    def log_memory_usage(self, stage: str, cpu_memory: float = None, gpu_memory: float = None):
        """메모리 사용량 로깅"""
        if cpu_memory is None:
            cpu_memory = psutil.virtual_memory().used / 1024**3
        
        memory_info = f"   ? {stage} - CPU: {cpu_memory:.2f}GB"
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if gpu_memory is None:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
            memory_info += f", GPU: {gpu_memory:.2f}GB"
            
        print(memory_info)
        
    def update_step_progress(self, current: int, message: str = ""):
        """단계 진행 상황 업데이트"""
        self.step_progress = current
        if self.step_total > 0:
            progress = (current / self.step_total) * 100
            print(f"   ? 진행: {current}/{self.step_total} ({progress:.1f}%) {message}")
        else:
            print(f"   ? {message}")
            
    def finish_step(self, message: str = ""):
        """단계 완료"""
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            print(f"   ? 완료: {message} ({self._format_time(step_time)})")
        
    def _print_system_info(self):
        """시스템 정보 출력"""
        print(f"? 시스템 정보:")
        print(f"   ??  CPU: {psutil.cpu_count()}코어")
        print(f"   ? RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ? GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print(f"   ? GPU: 사용 불가")
            
    def _format_time(self, seconds):
        """시간 포맷팅"""
        if seconds < 60:
            return f"{seconds:.2f}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}분 {secs:.1f}초"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}시간 {minutes}분 {secs:.1f}초"

class handclass:
    """Hand class - MediaPipe detection 결과를 저장하는 클래스"""
    def __init__(self, handtype, handlandmark, handframe):
        self.handtype = handtype
        self.handlandmark = handlandmark
        self.handframe = handframe
        self.handdepth = 1  # default
    
    def set_handdepth(self, handdepth):
        self.handdepth = handdepth

def process_raw_video_with_mediapipe(video_path: str, video_name: str, 
                                   min_detection_confidence: float = 0.8,
                                   min_tracking_confidence: float = 0.5,
                                   frame_limit: int = None) -> Tuple[List, float]:
    """Raw 비디오를 MediaPipe로 처리하여 handlist 생성"""
    print(f"🎬 Raw 비디오 MediaPipe 처리 시작: {video_name}")
    
    # MediaPipe 초기화
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    ) as hands:
        
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ratio = height / width
        
        print(f"   📊 비디오 정보: {width}x{height}, {fps:.1f}FPS, {total_frames:,}프레임")
        
        # 프레임 제한 적용
        if frame_limit and total_frames > frame_limit:
            total_frames = frame_limit
            print(f"   ⏱️  Quick test: {frame_limit:,}프레임으로 제한")
        
        handlist = []
        frame = 0
        
        if HAS_STQDM:
            progress_bar = stqdm(range(total_frames), desc=f"🎬 {video_name} MediaPipe 처리")
        else:
            progress_bar = range(total_frames)
            print(f"   🔄 {total_frames:,}프레임 처리 중... (진행률 표시: pip install stqdm)")
        
        for _ in progress_bar:
            if frame >= total_frames:
                break
                
            ret, cv_image = cap.read()
            if not ret:
                break
            
            # OpenCV 이미지를 RGB로 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results = hands.process(rgb_image)
            
            # 손이 감지된 경우 handclass 객체 생성
            handsinfo = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for j in range(len(results.multi_handedness)):
                    # 좌표 정규화: [0,1] -> [-1,1]
                    landmarks = results.multi_hand_landmarks[j]
                    for landmark in landmarks.landmark:
                        landmark.x = landmark.x * 2 - 1
                        landmark.y = landmark.y * 2 - 1
                    
                    # handclass 객체 생성
                    try:
                        # MediaPipe handedness 정보 올바르게 추출
                        handedness = results.multi_handedness[j]
                        if hasattr(handedness, 'classification') and len(handedness.classification) > 0:
                            hand_type = handedness.classification[0].category_name
                            if frame < 10:  # 첫 10프레임만 디버깅 출력
                                print(f"   🔍 감지된 손 타입: {hand_type} (프레임 {frame})")
                        else:
                            hand_type = "Left" if j == 0 else "Right"  # 기본값
                            if frame < 10:  # 첫 10프레임만 디버깅 출력
                                print(f"   ⚠️ handedness 정보 없음, 기본값 사용: {hand_type} (프레임 {frame})")
                    except (AttributeError, IndexError) as e:
                        hand_type = "Left" if j == 0 else "Right"  # 기본값
                        if frame < 10:  # 첫 10프레임만 디버깅 출력
                            print(f"   ⚠️ handedness 추출 실패: {e}, 기본값 사용: {hand_type} (프레임 {frame})")
                    
                    hand_obj = handclass(
                        handtype=hand_type,
                        handlandmark=landmarks.landmark,
                        handframe=frame
                    )
                    handsinfo.append(hand_obj)
            
            handlist.append(handsinfo)
            frame += 1
        
        cap.release()
        
        # 통계
        total_hands = sum(len(hands) for hands in handlist)
        frames_with_hands = len([hands for hands in handlist if hands])
        
        print(f"✅ MediaPipe 처리 완료:")
        print(f"   📊 총 손 감지: {total_hands:,}개")
        print(f"   🎯 손이 있는 프레임: {frames_with_hands:,}/{len(handlist):,}개")
        print(f"   📐 비디오 비율: {ratio:.3f}")
        
        return handlist, ratio

def save_depth_data(handlist: List, video_name: str, output_dir: str = None) -> str:
    """깊이 데이터를 파일로 저장합니다"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'depth_data')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 깊이 데이터 저장 중: {video_name}")
    
    # 깊이 데이터 수집
    depth_data = {'Left': [], 'Right': [], 'All': []}
    frame_data = []
    
    for frame_idx, hands in enumerate(handlist):
        if not hands:
            continue
            
        frame_info = {'frame': frame_idx, 'hands': []}
        
        for hand in hands:
            if hasattr(hand, 'handdepth') and hand.handdepth is not None:
                depth = float(hand.handdepth)
                hand_type = getattr(hand, 'handtype', 'Unknown')
                
                # 전체 데이터에 추가
                depth_data['All'].append(depth)
                
                # 손 타입별 데이터에 추가
                if hand_type in depth_data:
                    depth_data[hand_type].append(depth)
                
                # 프레임별 데이터에 추가
                frame_info['hands'].append({
                    'type': hand_type,
                    'depth': depth,
                    'frame': frame_idx
                })
        
        if frame_info['hands']:
            frame_data.append(frame_info)
    
    # 데이터 정리
    depth_summary = {
        'video_name': video_name,
        'timestamp': datetime.now().isoformat(),
        'total_hands': len(depth_data['All']),
        'left_hands': len(depth_data['Left']),
        'right_hands': len(depth_data['Right']),
        'total_frames': len(frame_data),
        'depth_range': (min(depth_data['All']), max(depth_data['All'])) if depth_data['All'] else (0, 0),
        'depth_data': depth_data,
        'frame_data': frame_data
    }
    
    # 기본 통계 계산
    if depth_data['All']:
        depths_array = np.array(depth_data['All'])
        depth_summary['statistics'] = {
            'mean': float(np.mean(depths_array)),
            'std': float(np.std(depths_array)),
            'min': float(np.min(depths_array)),
            'max': float(np.max(depths_array)),
            'median': float(np.median(depths_array)),
            'q25': float(np.percentile(depths_array, 25)),
            'q75': float(np.percentile(depths_array, 75)),
            'q90': float(np.percentile(depths_array, 90)),
            'q95': float(np.percentile(depths_array, 95))
        }
    
    # 저장
    save_path = os.path.join(output_dir, f'{video_name}_depth_data.json')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(depth_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📊 저장 완료:")
    print(f"   파일: {save_path}")
    print(f"   총 손: {depth_summary['total_hands']:,}개")
    print(f"   좌손: {depth_summary['left_hands']:,}개, 우손: {depth_summary['right_hands']:,}개")
    print(f"   유효 프레임: {depth_summary['total_frames']:,}개")
    if depth_data['All']:
        stats = depth_summary['statistics']
        print(f"   깊이 범위: {stats['min']:.3f} ~ {stats['max']:.3f}")
        print(f"   평균: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"   중앙값: {stats['median']:.3f}")
        print(f"   90분위수: {stats['q90']:.3f}")
    
    return save_path

class TorchFloatingDetector:
    """PyTorch 기반 Floating Hand Detection"""
    
    # 기본 설정
    QUICK_TEST = True  # 1200프레임만 처리
    FRAME_LIMIT = 1200
    
    # 캐싱 설정
    ENABLE_CACHING = True  # 처리 완료된 데이터 캐싱 활성화
    ENABLE_LANDMARK_CACHING = True  # MediaPipe landmark 데이터 캐싱 활성화
    
    # 비디오 생성 설정
    GENERATE_VIDEO = True  # 결과 비디오 생성
    
    # 임계값 설정
    AUTO_THRESHOLD = False  # 자동 임계값 사용 여부
    THRESHOLD_METHOD = 'midi_based'  # 'statistical', 'clustering', 'valley', 'midi_based'
    FIXED_THRESHOLD = 0.9  # 고정 임계값
    
    # 비디오 처리 설정
    TARGET_VIDEO_DIR = "/home/jhbae/PianoVAM-Code/FingeringDetection/videocapture"
    MAX_VIDEOS = 10  # 처리할 비디오 개수 제한
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_capture_dir = self.TARGET_VIDEO_DIR
        self.target_videos = []
        self.current_video = None
        self.quick_test = self.QUICK_TEST
        self.frame_limit = self.FRAME_LIMIT
        self.enable_caching = self.ENABLE_CACHING
        self.enable_landmark_caching = self.ENABLE_LANDMARK_CACHING
        self.generate_video = self.GENERATE_VIDEO
        
        # 고정 임계값 설정
        self.auto_threshold = self.AUTO_THRESHOLD
        self.threshold_method = self.THRESHOLD_METHOD
        self.fixed_threshold = self.FIXED_THRESHOLD
        
        # 임계값 캐싱
        self.cached_threshold = {'Left': self.FIXED_THRESHOLD, 'Right': self.FIXED_THRESHOLD}
        
        # 처리할 비디오 목록 초기화
        self._initialize_video_list()
        
        # 캐싱 경로 설정
        self.cache_dir = os.path.join(self.script_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 깊이 데이터 저장 설정
        self.save_depth_data = True  # 깊이 데이터 저장 여부
        self.depth_data_dir = os.path.join(self.script_dir, 'depth_data')
        
        # Raw 비디오 처리 설정
        self.use_raw_videos = False  # Raw 비디오 직접 처리 여부
        
        # MediaPipe 설정
        self.min_detection_confidence = 0.8
        self.min_tracking_confidence = 0.5
        
    def _initialize_video_list(self):
        """처리 가능한 비디오 목록을 초기화합니다"""
        if not os.path.exists(self.video_capture_dir):
            print(f"? 비디오 디렉토리가 존재하지 않습니다: {self.video_capture_dir}")
            return
        
        # 전체 mp4 파일 개수 확인
        all_mp4_files = [f for f in os.listdir(self.video_capture_dir) if f.endswith('.mp4')]
        print(f"📁 videocapture 폴더에서 발견된 비디오: {len(all_mp4_files)}개")
        
        processed_videos = []
        raw_videos = []
        
        # .mp4 파일과 대응되는 데이터가 있는 비디오들을 찾기
        for item in all_mp4_files:
            video_name = item[:-4]
            
            # 해당 pkl 데이터가 있는지 확인
            data_dir = os.path.join(self.video_capture_dir, f"{video_name}_858550")
            if os.path.exists(data_dir):
                handlist_file = os.path.join(data_dir, f"handlist_{video_name}_858550.pkl")
                floating_file = os.path.join(data_dir, f"floatingframes_{video_name}_858550.pkl")
                
                if os.path.exists(handlist_file) and os.path.exists(floating_file):
                    processed_videos.append(video_name)
                else:
                    raw_videos.append(video_name)
            else:
                raw_videos.append(video_name)
        
        print(f"📊 분석 결과:")
        print(f"   ✅ 이미 처리된 비디오: {len(processed_videos)}개")
        print(f"   🔄 미처리 비디오: {len(raw_videos)}개")
        
        # 처리된 비디오가 있으면 그것을 우선 사용
        if len(processed_videos) > 0:
            print(f"\n🎯 이미 처리된 비디오를 사용합니다")
            self.target_videos = processed_videos
            self.use_raw_videos = False
        else:
            print(f"\n🎯 Raw 비디오를 직접 처리합니다")
            self.target_videos = raw_videos
            self.use_raw_videos = True
        
        # MAX_VIDEOS 제한 적용
        if len(self.target_videos) > self.MAX_VIDEOS:
            print(f"📝 {len(self.target_videos)}개 중 최신 {self.MAX_VIDEOS}개만 선택")
            # 최신 파일 순으로 정렬 후 선택
            self.target_videos = sorted(self.target_videos, reverse=True)[:self.MAX_VIDEOS]
        
        print(f"\n🎯 처리 대상 비디오: {len(self.target_videos)}개")
        processing_type = "Raw 비디오 직접 처리" if self.use_raw_videos else "이미 처리된 데이터 사용"
        print(f"   📋 처리 방식: {processing_type}")
        for i, video in enumerate(self.target_videos, 1):
            print(f"   {i}. {video}")
    
    def set_current_video(self, video_name: str):
        """현재 처리할 비디오를 설정합니다"""
        self.current_video = video_name
        self.target_video = video_name
        
    def find_target_data(self) -> Dict[str, Any]:
        """현재 비디오의 데이터 경로를 찾습니다"""
        video_name = self.current_video
        video_file = os.path.join(self.video_capture_dir, f"{video_name}.mp4")
        
        print(f"🔍 데이터 검색: {video_name}")
        
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_file}")
        
        # use_raw_videos 플래그를 먼저 확인
        if self.use_raw_videos:
            print(f"   🎬 Raw 비디오 모드 설정됨 → Raw 비디오 직접 처리")
            return {
                'video_name': video_name,
                'video_file': video_file,
                'is_raw_video': True
            }
        
        # 처리된 데이터가 있는지 확인
        data_dir = os.path.join(self.video_capture_dir, f"{video_name}_858550")
        handlist_file = os.path.join(data_dir, f"handlist_{video_name}_858550.pkl")
        floating_file = os.path.join(data_dir, f"floatingframes_{video_name}_858550.pkl")
        
        has_processed_data = (os.path.exists(data_dir) and 
                             os.path.exists(handlist_file) and 
                             os.path.exists(floating_file))
        
        # 처리된 데이터가 없으면 Raw 비디오 모드로 강제 설정
        if not has_processed_data:
            print(f"   🎬 처리된 데이터가 없음 → Raw 비디오 모드로 진행")
            return {
                'video_name': video_name,
                'video_file': video_file,
                'is_raw_video': True
            }
        
        # 처리된 데이터가 있으면 기존 데이터 사용
        print(f"   📁 처리된 데이터 발견 → 기존 데이터 사용")
        return {
            'video_name': video_name,
            'data_dir': data_dir,
            'handlist_file': handlist_file,
            'floating_file': floating_file,
            'video_file': video_file,
            'is_raw_video': False
        }
    
    def load_data(self, data_info: Dict[str, Any]) -> Tuple[List, List, float]:
        """손 데이터 및 비디오 정보를 로딩합니다"""
        
        # 디버깅: data_info 내용 확인
        print(f"🔍 load_data 디버깅:")
        print(f"   data_info 키들: {list(data_info.keys())}")
        print(f"   is_raw_video: {data_info.get('is_raw_video', False)}")
        print(f"   self.use_raw_videos: {self.use_raw_videos}")
        
        # Raw 비디오 처리인 경우
        if data_info.get('is_raw_video', False):
            print(f"🎬 Raw 비디오 직접 처리 모드")
            
            # MediaPipe로 handlist 생성
            frame_limit = self.frame_limit if self.quick_test else None
            handlist, ratio = process_raw_video_with_mediapipe(
                video_path=data_info['video_file'],
                video_name=data_info['video_name'],
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                frame_limit=frame_limit
            )
            
            # 기존 floating 데이터 없음 (빈 리스트)
            existing_floating = []
            
            print(f"🎬 Raw 비디오 처리 완료:")
            print(f"   📊 생성된 handlist: {len(handlist)}프레임")
            print(f"   📐 비디오 비율: {ratio:.3f}")
            
            return handlist, existing_floating, ratio
        
        # 기존 처리된 데이터 사용인 경우
        print(f"📁 기존 처리된 데이터 로딩")
        
        # handlist 로딩
        with open(data_info['handlist_file'], 'rb') as f:
            handlist = pickle.load(f)
        
        # 기존 floating 결과 로딩 (참고용)
        with open(data_info['floating_file'], 'rb') as f:
            existing_floating = pickle.load(f)
        
        # 비디오 정보 로딩
        cap = cv2.VideoCapture(data_info['video_file'])
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {data_info['video_file']}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ratio = height / width
        cap.release()
        
        print(f"📁 데이터 로딩 완료:")
        print(f"   📊 handlist: {len(handlist)}프레임")
        print(f"   📋 기존 floating: {len(existing_floating)}개")
        print(f"   📏 비디오 해상도: {width}x{height} ({ratio:.3f})")
        print(f"   🎬 비디오 FPS: {fps}")
        print(f"   📈 총 프레임: {total_frames:,}개")
        
        # Quick test인 경우 프레임 제한
        if self.quick_test and len(handlist) > self.frame_limit:
            handlist = handlist[:self.frame_limit]
            print(f"   ⏱️  Quick test: {self.frame_limit}프레임으로 제한")
        
        return handlist, existing_floating, ratio
    
    def run_pytorch_detection(self, handlist: List, ratio: float) -> Tuple[List, float, List]:
        """PyTorch 기반 floating detection 실행"""
        monitor.start_step("PyTorch GPU 가속 알고리즘 실행", 4)
        
        start_time = time.time()
        
        try:
            import copy
            handlist_copy = copy.deepcopy(handlist)
            
            total_hands = sum(len(hands) for hands in handlist_copy if hands)
            total_frames = len([hands for hands in handlist_copy if hands])
            
            print(f"? PyTorch 처리 대상 분석:")
            print(f"   ? 총 손 데이터: {total_hands:,}개")
            print(f"   ??  유효 프레임: {total_frames:,}개")
            print(f"   ? 비디오 비율: {ratio:.3f}")
            
            # 디버깅: handlist 구조 확인
            print(f"\n🔍 handlist 구조 디버깅:")
            if handlist_copy:
                print(f"   총 프레임: {len(handlist_copy)}")
                for i, hands in enumerate(handlist_copy[:3]):  # 첫 3프레임만 확인
                    if hands:
                        print(f"   프레임 {i}: {len(hands)}개 손")
                        for j, hand in enumerate(hands):
                            print(f"     손 {j}: {hand.handtype}")
                            print(f"       handlandmark 타입: {type(hand.handlandmark)}")
                            print(f"       handlandmark 길이: {len(hand.handlandmark) if hasattr(hand.handlandmark, '__len__') else 'N/A'}")
                            if hasattr(hand.handlandmark, '__len__') and len(hand.handlandmark) > 0:
                                print(f"       첫 번째 landmark 타입: {type(hand.handlandmark[0])}")
                                if hasattr(hand.handlandmark[0], 'x'):
                                    print(f"       첫 번째 landmark 좌표: x={hand.handlandmark[0].x:.3f}, y={hand.handlandmark[0].y:.3f}")
                            break  # 첫 번째 손만 확인
                        break  # 유효한 프레임 하나만 확인
            
            # GPU 정보 출력
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_props = torch.cuda.get_device_properties(0)
                print(f"? GPU 가속 환경:")
                print(f"   ? GPU: {gpu_name}")
                print(f"   ? VRAM: {gpu_memory:.1f}GB")
                print(f"   ? Compute: {gpu_props.major}.{gpu_props.minor}")
                print(f"   ? CUDA: {torch.version.cuda}")
            else:
                print(f"??  CPU 모드로 실행 (GPU 가속 불가)")
            
            # 알고리즘 정보 로깅
            monitor.log_algorithm_start(
                "PyTorch GPU 가속",
                "고정점 반복법 (Fixed-point Iteration)",
                "32-bit 단정밀도 (성능 최적화)",
                "배치 병렬 처리 (GPU vectorized)",
                [
                    "GPU 텐서 연산 최적화",
                    "배치 처리 (2048-8192개 동시)",
                    "메모리 스트리밍",
                    "벡터화 연산 (CUDA 커널)",
                    "자동 미분 비활성화"
                ]
            )
            
            # 1단계: 모델 생성
            monitor.update_step_progress(1, "손 골격 모델 생성 중...")
            model_start = time.time()
            
            print(f"? PyTorch 모델 생성 시작:")
            lhmodel, rhmodel = pytorch_version.modelskeleton(handlist_copy)
            model_time = time.time() - model_start
            
            print(f"? PyTorch 모델 생성 완료:")
            print(f"   ??  좌손 모델: {len(lhmodel) if lhmodel else 0}개 관절")
            print(f"   ??  우손 모델: {len(rhmodel) if rhmodel else 0}개 관절")
            print(f"   ??  소요 시간: {model_time:.2f}초")
            
            # 2단계: 깊이 계산
            monitor.update_step_progress(2, f"PyTorch 벡터화 연산으로 {total_hands:,}개 손 깊이 계산 중...")
            
            print(f"\n? PyTorch 깊이 계산 시작:")
            print(f"   ? 해법 방식: 고정점 반복법 (Fixed-point Iteration)")
            print(f"   ? 비선형 방정식 시스템: 3변수 3방정식 (벡터화)")
            print(f"   ? 허용 오차: 1e-6 (성능 최적화)")
            print(f"   ??  처리 방식: GPU 배치 병렬 처리")
            
            start_depth = time.time()
            pytorch_version.depthlist(handlist_copy, lhmodel, rhmodel, ratio)
            depth_time = time.time() - start_depth
            
            print(f"? PyTorch 깊이 계산 완료:")
            print(f"   ??  총 소요 시간: {depth_time:.2f}초")
            print(f"   ? 평균 처리 속도: {total_hands/depth_time:.1f}개/초")
            
            # 3단계: 결함 프레임 검출
            monitor.update_step_progress(3, "결함 프레임 분석 중...")
            
            print(f"? 결함 프레임 검출:")
            defect_start = time.time()
            defective_frames = pytorch_version.faultyframes(handlist_copy)
            defect_time = time.time() - defect_start
            
            print(f"? 결함 프레임 검출 완료:")
            print(f"   ? 결함 프레임: {len(defective_frames)}개")
            print(f"   ??  소요 시간: {defect_time:.2f}초")
            
            # 4단계: Floating 판정
            monitor.update_step_progress(4, "Floating 상태 판정 중...")
            
            print(f"? Floating 상태 판정:")
            floating_start = time.time()
            
            # 임계값 설정
            if self.auto_threshold:
                threshold = self.calculate_auto_threshold(handlist_copy)
                print(f"   ? 자동 임계값: 좌손={threshold['Left']:.3f}, 우손={threshold['Right']:.3f}")
            else:
                threshold = self.fixed_threshold
                print(f"   ? 고정 임계값: {threshold}")
            
            # 프레임 수 계산
            frame_count = len(handlist_copy)
            
            floating_frames = pytorch_version.detectfloatingframes(
                handlist_copy, frame_count, defective_frames, lhmodel, rhmodel, ratio, threshold
            )
            floating_time = time.time() - floating_start
            
            print(f"? Floating 판정 완료:")
            print(f"   ? Floating 감지: {len(floating_frames)}개")
            print(f"   ??  소요 시간: {floating_time:.2f}초")
            
            monitor.finish_step(f"총 {len(floating_frames)}개 floating 감지")
            
        except Exception as e:
            print(f"? PyTorch 처리 중 오류 발생: {e}")
            raise
        
        pytorch_time = time.time() - start_time
        print(f"\n? PyTorch 전체 처리 완료:")
        print(f"   ??  총 소요 시간: {pytorch_time:.2f}초")
        print(f"   ? 전체 처리량: {total_hands/pytorch_time:.1f}개/초")
        
        return floating_frames, pytorch_time, handlist_copy  # 업데이트된 handlist_copy도 반환
    
    def calculate_auto_threshold(self, handlist: List) -> Dict[str, float]:
        """자동 임계값 계산"""
        if self.threshold_method == 'midi_based' and midicomparison:
            return self.calculate_midi_based_threshold(handlist)
        elif self.threshold_method == 'valley':
            return self.calculate_valley_threshold(handlist)
        else:
            # 기본값 반환
            return {'Left': self.fixed_threshold, 'Right': self.fixed_threshold}
    
    def calculate_midi_based_threshold(self, handlist: List) -> Dict[str, float]:
        """MIDI 데이터 기반 임계값 계산"""
        try:
            # MIDI 기반 임계값 계산 로직 (구현 필요)
            # 현재는 기본값 반환
            return {'Left': self.fixed_threshold, 'Right': self.fixed_threshold}
        except Exception as e:
            print(f"??  MIDI 기반 임계값 계산 실패: {e}")
            return {'Left': self.fixed_threshold, 'Right': self.fixed_threshold}
    
    def calculate_valley_threshold(self, handlist: List) -> Dict[str, float]:
        """Valley detection 기반 임계값 계산"""
        try:
            # Valley detection 로직 (구현 필요)
            # 현재는 기본값 반환
            return {'Left': self.fixed_threshold, 'Right': self.fixed_threshold}
        except Exception as e:
            print(f"??  Valley 기반 임계값 계산 실패: {e}")
            return {'Left': self.fixed_threshold, 'Right': self.fixed_threshold}
    
    def analyze_results(self, floating_results: List, processing_time: float, handlist: List) -> Dict[str, Any]:
        """결과 분석"""
        print("? 결과 분석 중...")
        
        # 기본 통계
        total_hands = sum(len(hands) for hands in handlist if hands)
        total_frames = len([hands for hands in handlist if hands])
        floating_count = len(floating_results)
        
        # 손 타입별 분석
        left_floating = len([f for f in floating_results if 'Left' in str(f)])
        right_floating = len([f for f in floating_results if 'Right' in str(f)])
        
        # 깊이 값 분석
        depth_values = []
        for hands in handlist:
            if hands:
                for hand in hands:
                    if hasattr(hand, 'depth') and hand.depth is not None:
                        depth_values.append(hand.depth)
        
        analysis = {
            'total_hands': total_hands,
            'total_frames': total_frames,
            'floating_count': floating_count,
            'left_floating': left_floating,
            'right_floating': right_floating,
            'processing_time': processing_time,
            'hands_per_second': total_hands / processing_time if processing_time > 0 else 0,
            'depth_stats': {
                'mean': np.mean(depth_values) if depth_values else 0,
                'std': np.std(depth_values) if depth_values else 0,
                'min': np.min(depth_values) if depth_values else 0,
                'max': np.max(depth_values) if depth_values else 0
            }
        }
        
        return analysis
    
    def create_result_video(self, data_info: Dict[str, Any], floating_results: List, handlist: List) -> str:
        """결과 시각화 비디오 생성"""
        monitor.start_step("결과 비디오 생성", 1)
        
        video_path = data_info['video_file']
        output_path = os.path.join(self.script_dir, f"{self.current_video}_pytorch_result.mp4")
        
        print(f"? 결과 비디오 생성:")
        print(f"   ? 입력: {video_path}")
        print(f"   ? 출력: {output_path}")
        
        try:
            # 비디오 캡처 및 라이터 설정
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Quick test인 경우 프레임 제한
            if self.quick_test:
                total_frames = min(total_frames, self.frame_limit)
            
            # 비디오 라이터 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print(f"   ? 해상도: {width}x{height}")
            print(f"   ? FPS: {fps}")
            print(f"   ??  총 프레임: {total_frames}")
            
            # floating 결과를 프레임별로 정리
            floating_by_frame = {}
            for result in floating_results:
                # floating 결과 파싱: [frame, handtype, metric_value, 'floating'/'notfloating']
                frame_num = result[0]  # frame
                hand_type = result[1]  # handtype
                metric_value = result[2]  # metric_value
                floating_status = result[3]  # 'floating' or 'notfloating'
                
                if frame_num not in floating_by_frame:
                    floating_by_frame[frame_num] = {}
                
                floating_by_frame[frame_num][f'{hand_type}_floating'] = (floating_status == 'floating')
                floating_by_frame[frame_num][f'{hand_type}_depth'] = metric_value
            
            # MediaPipe 초기화
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or (self.quick_test and frame_count >= self.frame_limit):
                        break
                    
                    # 진행 상황 표시
                    if frame_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"   ? 진행: {frame_count}/{total_frames} ({progress:.1f}%)")
                    
                    # 손 감지
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    
                    # 현재 프레임의 floating 상태 가져오기
                    current_floating = floating_by_frame.get(frame_count, {})
                    current_depth = {
                        'Left_depth': current_floating.get('Left_depth', 0.0),
                        'Right_depth': current_floating.get('Right_depth', 0.0)
                    }
                    
                    # 시각화 적용
                    if results.multi_hand_landmarks:
                        annotated_frame = draw_hand_landmarks_with_floating(
                            frame, 
                            results.multi_hand_landmarks,
                            results.multi_handedness,
                            current_floating,
                            current_depth,
                            self.cached_threshold
                        )
                    else:
                        annotated_frame = frame.copy()
                    
                    # 헤더 정보 추가
                    annotated_frame = draw_header_info(
                        annotated_frame,
                        frame_count + 1,
                        current_floating,
                        current_depth,
                        total_frames
                    )
                    
                    out.write(annotated_frame)
                    frame_count += 1
            
            cap.release()
            out.release()
            
            print(f"? 비디오 생성 완료: {output_path}")
            monitor.finish_step(f"비디오 생성 완료: {frame_count}프레임")
            
            return output_path
            
        except Exception as e:
            print(f"? 비디오 생성 실패: {e}")
            return ""
    
    def run_detection(self) -> Dict[str, Any]:
        """전체 detection 실행"""
        mode_text = f"({self.frame_limit}프레임 제한)" if self.quick_test else "(전체 프레임)"
        print(f"? {self.target_video} PyTorch Detection 시작 {mode_text}")
        print("=" * 60)
        
        # 모니터링 시작
        monitor.start_monitoring(total_steps=4)
        
        try:
            # 1. 데이터 찾기 및 로드
            monitor.start_step("데이터 로드 및 초기화", 3)
            
            print("? 데이터 검색 중...")
            data_info = self.find_target_data()
            data_location = data_info.get('data_dir', f"Raw 비디오: {data_info['video_file']}")
            monitor.update_step_progress(1, f"데이터 위치: {data_location}")
            
            print("? 손 데이터 및 비디오 정보 로딩 중...")
            handlist, existing_floating, ratio = self.load_data(data_info)
            monitor.update_step_progress(2, f"handlist: {len(handlist)}프레임 로딩 완료")
            
            actual_frame_count = len([hand for hands in handlist if hands for hand in hands])
            total_hands = sum(len(hands) for hands in handlist if hands)
            
            monitor.update_step_progress(3, f"분석 준비: {total_hands:,}개 손 데이터")
            monitor.finish_step(f"총 {total_hands:,}개 손, {len(handlist)}프레임 로딩 완료")
            
            # 2. PyTorch Detection 실행
            print(f"\n? PyTorch Floating Detection 시작...")
            floating_results, processing_time, updated_handlist = self.run_pytorch_detection(handlist, ratio)
            
            # 3. 결과 분석
            monitor.start_step("결과 분석", 1)
            analysis = self.analyze_results(floating_results, processing_time, updated_handlist)
            monitor.finish_step(f"분석 완료: {analysis['floating_count']}개 floating 감지")
            
            # 4. 깊이 데이터 저장 (선택적)
            if self.save_depth_data:
                monitor.start_step("깊이 데이터 저장", 1)
                depth_data_path = save_depth_data(updated_handlist, self.current_video, self.depth_data_dir)
                analysis['depth_data_path'] = depth_data_path
                monitor.finish_step("깊이 데이터 저장 완료")
            
            # 5. 결과 비디오 생성
            video_path = ""
            if self.generate_video:
                video_path = self.create_result_video(data_info, floating_results, updated_handlist)
                analysis['result_video'] = video_path
            
            analysis['floating_results'] = floating_results
            analysis['video_info'] = data_info
            
            return analysis
            
        finally:
            # 모니터링 종료
            monitor.stop_monitoring()
    
    def check_video_resolutions(self):
        """모든 대상 비디오의 해상도를 체크합니다"""
        print(f"\n? 대상 비디오 해상도 체크:")
        print("=" * 60)
        
        resolutions = {}
        for video_name in self.target_videos:
            video_file = os.path.join(self.video_capture_dir, f"{video_name}.mp4")
            
            if os.path.exists(video_file):
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    ratio = height / width
                    duration = total_frames / fps if fps > 0 else 0
                    
                    resolutions[video_name] = {
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'total_frames': total_frames,
                        'ratio': ratio,
                        'duration': duration
                    }
                    
                    print(f"? {video_name}:")
                    print(f"   해상도: {width}x{height} (비율: {ratio:.3f})")
                    print(f"   FPS: {fps}, 총 프레임: {total_frames:,}개")
                    print(f"   길이: {duration:.1f}초 ({duration/60:.1f}분)")
                    
                    cap.release()
                else:
                    print(f"? {video_name}: 비디오 열기 실패")
            else:
                print(f"? {video_name}: 파일 없음")
        
        # 해상도 통계
        if resolutions:
            print(f"\n? 해상도 통계:")
            unique_resolutions = set((info['width'], info['height']) for info in resolutions.values())
            print(f"   고유 해상도: {len(unique_resolutions)}가지")
            for width, height in sorted(unique_resolutions):
                count = sum(1 for info in resolutions.values() if info['width'] == width and info['height'] == height)
                print(f"   {width}x{height}: {count}개 비디오")
            
            avg_fps = sum(info['fps'] for info in resolutions.values()) / len(resolutions)
            total_duration = sum(info['duration'] for info in resolutions.values())
            print(f"   평균 FPS: {avg_fps:.1f}")
            print(f"   총 길이: {total_duration:.1f}초 ({total_duration/60:.1f}분)")
        
        return resolutions

    def extract_depth_data_only(self, video_name: str = None) -> str:
        """깊이 데이터만 추출하고 저장하는 함수 (분석 없이)"""
        if video_name:
            self.set_current_video(video_name)
        elif not self.current_video:
            if self.target_videos:
                self.set_current_video(self.target_videos[0])
            else:
                print("⚠️ 분석할 비디오가 없습니다.")
                return ""
        
        print(f"💾 {self.current_video} 깊이 데이터 추출 모드")
        print("=" * 60)
        
        try:
            # 1. 데이터 로드
            print("📁 데이터 로딩 중...")
            data_info = self.find_target_data()
            handlist, existing_floating, ratio = self.load_data(data_info)
            
            # 2. PyTorch 깊이 계산만 실행
            print("🔬 PyTorch 깊이 계산 중...")
            
            # handlist 복사
            import copy
            handlist_copy = copy.deepcopy(handlist)
            
            # 모델 생성
            lhmodel, rhmodel = pytorch_version.modelskeleton(handlist_copy)
            
            # 깊이 계산
            pytorch_version.depthlist(handlist_copy, lhmodel, rhmodel, ratio)
            
            # 3. 깊이 데이터 저장
            print("💾 깊이 데이터 저장 중...")
            depth_data_path = save_depth_data(handlist_copy, self.current_video, self.depth_data_dir)
            
            print(f"✅ 깊이 데이터 추출 완료: {depth_data_path}")
            return depth_data_path
            
        except Exception as e:
            print(f"❌ 깊이 데이터 추출 실패: {e}")
            return ""
    
    def batch_depth_extraction(self) -> Dict[str, str]:
        """모든 비디오에 대해 깊이 데이터만 추출"""
        print(f"💾 일괄 깊이 데이터 추출 시작 (총 {len(self.target_videos)}개 비디오)")
        print("📋 목적: 분석용 깊이 데이터 수집")
        print("=" * 60)
        
        results = {}
        
        for i, video_name in enumerate(self.target_videos, 1):
            print(f"\n[{i}/{len(self.target_videos)}] {video_name} 데이터 추출 중...")
            
            try:
                depth_data_path = self.extract_depth_data_only(video_name)
                results[video_name] = {
                    'depth_data_path': depth_data_path,
                    'status': 'success' if depth_data_path else 'failed'
                }
                    
            except Exception as e:
                print(f"❌ {video_name} 실패: {e}")
                results[video_name] = {
                    'depth_data_path': '',
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 결과 요약
        successful = [v for v in results.values() if v['status'] == 'success']
        failed = [v for v in results.values() if v['status'] == 'failed']
        
        print(f"\n{'='*60}")
        print(f"💾 일괄 깊이 데이터 추출 완료!")
        print(f"   성공: {len(successful)}/{len(self.target_videos)}개")
        print(f"   실패: {len(failed)}개")
        print(f"   저장 위치: {self.depth_data_dir}")
        
        if successful:
            print(f"\n✅ 성공한 비디오:")
            for video_name, result in results.items():
                if result['status'] == 'success':
                    print(f"   💾 {video_name}: {os.path.basename(result['depth_data_path'])}")
        
        if failed:
            print(f"\n❌ 실패한 비디오:")
            for video_name, result in results.items():
                if result['status'] == 'failed':
                    print(f"   ⚠️ {video_name}: {result.get('error', '알 수 없는 오류')}")
        
        return results

    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        mode_text = f" (Quick Test: {self.frame_limit}프레임)" if self.quick_test else ""
        print(f"? {self.target_video} PyTorch Detection 결과{mode_text}")
        print("=" * 60)
        
        print(f"\n??  성능:")
        print(f"   처리 시간: {results['processing_time']:.2f}초 (GPU)")
        print(f"   처리 속도: {results['hands_per_second']:.1f}개/초")
        
        print(f"\n? Detection 결과:")
        print(f"   총 손 데이터: {results['total_hands']:,}개")
        print(f"   총 프레임: {results['total_frames']:,}개")
        print(f"   Floating 감지: {results['floating_count']}개")
        print(f"   좌손 floating: {results['left_floating']}개")
        print(f"   우손 floating: {results['right_floating']}개")
        
        print(f"\n? 깊이 통계:")
        depth_stats = results['depth_stats']
        print(f"   평균 깊이: {depth_stats['mean']:.3f}")
        print(f"   표준편차: {depth_stats['std']:.3f}")
        print(f"   최소값: {depth_stats['min']:.3f}")
        print(f"   최대값: {depth_stats['max']:.3f}")
        
        if results.get('result_video'):
            print(f"\n? 결과 비디오: {results['result_video']}")
            print(f"   ??  Hand landmarks 및 floating 상태 시각화")
            print(f"   ? 임계값: {self.fixed_threshold}")
        
        if results.get('depth_data_path'):
            print(f"\n💾 깊이 데이터: {results['depth_data_path']}")
            print(f"   📊 별도 분석 스크립트에서 활용 가능")

def process_multiple_videos():
    """여러 비디오를 순차적으로 처리하는 메인 함수 (통합 모드)"""
    detector = TorchFloatingDetector()
    
    if not detector.target_videos:
        print("? 처리할 비디오가 없습니다.")
        return
    
    print(f"🎯 통합 처리 실행:")
    print(f"   ✅ Floating Detection (임계값: {detector.fixed_threshold})")
    print(f"   💾 깊이 데이터 저장")
    print(f"   📹 결과 비디오 생성")
    print(f"   📊 최대 {len(detector.target_videos)}개 비디오 처리")
    print("")
    
    # 비디오 해상도 체크
    resolutions = detector.check_video_resolutions()
    
    all_results = []
    
    for i, video_name in enumerate(detector.target_videos, 1):
        print(f"\n{'='*80}")
        print(f"? [{i}/{len(detector.target_videos)}] 비디오 처리 중: {video_name}")
        print(f"   임계값: {detector.fixed_threshold}")
        print(f"{'='*80}")
        
        # 현재 비디오 설정
        detector.set_current_video(video_name)
        
        try:
            # Detection 실행
            results = detector.run_detection()
            
            if results:
                all_results.append({
                    'video_name': video_name,
                    'results': results
                })
                
                # 개별 결과 요약 출력
                print(f"\n? [{i}/{len(detector.target_videos)}] 완료: {video_name}")
                detector.print_summary(results)
                
        except Exception as e:
            print(f"? [{i}/{len(detector.target_videos)}] 실패: {video_name}")
            print(f"   오류: {str(e)}")
            continue
    
    # 전체 결과 요약
    print(f"\n{'='*80}")
    print(f"🎯 통합 모드 전체 처리 완료!")
    print(f"   ✅ 성공: {len(all_results)}/{len(detector.target_videos)}개")
    print(f"   🎯 임계값: {detector.fixed_threshold}")
    print(f"{'='*80}")
    
    for result in all_results:
        video_name = result['video_name']
        res = result['results']
        print(f"\n📊 {video_name}:")
        print(f"   ⏱️  처리 시간: {res.get('processing_time', 0):.2f}초")
        print(f"   🎯 Floating 감지: {res.get('floating_count', 0)}개")
        print(f"   ⚡ 처리 속도: {res.get('hands_per_second', 0):.1f}개/초")
        
        if res.get('depth_data_path'):
            print(f"   💾 깊이 데이터: 저장됨")
        
        if res.get('result_video'):
            print(f"   📹 결과 비디오: 생성됨")

def check_video_resolutions_only():
    """비디오 해상도만 체크하는 함수"""
    detector = TorchFloatingDetector()
    
    if not detector.target_videos:
        print("? 처리할 비디오가 없습니다.")
        return
    
    print("? PyTorch Floating Hand Detection - 비디오 해상도 체크")
    print("=" * 60)
    
    # 해상도 체크
    resolutions = detector.check_video_resolutions()
    
    return resolutions

def extract_depth_data():
    """깊이 데이터만 추출하여 저장하는 함수 (임계값 설정을 위한 데이터 수집)"""
    detector = TorchFloatingDetector()
    
    if not detector.target_videos:
        print("? 처리할 비디오가 없습니다.")
        return
    
    print("💾 PyTorch Floating Hand Detection - 깊이 데이터 추출")
    print("=" * 60)
    print("📋 목적: 거리 데이터 수집 (분석은 별도 스크립트에서)")
    print("💾 출력: JSON 형태의 깊이 데이터 파일")
    print("=" * 60)
    
    # 깊이 데이터 추출 실행
    results = detector.batch_depth_extraction()
    
    print(f"\n💾 깊이 데이터 추출 완료!")
    print(f"💡 다음 단계:")
    print(f"   1. 별도의 분석 스크립트로 데이터 분포를 분석하세요")
    print(f"   2. 히스토그램과 통계를 확인하여 적절한 임계값을 찾으세요")
    print(f"   3. 코드의 FIXED_THRESHOLD 값을 업데이트하세요")
    
    return results

# 전역 모니터 인스턴스
monitor = PerformanceMonitor()

if __name__ == "__main__":
    print("🔬 PyTorch Floating Hand Detection")
    print("=" * 60)
    
    # 통합 모드 자동 실행
    print(f"📊 현재 설정: 최대 {TorchFloatingDetector.MAX_VIDEOS}개 비디오, 임계값 {TorchFloatingDetector.FIXED_THRESHOLD}")
    print("\n🎯 통합 모드: Floating Detection + 깊이값 저장 + 결과 비디오 생성")
    print("=" * 60)
    
    try:
        process_multiple_videos()
    except KeyboardInterrupt:
        print("\n\n❌ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")
        import traceback
        traceback.print_exc() 
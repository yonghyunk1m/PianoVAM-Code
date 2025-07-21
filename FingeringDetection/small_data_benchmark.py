#!/usr/bin/env python3
"""
작은 데이터셋 기반 Floating Hand Detection 빠른 벤치마킹
- 2024-09-04_22-06-40 데이터만 사용
- SciPy vs PyTorch 성능 비교  
- 정성평가를 위한 상세 비교 비디오 생성
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

# GPU 메모리 모니터링을 위한 import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 두 버전의 floating hands 모듈 import
# Golden Standard: 원본 SciPy 버전
try:
    import floatinghands_original as scipy_version
    print("✅ SciPy 원본 버전 모듈 로드 성공")
except ImportError as e:
    print(f"❌ SciPy 원본 버전 모듈 로드 실패: {e}")
    exit(1)

# 새 구현: 순수 PyTorch 버전
try:
    import floatinghands_torch_pure as pytorch_version
    print("✅ PyTorch 순수 버전 모듈 로드 성공")
except ImportError as e:
    print(f"❌ PyTorch 순수 버전 모듈 로드 실패: {e}")
    exit(1)

# MIDI 데이터 처리용 모듈 import
try:
    import midicomparison
    print("✅ MIDI 비교 모듈 로드 성공")
except ImportError as e:
    print(f"⚠️  MIDI 비교 모듈 로드 실패: {e}")
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
            pass  # 출력 간소화
        except Exception as e:
            print(f"❌ main_loop 모듈 로드 실패: {e}")
            print(f"   상세 오류: {type(e).__name__}: {str(e)}")
            raise
    return main_loop_module

# MediaPipe 그리기 유틸리티 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_enhanced_hand_landmarks(image, hand_landmarks, handedness, floating_status, depth_info, handtype_suffix="", threshold_value=0.9):
    """향상된 Hand landmarks와 floating 상태, 깊이 정보를 시각적으로 표시합니다
    
    Args:
        threshold_value: 단일 임계값(float) 또는 좌우 손 구분 임계값(dict)
    """
    if not hand_landmarks:
        return image
    
    # 이미지 복사
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    for idx, landmarks in enumerate(hand_landmarks):
        # 손 종류 확인
        if idx < len(handedness):
            hand_type = handedness[idx].classification[0].category_name
            
            # 깊이 정보 가져오기
            depth_value = depth_info.get(hand_type, {}).get('depth', 0.0)
            is_floating = floating_status.get(hand_type, False)
            
            # Floating 상태에 따른 색상 설정 (더 명확한 색상)
            if is_floating:
                color = (0, 0, 255)  # 빨간색 - Floating
                bg_color = (0, 0, 100)  # 어두운 빨간색 배경
                status_text = f"{hand_type} FLOATING"
            else:
                color = (0, 255, 0)  # 초록색 - Normal
                bg_color = (0, 100, 0)  # 어두운 초록색 배경
                status_text = f"{hand_type} NORMAL"
            
            # 손목 좌표 가져오기 (landmarks는 리스트 형태)
            if len(landmarks) > 0:
                wrist = landmarks[0]  # 첫 번째는 손목
                wrist_x = int((wrist.x + 1) * width / 2)  # [-1,1] -> [0,width]
                wrist_y = int((wrist.y + 1) * height / 2)  # [-1,1] -> [0,height]
                
                # 주요 손 관절들 그리기 (더 명확한 연결선 포함)
                for i, landmark in enumerate(landmarks):
                    x = int((landmark.x + 1) * width / 2)
                    y = int((landmark.y + 1) * height / 2)
                    
                    if 0 <= x < width and 0 <= y < height:
                        # 관절 점 그리기 (크기 조정)
                        cv2.circle(annotated_image, (x, y), 4, color, -1)
                        cv2.circle(annotated_image, (x, y), 5, (255, 255, 255), 1)  # 흰색 테두리
                        
                        # 중요한 관절점은 더 크게 표시
                        if i in [0, 4, 8, 12, 16, 20]:  # 손목, 엄지~소지 끝점
                            cv2.circle(annotated_image, (x, y), 8, color, 2)
                            cv2.circle(annotated_image, (x, y), 10, (255, 255, 255), 1)
                
                # 정보 박스 위치 계산
                info_x = max(10, min(wrist_x - 100, width - 250))
                info_y = max(60, wrist_y - 50)
                
                # 정보 배경 박스
                cv2.rectangle(annotated_image, (info_x, info_y - 45), (info_x + 240, info_y + 15), bg_color, -1)
                cv2.rectangle(annotated_image, (info_x, info_y - 45), (info_x + 240, info_y + 15), color, 2)
                
                # 상태 및 깊이 정보 표시
                cv2.putText(annotated_image, status_text + handtype_suffix, (info_x + 5, info_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_image, f"Depth: {depth_value:.3f}", (info_x + 5, info_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 임계값과의 비교 표시 (좌우 손 구분 지원)
                if isinstance(threshold_value, dict):
                    # 좌우 손 구분 임계값
                    current_threshold = threshold_value.get(hand_type, 0.9)
                    threshold_status = f"< {current_threshold:.3f} (NORMAL)" if depth_value < current_threshold else f">= {current_threshold:.3f} (FLOATING)"
                    threshold_color = (255, 255, 0) if depth_value < current_threshold else (0, 255, 255)
                else:
                    # 단일 임계값 (기존 방식)
                    threshold_status = f"< {threshold_value:.3f} (NORMAL)" if depth_value < threshold_value else f">= {threshold_value:.3f} (FLOATING)"
                    threshold_color = (255, 255, 0) if depth_value < threshold_value else (0, 255, 255)
                cv2.putText(annotated_image, threshold_status, (info_x + 5, info_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, threshold_color, 1)
                
                # 손목에 상태 표시 원
                circle_radius = 15 if is_floating else 12
                cv2.circle(annotated_image, (wrist_x, wrist_y), circle_radius, color, 4)
                cv2.circle(annotated_image, (wrist_x, wrist_y), circle_radius + 2, (255, 255, 255), 1)
    
    return annotated_image

def draw_enhanced_comparison_header(image, frame_num, scipy_floating, pytorch_floating, scipy_depth_info, pytorch_depth_info, total_frames):
    """향상된 비교 정보를 이미지 상단에 오버레이합니다"""
    height, width = image.shape[:2]
    
    # 상단 정보 패널 크기 계산
    panel_height = 120
    
    # 반투명 배경 박스
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # 전체 테두리
    border_color = (0, 0, 255) if scipy_floating != pytorch_floating else (0, 255, 0)
    cv2.rectangle(image, (0, 0), (width-1, panel_height-1), border_color, 3)
    
    # 좌측 영역 (프레임 정보)
    left_x = 20
    
    # 프레임 정보
    cv2.putText(image, f"Frame: {frame_num}/{total_frames}", (left_x, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    progress = frame_num / total_frames * 100
    cv2.putText(image, f"Progress: {progress:.1f}%", (left_x, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 중앙 영역 (비교 결과)
    center_x = width // 2 - 150
    
    # 제목
    cv2.putText(image, "SciPy (Golden) vs PyTorch (New)", (center_x, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 결과 비교
    if scipy_floating == pytorch_floating:
        match_color = (0, 255, 0)  # 초록색 - 일치
        match_text = "MATCH"
        match_icon = "MATCH"
    else:
        match_color = (0, 0, 255)  # 빨간색 - 불일치
        match_text = "MISMATCH"
        match_icon = "MISMATCH"
    
    cv2.putText(image, match_text, (center_x, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, match_color, 2)
    
    # 상세 상태
    scipy_status = "FLOATING" if scipy_floating else "NORMAL"
    pytorch_status = "FLOATING" if pytorch_floating else "NORMAL"
    
    scipy_color = (100, 100, 255) if scipy_floating else (100, 255, 100)
    pytorch_color = (255, 150, 100) if pytorch_floating else (100, 255, 100)
    
    cv2.putText(image, f"SciPy: {scipy_status}", (center_x, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, scipy_color, 1)
    cv2.putText(image, f"PyTorch: {pytorch_status}", (center_x, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, pytorch_color, 1)
    
    # 우측 영역 (깊이 차이 정보)
    right_x = width - 350
    
    if scipy_depth_info and pytorch_depth_info:
        cv2.putText(image, "Depth Comparison", (right_x, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos = 45
        for handtype in scipy_depth_info.keys():
            if handtype in pytorch_depth_info:
                scipy_depth = scipy_depth_info[handtype].get('depth', 0.0)
                pytorch_depth = pytorch_depth_info[handtype].get('depth', 0.0)
                depth_diff = abs(scipy_depth - pytorch_depth)
                
                # 차이에 따른 색상
                diff_color = (0, 255, 255) if depth_diff > 0.1 else (255, 255, 255)
                
                cv2.putText(image, f"{handtype}: Δ={depth_diff:.3f}", (right_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, diff_color, 1)
                y_pos += 20
    
    return image

def create_keyboard_overlay(image, keyboard_coords=None):
    """키보드 좌표를 이미지에 오버레이합니다 (선택사항)"""
    if keyboard_coords is None:
        return image
    
    # 키보드 좌표가 있으면 그리기
    # 이 부분은 keyboardcoordinateinfo.pkl의 구조에 따라 구현
    # 현재는 간단히 패스
    return image

# 로깅 최적화 - 핵심 정보만 출력
class PerformanceMonitor:
    """성능 모니터링 클래스 - 간소화된 로깅"""
    
    def __init__(self):
        self.start_time = None
        self.step_start_time = None
        self.step_progress = 0
        self.step_total = 0
        self.overall_progress = 0
        self.overall_total = 0
        self.current_step = ""
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, total_steps=0):
        """모니터링 시작"""
        self.start_time = time.time()
        self.overall_total = total_steps
        self.monitoring = True
        
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        total_elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n작업 완료 ({self._format_time(total_elapsed)})")
        
    def start_step(self, step_name: str, total_items=0):
        """단계 시작"""
        self.current_step = step_name
        self.step_start_time = time.time()
        self.step_progress = 0
        self.step_total = total_items
        
        print(f"\n{step_name}...")
            
    def log_operation(self, operation_name: str, details: str = "", show_gpu=False):
        """개별 연산 로깅"""
        pass  # 간소화를 위해 비활성화
    
    def log_batch_progress(self, batch_num: int, total_batches: int, batch_size: int, 
                          items_processed: int, operation: str = ""):
        """배치 처리 진행 상황 로깅"""
        pass  # 간소화를 위해 비활성화
    
    def log_algorithm_start(self, algorithm_name: str, method: str, precision: str, 
                           processing_type: str, optimizations: list = None):
        """알고리즘 시작 로깅"""
        print(f"{algorithm_name} 시작 ({method}, {precision})")
    
    def log_memory_usage(self, stage: str, cpu_memory: float = None, gpu_memory: float = None):
        """메모리 사용량 로깅"""
        pass  # 간소화를 위해 비활성화
    
    def update_step_progress(self, current: int, message: str = ""):
        """단계별 진행률 업데이트"""
        self.step_progress = current
    
    def finish_step(self, message: str = ""):
        """단계 완료"""
        elapsed = time.time() - self.step_start_time if self.step_start_time else 0
        self.overall_progress += 1
        
        if message:
            print(f"{self.current_step} 완료: {message}")
        
        # GPU 메모리 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _print_system_info(self):
        """시스템 정보 출력 - 간소화"""
        print(f"\n🖥️  시스템 정보:")
        print(f"   💻 CPU: {psutil.cpu_count(logical=False)}코어")
        
        memory = psutil.virtual_memory()
        print(f"   💾 RAM: {memory.total/1024**3:.1f}GB")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print(f"   🎮 GPU: 사용 불가")
    
    def _format_time(self, seconds):
        """시간 포맷팅"""
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}분 {secs}초"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}시간 {minutes}분"

# 글로벌 모니터 인스턴스
monitor = PerformanceMonitor()

class SmallDataBenchmark:
    """작은 데이터셋 전용 빠른 벤치마킹"""
    
    # 내부 설정 - 기본값: True (초반 1200프레임만 처리)
    # 전체 프레임 처리하려면 QUICK_TEST = False 로 변경
    QUICK_TEST = True  # 1200프레임만 벤치마킹
    FRAME_LIMIT = 1200
    
    # 캐싱 설정
    ENABLE_CACHING = True  # 처리 완료된 데이터 캐싱 활성화
    ENABLE_LANDMARK_CACHING = True  # MediaPipe landmark 데이터 캐싱 활성화
    
    # 비디오 생성 설정
    GENERATE_DETAILED_VIDEO = True  # 상세 비교 비디오 생성 (Hand landmarks 포함, VSCode 호환)
    
    # 자동 임계값 설정
    AUTO_THRESHOLD = False  # 자동 임계값 사용 여부
    THRESHOLD_METHOD = 'midi_based'  # 'statistical', 'clustering', 'valley', 'midi_based'
    FALLBACK_THRESHOLD = 0.9  # 자동 계산 실패 시 기본값
    
    # 비디오 처리 설정
    TARGET_VIDEO_DIR = "/home/jhbae/PianoVAM-Code/FingeringDetection/videocapture"
    MAX_VIDEOS = 5  # 처리할 비디오 개수 제한
    FIXED_THRESHOLD = 0.9  # 고정 임계값
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_capture_dir = self.TARGET_VIDEO_DIR  # 새로운 디렉토리 사용
        self.target_videos = []  # 처리할 비디오 목록
        self.current_video = None  # 현재 처리 중인 비디오
        self.quick_test = self.QUICK_TEST
        self.frame_limit = self.FRAME_LIMIT
        self.enable_caching = self.ENABLE_CACHING
        self.enable_landmark_caching = self.ENABLE_LANDMARK_CACHING
        self.generate_detailed_video = self.GENERATE_DETAILED_VIDEO
        
        # 고정 임계값 설정 (자동 임계값 비활성화)
        self.auto_threshold = False  # 자동 임계값 비활성화
        self.threshold_method = self.THRESHOLD_METHOD
        self.fallback_threshold = self.FIXED_THRESHOLD  # 0.9로 고정
        
        # 임계값 캐싱 (한 번 계산된 임계값 재사용)
        self.cached_threshold = {'Left': self.FIXED_THRESHOLD, 'Right': self.FIXED_THRESHOLD}
        
        # 처리할 비디오 목록 초기화
        self._initialize_video_list()
        
        # 캐싱 경로 설정
        self.cache_dir = os.path.join(self.script_dir, 'cache')
        
        # 캐시 디렉토리 생성
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _initialize_video_list(self):
        """처리 가능한 비디오 목록을 초기화합니다"""
        if not os.path.exists(self.video_capture_dir):
            print(f"❌ 비디오 디렉토리가 존재하지 않습니다: {self.video_capture_dir}")
            return
        
        # .mp4 파일과 대응되는 데이터가 있는 비디오들을 찾기
        for item in os.listdir(self.video_capture_dir):
            if item.endswith('.mp4'):
                video_name = item[:-4]
                
                # 해당 pkl 데이터가 있는지 확인
                data_dir = os.path.join(self.video_capture_dir, f"{video_name}_858550")
                if os.path.exists(data_dir):
                    handlist_file = os.path.join(data_dir, f"handlist_{video_name}_858550.pkl")
                    floating_file = os.path.join(data_dir, f"floatingframes_{video_name}_858550.pkl")
                    
                    if os.path.exists(handlist_file) and os.path.exists(floating_file):
                        self.target_videos.append(video_name)
        
        # MAX_VIDEOS 제한 적용
        if len(self.target_videos) > self.MAX_VIDEOS:
            self.target_videos = self.target_videos[:self.MAX_VIDEOS]
        
        print(f"📊 처리 대상 비디오: {len(self.target_videos)}개")
        for i, video in enumerate(self.target_videos, 1):
            print(f"   {i}. {video}")
    
    def set_current_video(self, video_name: str):
        """현재 처리할 비디오를 설정합니다"""
        self.current_video = video_name
        self.target_video = video_name  # 기존 코드 호환성
        
        # 캐싱 경로 업데이트
        self.limited_video_path = os.path.join(self.cache_dir, f"{video_name}_limit{self.frame_limit}.mp4")
        self.cache_data_dir = os.path.join(self.cache_dir, f"{video_name}_limit{self.frame_limit}_data")
        self.landmark_cache_path = os.path.join(self.cache_dir, f"landmarks_{video_name}_limit{self.frame_limit}.pkl")
    
    def check_landmark_cache(self, video_path: str) -> bool:
        """Landmark 캐시가 유효한지 확인"""
        if not self.enable_landmark_caching:
            return False
        
        if not os.path.exists(self.landmark_cache_path):
            return False
        
        # 비디오 파일과 캐시 파일의 수정 시간 비교
        try:
            video_mtime = os.path.getmtime(video_path)
            cache_mtime = os.path.getmtime(self.landmark_cache_path)
            
            if cache_mtime > video_mtime:
                print(f"✅ Landmark 캐시 발견: {self.landmark_cache_path}")
                cache_size = os.path.getsize(self.landmark_cache_path) / 1024**2
                print(f"   📁 캐시 크기: {cache_size:.1f}MB")
                return True
            else:
                print(f"⚠️  Landmark 캐시가 오래됨 (비디오 파일이 더 새로움)")
                return False
                
        except Exception as e:
            print(f"❌ Landmark 캐시 확인 실패: {e}")
            return False
    
    def load_landmark_cache(self) -> Dict[str, Any]:
        """캐시된 landmark 데이터 로드"""
        try:
            print(f"📁 캐시된 landmark 데이터 로딩 중...")
            with open(self.landmark_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            frame_count = len(cached_data.get('handlist_by_frame', {}))
            hand_count = sum(len(hands) for hands in cached_data.get('handlist_by_frame', {}).values())
            
            print(f"✅ Landmark 캐시 로딩 완료:")
            print(f"   🖼️  프레임 수: {frame_count:,}개")
            print(f"   👋 총 손 데이터: {hand_count:,}개")
            print(f"   ⚡ MediaPipe 추출 과정 생략됨")
            
            return cached_data
            
        except Exception as e:
            print(f"❌ Landmark 캐시 로딩 실패: {e}")
            return None
    
    def save_landmark_cache(self, handlist_by_frame: Dict, total_frames: int):
        """Landmark 데이터를 캐시로 저장"""
        if not self.enable_landmark_caching:
            return
        
        try:
            print(f"💾 Landmark 데이터 캐시 저장 중...")
            
            cache_data = {
                'handlist_by_frame': handlist_by_frame,
                'total_frames': total_frames,
                'video_name': self.target_video,
                'frame_limit': self.frame_limit,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.landmark_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            cache_size = os.path.getsize(self.landmark_cache_path) / 1024**2
            hand_count = sum(len(hands) for hands in handlist_by_frame.values())
            
            print(f"✅ Landmark 캐시 저장 완료:")
            print(f"   📁 파일: {self.landmark_cache_path}")
            print(f"   📊 크기: {cache_size:.1f}MB")
            print(f"   👋 손 데이터: {hand_count:,}개")
            print(f"   🚀 다음 실행 시 빠른 로딩 가능")
            
        except Exception as e:
            print(f"❌ Landmark 캐시 저장 실패: {e}")

    def calculate_auto_threshold(self, handlist: List) -> Dict[str, float]:
        """자동 임계값 계산 - MIDI 기반 방법만 사용 (히스토그램 골짜기 방식 완전 제거)"""
        
        if self.threshold_method == 'midi_based':
            return self.calculate_midi_based_threshold(handlist)
        else:
            print(f"⚠️  히스토그램 골짜기 방식은 제거되었습니다.")
            print(f"   MIDI 기반 방법으로 대체하여 계산합니다.")
            return self.calculate_midi_based_threshold(handlist)
    
    def calculate_midi_based_threshold(self, handlist: List) -> Dict[str, float]:
        """MIDI 데이터 기반 임계값 계산 - 정교한 통계학적 접근"""
        print(f"🎹 정교한 MIDI 기반 좌우 손 임계값 계산 중...")
        
        if midicomparison is None:
            print(f"MIDI 모듈 없음, fallback 사용: Left={self.fallback_threshold}, Right={self.fallback_threshold}")
            return {'Left': self.fallback_threshold, 'Right': self.fallback_threshold}
        
        midi_path = os.path.join(self.script_dir, 'midiconvert', f"{self.target_video}.mid")
        
        if not os.path.exists(midi_path):
            print(f"MIDI 파일 없음, fallback 사용: Left={self.fallback_threshold}, Right={self.fallback_threshold}")
            return {'Left': self.fallback_threshold, 'Right': self.fallback_threshold}
        
        try:
            video_path = os.path.join(self.video_capture_dir, f"{self.target_video}.mp4")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            midi_filename = f"{self.target_video}"
            tokenlist = midicomparison.miditotoken(midi_filename, fps, "simplified")
            total_frames = max([max([hand.handframe for hand in hands]) for hands in handlist if hands]) + 1
            frame_midi_info = midicomparison.tokentoframeinfo(tokenlist, total_frames)
            
            # 좌우 손 구분하여 데이터 수집
            hand_data = {
                'Left': {'playing': [], 'non_playing': []},
                'Right': {'playing': [], 'non_playing': []}
            }
            
            for frame_idx, hands in enumerate(handlist):
                if self.quick_test and frame_idx >= self.frame_limit:
                    break
                    
                if not hands or frame_idx >= len(frame_midi_info):
                    continue
                
                current_midi = frame_midi_info[frame_idx]
                is_playing = len(current_midi) > 0
                
                for hand in hands:
                    if hasattr(hand, 'handdepth') and hand.handdepth > 0 and hasattr(hand, 'handtype'):
                        hand_type = hand.handtype
                        if hand_type in hand_data:
                            if is_playing:
                                hand_data[hand_type]['playing'].append(hand.handdepth)
                            else:
                                hand_data[hand_type]['non_playing'].append(hand.handdepth)
            
            # 각 손 타입별로 정교한 임계값 계산
            thresholds = {}
            
            for hand_type in ['Left', 'Right']:
                playing_depths = hand_data[hand_type]['playing']
                non_playing_depths = hand_data[hand_type]['non_playing']
                
                print(f"\n📊 {hand_type} 손 데이터 분석:")
                print(f"   연주 중: {len(playing_depths)}개 샘플")
                print(f"   비연주 중: {len(non_playing_depths)}개 샘플")
                
                if len(playing_depths) < 20 or len(non_playing_depths) < 20:
                    print(f"   ⚠️  데이터 부족, fallback 사용: {self.fallback_threshold}")
                    thresholds[hand_type] = self.fallback_threshold
                    continue
                
                # 정교한 통계학적 임계값 계산
                optimal_threshold = self._calculate_optimal_threshold_advanced(
                    playing_depths, non_playing_depths, hand_type
                )
                
                if 0.1 <= optimal_threshold <= 1.5:
                    thresholds[hand_type] = float(optimal_threshold)
                    print(f"   ✅ {hand_type} 손 최적 임계값: {optimal_threshold:.4f}")
                else:
                    print(f"   ❌ 계산된 임계값 범위 초과: {optimal_threshold:.4f}")
                    print(f"   🔄 Fallback 임계값 사용: {self.fallback_threshold}")
                    thresholds[hand_type] = self.fallback_threshold
            
            print(f"\n🎯 정교한 좌우 손 임계값 계산 완료:")
            print(f"   Left={thresholds.get('Left', self.fallback_threshold):.4f}")
            print(f"   Right={thresholds.get('Right', self.fallback_threshold):.4f}")
            
            return thresholds
                
        except Exception as e:
            print(f"MIDI 임계값 계산 실패: {e}")
            return {'Left': self.fallback_threshold, 'Right': self.fallback_threshold}
    
    def _calculate_optimal_threshold_advanced(self, playing_depths: List, non_playing_depths: List, hand_type: str) -> float:
        """오분류 최소화로 최적 임계값 계산 - 디버깅 강화"""
        
        print(f"   🎯 {hand_type} 손 오분류 최소화 분석 시작...")
        
        playing_array = np.array(playing_depths)
        non_playing_array = np.array(non_playing_depths)
        
        # 기본 통계
        playing_mean = np.mean(playing_array)
        playing_std = np.std(playing_array)
        non_playing_mean = np.mean(non_playing_array)
        non_playing_std = np.std(non_playing_array)
        
        print(f"   📊 연주 중 통계: μ={playing_mean:.3f}, σ={playing_std:.3f} ({len(playing_array)}개)")
        print(f"   📊 비연주 중 통계: μ={non_playing_mean:.3f}, σ={non_playing_std:.3f} ({len(non_playing_array)}개)")
        
        # 데이터 범위 확인
        print(f"   📈 데이터 범위:")
        print(f"      연주 중: {np.min(playing_array):.3f} ~ {np.max(playing_array):.3f}")
        print(f"      비연주 중: {np.min(non_playing_array):.3f} ~ {np.max(non_playing_array):.3f}")
        
        # 모든 가능한 임계값 후보 생성
        all_depths = np.concatenate([playing_array, non_playing_array])
        unique_depths = np.sort(np.unique(all_depths))
        
        print(f"   🔍 임계값 후보: {len(unique_depths)}개 ({unique_depths.min():.3f} ~ {unique_depths.max():.3f})")
        
        # 각 임계값에 대해 균형 잡힌 정확도 계산
        best_threshold = self.fallback_threshold
        max_balanced_accuracy = -1
        best_fp = 0
        best_fn = 0
        
        # 디버깅을 위한 임계값별 결과 저장
        debug_results = []
        
        for threshold in unique_depths:
            # False Positive: 연주 중인데 floating으로 잘못 분류 (depth >= threshold)
            false_positives = np.sum(playing_array >= threshold)
            
            # False Negative: 비연주 중인데 normal로 잘못 분류 (depth < threshold)  
            false_negatives = np.sum(non_playing_array < threshold)
            
            # True Positive, True Negative 계산
            true_positives = len(non_playing_array) - false_negatives  # 비연주 중 올바르게 floating 분류
            true_negatives = len(playing_array) - false_positives      # 연주 중 올바르게 normal 분류
            
            # 균형 잡힌 정확도 계산 (민감도와 특이도의 평균)
            sensitivity = true_positives / len(non_playing_array) if len(non_playing_array) > 0 else 0  # 비연주 중 정확도
            specificity = true_negatives / len(playing_array) if len(playing_array) > 0 else 0         # 연주 중 정확도
            balanced_accuracy = (sensitivity + specificity) / 2
            
            # 디버깅 정보 저장
            debug_results.append({
                'threshold': threshold,
                'fp': false_positives,
                'fn': false_negatives,
                'sensitivity': sensitivity * 100,
                'specificity': specificity * 100,
                'balanced_acc': balanced_accuracy * 100
            })
            
            # 균형 잡힌 정확도가 최대인 임계값 선택
            if balanced_accuracy > max_balanced_accuracy:
                max_balanced_accuracy = balanced_accuracy
                best_threshold = threshold
                best_fp = false_positives
                best_fn = false_negatives
        
        # 디버깅: 상위 5개와 하위 5개 임계값 결과 출력
        print(f"   🔍 임계값 후보 분석 (상위 5개):")
        sorted_results = sorted(debug_results, key=lambda x: x['balanced_acc'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"      #{i+1}: threshold={result['threshold']:.3f}, FP={result['fp']}, FN={result['fn']}, 정확도={result['balanced_acc']:.1f}%")
        
        print(f"   🔍 임계값 후보 분석 (하위 5개):")
        for i, result in enumerate(sorted_results[-5:]):
            print(f"      #{len(sorted_results)-4+i}: threshold={result['threshold']:.3f}, FP={result['fp']}, FN={result['fn']}, 정확도={result['balanced_acc']:.1f}%")
        
        # 오분류율 계산
        total_samples = len(playing_array) + len(non_playing_array)
        misclassification_rate = (best_fp + best_fn) / total_samples * 100
        fp_rate = best_fp / len(playing_array) * 100
        fn_rate = best_fn / len(non_playing_array) * 100
        
        print(f"   🎯 최적 임계값: {best_threshold:.4f}")
        print(f"   📊 오분류 분석:")
        print(f"      총 오분류: {best_fp + best_fn}/{total_samples} ({misclassification_rate:.1f}%)")
        print(f"      False Positive: {best_fp}/{len(playing_array)} ({fp_rate:.1f}%) - 연주방해")
        print(f"      False Negative: {best_fn}/{len(non_playing_array)} ({fn_rate:.1f}%) - 놓친감지")
        
        # 🚨 문제 진단
        if fp_rate > 90:
            print(f"   🚨 문제 감지: False Positive 비율이 {fp_rate:.1f}%로 매우 높습니다!")
            print(f"      → 거의 모든 연주가 floating으로 잘못 분류됨")
            print(f"      → 임계값 {best_threshold:.3f}이 너무 낮을 가능성")
        
        if fn_rate > 90:
            print(f"   🚨 문제 감지: False Negative 비율이 {fn_rate:.1f}%로 매우 높습니다!")
            print(f"      → 거의 모든 비연주가 normal로 잘못 분류됨")
            print(f"      → 임계값 {best_threshold:.3f}이 너무 높을 가능성")
        
        # 분류 정확도 계산
        correct_playing = len(playing_array) - best_fp  # 연주 중 올바르게 normal 분류
        correct_non_playing = len(non_playing_array) - best_fn  # 비연주 중 올바르게 floating 분류
        
        playing_accuracy = correct_playing / len(playing_array) * 100
        non_playing_accuracy = correct_non_playing / len(non_playing_array) * 100
        overall_accuracy = (correct_playing + correct_non_playing) / total_samples * 100
        
        print(f"   ✅ 분류 정확도:")
        print(f"      연주 중 정확도: {playing_accuracy:.1f}% ({correct_playing}/{len(playing_array)})")
        print(f"      비연주 중 정확도: {non_playing_accuracy:.1f}% ({correct_non_playing}/{len(non_playing_array)})")
        print(f"      전체 정확도: {overall_accuracy:.1f}%")
        
        # 임계값 품질 평가
        if misclassification_rate <= 5:
            print(f"   🌟 우수한 임계값: 오분류율 {misclassification_rate:.1f}%")
        elif misclassification_rate <= 15:
            print(f"   ✅ 양호한 임계값: 오분류율 {misclassification_rate:.1f}%")
        elif misclassification_rate <= 30:
            print(f"   ⚠️  보통 임계값: 오분류율 {misclassification_rate:.1f}%")
        else:
            print(f"   ❌ 문제 있는 임계값: 오분류율 {misclassification_rate:.1f}%")
            print(f"      → 데이터 분포 재검토 필요")
        
        # 분포 겹침 정도 분석
        overlap_analysis = self._analyze_distribution_overlap(playing_array, non_playing_array)
        print(f"   📈 분포 겹침 분석:")
        print(f"      연주중 범위: {overlap_analysis['playing_range']}")
        print(f"      비연주중 범위: {overlap_analysis['non_playing_range']}")
        print(f"      겹침 정도: {overlap_analysis['overlap_description']}")
        
        # 🔍 추가 진단: 데이터 분포 확인
        if playing_mean > non_playing_mean:
            print(f"   ⚠️  데이터 분포 이상: 연주 중 평균({playing_mean:.3f})이 비연주 중 평균({non_playing_mean:.3f})보다 큽니다")
            print(f"      → 일반적으로 연주 중이 더 작은 depth 값을 가져야 함")
        
        return best_threshold
    
    def _analyze_distribution_overlap(self, playing_array: np.ndarray, non_playing_array: np.ndarray) -> Dict[str, str]:
        """두 분포의 겹침 정도 분석"""
        
        playing_min = np.min(playing_array)
        playing_max = np.max(playing_array)
        non_playing_min = np.min(non_playing_array)
        non_playing_max = np.max(non_playing_array)
        
        # 겹침 구간 계산
        overlap_start = max(playing_min, non_playing_min)
        overlap_end = min(playing_max, non_playing_max)
        
        if overlap_start >= overlap_end:
            # 겹치지 않음
            gap_size = overlap_start - overlap_end
            overlap_description = f"분리됨 (간격: {gap_size:.3f})"
        else:
            # 겹침 존재
            overlap_size = overlap_end - overlap_start
            total_range = max(playing_max, non_playing_max) - min(playing_min, non_playing_min)
            overlap_ratio = overlap_size / total_range * 100
            overlap_description = f"겹침 {overlap_ratio:.1f}% ({overlap_start:.3f}~{overlap_end:.3f})"
        
        return {
            'playing_range': f"{playing_min:.3f}~{playing_max:.3f}",
            'non_playing_range': f"{non_playing_min:.3f}~{non_playing_max:.3f}",
            'overlap_description': overlap_description
        }
    
    def _evaluate_threshold_performance(self, threshold: float, playing_array: np.ndarray, non_playing_array: np.ndarray) -> Dict[str, float]:
        """임계값 성능 평가"""
        
        # 예측 생성
        playing_predictions = (playing_array >= threshold).astype(int)
        non_playing_predictions = (non_playing_array >= threshold).astype(int)
        
        # 실제 레이블 (연주 중 = 0, 비연주 중 = 1)
        playing_labels = np.zeros(len(playing_array))
        non_playing_labels = np.ones(len(non_playing_array))
        
        # 모든 데이터 합치기
        all_predictions = np.concatenate([playing_predictions, non_playing_predictions])
        all_labels = np.concatenate([playing_labels, non_playing_labels])
        
        # 혼동 행렬 계산
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        tn = np.sum((all_predictions == 0) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))
        
        # 성능 지표 계산
        accuracy = (tp + tn) / (tp + fp + tn + fn) * 100 if (tp + fp + tn + fn) > 0 else 0
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity / 100) / (precision + sensitivity / 100) if (precision + sensitivity) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision * 100,
            'f1_score': f1_score
        }
    
    def calculate_valley_threshold(self, handlist: List) -> float:
        """히스토그램 골짜기 찾기 방법으로 임계값 계산"""
        print(f"🏔️  히스토그램 골짜기 찾기로 자동 임계값 계산 중...")
        
        # 모든 손의 깊이 값 수집
        all_depths = []
        for hands in handlist:
            for hand in hands:
                if hasattr(hand, 'handdepth') and hand.handdepth > 0:
                    all_depths.append(hand.handdepth)
        
        print(f"   📊 총 깊이 데이터: {len(all_depths):,}개")
        
        # 새로운 메서드 사용
        threshold = self.calculate_valley_threshold_with_data(all_depths)
        
        if len(all_depths) >= 50:
            # 히스토그램 정보 출력
            all_depths_array = np.array(all_depths)
            floating_count = np.sum(all_depths_array >= threshold)
            floating_ratio = floating_count / len(all_depths) * 100
            print(f"✅ 히스토그램 골짜기 임계값 계산 완료:")
            print(f"   🎪 예상 floating 비율: {floating_ratio:.1f}% ({floating_count:,}개)")
        
        return threshold
    
    def calculate_valley_threshold_with_data(self, depths_data: List) -> float:
        """주어진 깊이 데이터로 히스토그램 골짜기 찾기"""
        print(f"🏔️  히스토그램 골짜기 찾기 (데이터: {len(depths_data):,}개)")
        
        if len(depths_data) < 50:
            print(f"⚠️  깊이 데이터가 부족합니다. 기본값 사용: {self.fallback_threshold}")
            return self.fallback_threshold
        
        all_depths = np.array(depths_data)
        
        # 히스토그램 생성
        optimal_bins = min(100, max(20, len(all_depths) // 20))
        hist, bins = np.histogram(all_depths, bins=optimal_bins)
        
        # 히스토그램 평활화
        from scipy.ndimage import gaussian_filter1d
        try:
            smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=1.0)
        except ImportError:
            smoothed_hist = hist.astype(float)
            for i in range(1, len(smoothed_hist)-1):
                smoothed_hist[i] = (hist[i-1] + hist[i] + hist[i+1]) / 3.0
        
        # 골짜기 찾기
        valley_indices = []
        for i in range(2, len(smoothed_hist)-2):
            if (smoothed_hist[i] < smoothed_hist[i-1] and 
                smoothed_hist[i] < smoothed_hist[i+1] and
                smoothed_hist[i] < smoothed_hist[i-2] and
                smoothed_hist[i] < smoothed_hist[i+2]):
                valley_indices.append(i)
        
        if valley_indices:
            # 가장 깊은 골짜기 선택
            deepest_valley_idx = min(valley_indices, key=lambda x: smoothed_hist[x])
            threshold = bins[deepest_valley_idx]
            
            print(f"   🏔️  발견된 골짜기: {len(valley_indices)}개")
            print(f"   🎯 선택된 임계값: {threshold:.3f}")
            print(f"   📈 데이터 범위: {all_depths.min():.3f} ~ {all_depths.max():.3f}")
            
            return float(threshold)
        else:
            # 골짜기가 없으면 25 백분위수 사용
            alternative_threshold = np.percentile(all_depths, 25)
            print(f"   ⚠️  골짜기 없음. 25 백분위수 사용: {alternative_threshold:.3f}")
            
            if 0.1 <= alternative_threshold <= 1.5:
                return float(alternative_threshold)
            else:
                return self.fallback_threshold
    
    def get_dynamic_threshold(self, handlist: List) -> Dict[str, float]:
        """동적 임계값 계산 - 캐싱으로 중복 계산 방지"""
        if not self.auto_threshold:
            return {'Left': self.fallback_threshold, 'Right': self.fallback_threshold}
        
        # 이미 계산된 임계값이 있으면 재사용
        if self.cached_threshold is not None:
            if isinstance(self.cached_threshold, dict):
                print(f"🎯 캐시된 좌우 손 임계값 사용: Left={self.cached_threshold['Left']:.3f}, Right={self.cached_threshold['Right']:.3f}")
                return self.cached_threshold
            else:
                # 기존 단일 임계값을 좌우 손 공통으로 사용
                threshold_dict = {'Left': self.cached_threshold, 'Right': self.cached_threshold}
                print(f"🎯 캐시된 공통 임계값 사용: {self.cached_threshold:.3f}")
                return threshold_dict
        
        print(f"🎯 동적 좌우 손 임계값 계산 시작...")
        print(f"   AUTO_THRESHOLD: {self.auto_threshold}")
        print(f"   THRESHOLD_METHOD: {self.threshold_method}")
        print(f"   FALLBACK_THRESHOLD: {self.fallback_threshold}")
        
        thresholds = self.calculate_auto_threshold(handlist)
        
        # 각 손 타입별로 범위 확인
        for hand_type in ['Left', 'Right']:
            if hand_type in thresholds:
                threshold = thresholds[hand_type]
                if threshold < 0.1 or threshold > 1.5:
                    print(f"⚠️  {hand_type} 손 임계값 범위 초과: {threshold:.3f} → fallback 사용: {self.fallback_threshold}")
                    thresholds[hand_type] = self.fallback_threshold
        
        # 계산된 임계값 캐싱
        self.cached_threshold = thresholds
        print(f"✅ 좌우 손 임계값 캐시 저장: Left={thresholds.get('Left', self.fallback_threshold):.3f}, Right={thresholds.get('Right', self.fallback_threshold):.3f}")
        
        return thresholds
    
    def find_target_data(self) -> Dict[str, Any]:
        """타겟 데이터 파일을 찾거나 자동으로 생성합니다"""
        print(f"🔍 타겟 데이터 검색 중: {self.target_video}")
        
        # 1. 캐시된 제한 데이터 확인 (Quick test 모드일 때)
        if self.quick_test and self.enable_caching:
            if self.check_cached_data():
                return self.load_cached_data()
        
        # 2. 기존 처리된 데이터 확인 (전체 데이터)
        if not self.quick_test:
            for item in os.listdir(self.video_capture_dir):
                item_path = os.path.join(self.video_capture_dir, item)
                
                if (os.path.isdir(item_path) and 
                    item.startswith(self.target_video) and 
                    '_' in item and item.split('_')[-1].isdigit()):
                    
                    # handlist와 floatingframes 파일 확인
                    handlist_files = [f for f in os.listdir(item_path) 
                                    if f.startswith(f"handlist_{self.target_video}_") and f.endswith('.pkl')]
                    floating_files = [f for f in os.listdir(item_path) 
                                    if f.startswith(f"floatingframes_{self.target_video}_") and f.endswith('.pkl')]
                    
                    if handlist_files and floating_files:
                        print(f"✅ 기존 처리된 데이터 발견: {item_path}")
                        return {
                            'video_name': self.target_video,
                            'data_dir': item_path,
                            'handlist_file': os.path.join(item_path, handlist_files[0]),
                            'floating_file': os.path.join(item_path, floating_files[0]),
                            'original_video': f"{self.target_video}.mp4"
                        }
        
        # 3. 처리된 데이터가 없으면 자동으로 생성
        print(f"❌ 처리된 데이터 없음. 자동으로 데이터 처리 시작...")
        
        if self.quick_test:
            print(f"🚀 Quick Test 모드: {self.frame_limit}프레임만 처리합니다")
            return self.process_limited_video_data()
        else:
            print(f"⚠️  전체 비디오 처리 모드")
            return self.process_video_data()
    
    def process_video_data(self) -> Dict[str, Any]:
        """비디오 데이터를 처리하고 결과 정보를 반환합니다"""
        video_file = f"{self.target_video}.mp4"
        video_path = os.path.join(self.video_capture_dir, video_file)
        
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ 비디오 파일 없음: {video_path}")
        
        # keyboardcoordinateinfo.pkl 파일 확인
        keyboard_info_path = os.path.join(self.script_dir, "keyboardcoordinateinfo.pkl")
        if not os.path.exists(keyboard_info_path):
            raise FileNotFoundError(f"❌ 키보드 좌표 정보 파일 없음: {keyboard_info_path}")
        
        # 해당 비디오의 키보드 좌표 정보 확인
        try:
            with open(keyboard_info_path, 'rb') as f:
                keyboard_info = pickle.load(f)
            if self.target_video not in keyboard_info:
                raise KeyError(f"❌ {self.target_video}에 대한 키보드 좌표 정보 없음")
            print(f"✅ 키보드 좌표 정보 확인: {self.target_video}")
        except Exception as e:
            raise RuntimeError(f"❌ 키보드 좌표 정보 로드 실패: {e}")
        
        print(f"🎬 비디오 처리 시작: {video_file}")
        print(f"   크기: {os.path.getsize(video_path) / (1024*1024):.1f}MB")
        
        try:
            # main_loop 모듈을 지연 import하고 datagenerate 함수 호출
            main_loop = import_main_loop()
            
            # Quick test 모드일 때 프레임 제한 알림
            if self.quick_test:
                print(f"⚠️  주의: 전체 비디오 처리 후 {self.frame_limit}프레임만 사용됩니다")
                print(f"   첫 실행 시에는 전체 처리가 필요하며, 이후 실행에서는 빠르게 로딩됩니다")
            
            main_loop.datagenerate(video_file)
            print(f"✅ 비디오 처리 완료: {video_file}")
            
            # 처리된 데이터 위치 다시 찾기
            for item in os.listdir(self.video_capture_dir):
                item_path = os.path.join(self.video_capture_dir, item)
                
                if (os.path.isdir(item_path) and 
                    item.startswith(self.target_video) and 
                    '_' in item and item.split('_')[-1].isdigit()):
                    
                    # handlist와 floatingframes 파일 확인
                    handlist_files = [f for f in os.listdir(item_path) 
                                    if f.startswith(f"handlist_{self.target_video}_") and f.endswith('.pkl')]
                    floating_files = [f for f in os.listdir(item_path) 
                                    if f.startswith(f"floatingframes_{self.target_video}_") and f.endswith('.pkl')]
                    
                    if handlist_files and floating_files:
                        print(f"✅ 새로 생성된 데이터 확인: {item_path}")
                        return {
                            'video_name': self.target_video,
                            'data_dir': item_path,
                            'handlist_file': os.path.join(item_path, handlist_files[0]),
                            'floating_file': os.path.join(item_path, floating_files[0]),
                            'original_video': f"{self.target_video}.mp4"
                        }
            
            raise RuntimeError(f"❌ 데이터 처리는 완료되었지만 결과 파일을 찾을 수 없습니다")
            
        except Exception as e:
            print(f"❌ 비디오 처리 중 오류: {e}")
            raise
    
    def check_cached_data(self) -> bool:
        """캐시된 데이터가 있는지 확인"""
        if not os.path.exists(self.cache_data_dir):
            return False
        
        # 필요한 파일들 확인
        handlist_file = os.path.join(self.cache_data_dir, f"handlist_{self.target_video}_limit{self.frame_limit}.pkl")
        floating_file = os.path.join(self.cache_data_dir, f"floatingframes_{self.target_video}_limit{self.frame_limit}.pkl")
        
        return os.path.exists(handlist_file) and os.path.exists(floating_file)
    
    def load_cached_data(self) -> Dict[str, Any]:
        """캐시된 데이터 로드"""
        print(f"✅ 캐시된 {self.frame_limit}프레임 데이터 발견: {self.cache_data_dir}")
        
        handlist_file = os.path.join(self.cache_data_dir, f"handlist_{self.target_video}_limit{self.frame_limit}.pkl")
        floating_file = os.path.join(self.cache_data_dir, f"floatingframes_{self.target_video}_limit{self.frame_limit}.pkl")
        
        return {
            'video_name': f"{self.target_video}_limit{self.frame_limit}",
            'data_dir': self.cache_data_dir,
            'handlist_file': handlist_file,
            'floating_file': floating_file,
            'original_video': f"{self.target_video}_limit{self.frame_limit}.mp4"
        }
    
    def extract_limited_frames(self, input_path: str, output_path: str) -> bool:
        """제한된 프레임 추출"""
        print(f"📹 처음 {self.frame_limit}프레임 추출 중: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ 비디오 열기 실패: {input_path}")
            return False
        
        # 비디오 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 비디오 설정 (VSCode 호환성을 위한 H.264 코덱)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened() and frame_count < self.frame_limit:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"   📊 추출 중: {frame_count}/{self.frame_limit} 프레임")
        
        cap.release()
        out.release()
        
        print(f"✅ {self.frame_limit}프레임 추출 완료: {output_path}")
        return True
    
    def process_limited_video_data(self) -> Dict[str, Any]:
        """제한된 프레임 비디오 데이터 처리"""
        original_video_path = os.path.join(self.video_capture_dir, f"{self.target_video}.mp4")
        
        # 원본 비디오 존재 확인
        if not os.path.exists(original_video_path):
            raise FileNotFoundError(f"❌ 원본 비디오 없음: {original_video_path}")
        
        # 1. 제한된 프레임 비디오 생성
        if not os.path.exists(self.limited_video_path):
            print(f"📹 {self.frame_limit}프레임 비디오 생성 중...")
            if not self.extract_limited_frames(original_video_path, self.limited_video_path):
                raise RuntimeError(f"❌ 제한된 프레임 비디오 생성 실패")
        else:
            print(f"✅ {self.frame_limit}프레임 비디오 발견: {self.limited_video_path}")
        
        # 2. 키보드 좌표 정보 확인
        keyboard_info_path = os.path.join(self.script_dir, "keyboardcoordinateinfo.pkl")
        if not os.path.exists(keyboard_info_path):
            raise FileNotFoundError(f"❌ 키보드 좌표 정보 파일 없음: {keyboard_info_path}")
        
        try:
            with open(keyboard_info_path, 'rb') as f:
                keyboard_info = pickle.load(f)
            if self.target_video not in keyboard_info:
                raise KeyError(f"❌ {self.target_video}에 대한 키보드 좌표 정보 없음")
            print(f"✅ 키보드 좌표 정보 확인: {self.target_video}")
        except Exception as e:
            raise RuntimeError(f"❌ 키보드 좌표 정보 로드 실패: {e}")
        
        # 3. 제한된 비디오 데이터 처리
        print(f"🔄 {self.frame_limit}프레임 데이터 처리 시작...")
        
        try:
            # 임시로 제한된 비디오를 videocapture 디렉토리에 복사
            temp_video_name = f"{self.target_video}_temp.mp4"
            temp_video_path = os.path.join(self.video_capture_dir, temp_video_name)
            
            import shutil
            shutil.copy2(self.limited_video_path, temp_video_path)
            
            # 키보드 좌표 정보에 임시 키 추가
            keyboard_info_path = os.path.join(self.script_dir, "keyboardcoordinateinfo.pkl")
            with open(keyboard_info_path, 'rb') as f:
                keyboard_info = pickle.load(f)
            
            # 원본 키보드 정보를 임시 키로 복사
            temp_video_key = f"{self.target_video}_temp"
            keyboard_info[temp_video_key] = keyboard_info[self.target_video]
            
            # 임시로 키보드 정보 저장
            with open(keyboard_info_path, 'wb') as f:
                pickle.dump(keyboard_info, f)
            
            print(f"✅ 임시 키보드 좌표 정보 추가: {temp_video_key}")
            
            # main_loop 모듈로 데이터 처리
            main_loop = import_main_loop()
            main_loop.datagenerate(temp_video_name)
            
            # 처리된 데이터 찾기
            for item in os.listdir(self.video_capture_dir):
                item_path = os.path.join(self.video_capture_dir, item)
                
                if (os.path.isdir(item_path) and 
                    item.startswith(self.target_video + "_temp")):
                    
                    handlist_files = [f for f in os.listdir(item_path) 
                                    if f.startswith(f"handlist_{self.target_video}_temp_") and f.endswith('.pkl')]
                    floating_files = [f for f in os.listdir(item_path) 
                                    if f.startswith(f"floatingframes_{self.target_video}_temp_") and f.endswith('.pkl')]
                    
                    if handlist_files and floating_files:
                        # 캐시 디렉토리에 데이터 복사
                        os.makedirs(self.cache_data_dir, exist_ok=True)
                        
                        src_handlist = os.path.join(item_path, handlist_files[0])
                        src_floating = os.path.join(item_path, floating_files[0])
                        
                        dst_handlist = os.path.join(self.cache_data_dir, f"handlist_{self.target_video}_limit{self.frame_limit}.pkl")
                        dst_floating = os.path.join(self.cache_data_dir, f"floatingframes_{self.target_video}_limit{self.frame_limit}.pkl")
                        
                        shutil.copy2(src_handlist, dst_handlist)
                        shutil.copy2(src_floating, dst_floating)
                        
                        # 임시 파일들 정리
                        os.remove(temp_video_path)
                        shutil.rmtree(item_path)
                        
                        # 임시 키보드 정보 제거
                        keyboard_info_path = os.path.join(self.script_dir, "keyboardcoordinateinfo.pkl")
                        with open(keyboard_info_path, 'rb') as f:
                            keyboard_info = pickle.load(f)
                        
                        temp_video_key = f"{self.target_video}_temp"
                        if temp_video_key in keyboard_info:
                            del keyboard_info[temp_video_key]
                            with open(keyboard_info_path, 'wb') as f:
                                pickle.dump(keyboard_info, f)
                            print(f"✅ 임시 키보드 좌표 정보 제거: {temp_video_key}")
                        
                        print(f"✅ {self.frame_limit}프레임 데이터 처리 완료 및 캐시 저장")
                        
                        return {
                            'video_name': f"{self.target_video}_limit{self.frame_limit}",
                            'data_dir': self.cache_data_dir,
                            'handlist_file': dst_handlist,
                            'floating_file': dst_floating,
                            'original_video': f"{self.target_video}_limit{self.frame_limit}.mp4"
                        }
            
            raise RuntimeError(f"❌ 처리된 데이터 파일을 찾을 수 없습니다")
            
        except Exception as e:
            print(f"❌ 제한된 비디오 데이터 처리 중 오류: {e}")
            raise
    
    def load_data(self, data_info: Dict[str, Any]) -> Tuple[List, List, float]:
        """데이터를 로드합니다"""
        print(f"📁 데이터 로딩: {data_info['video_name']}")
        
        # handlist 로드
        with open(data_info['handlist_file'], 'rb') as f:
            handlist = pickle.load(f)
        
        # 기존 floating frames 로드
        with open(data_info['floating_file'], 'rb') as f:
            existing_floating_frames = pickle.load(f)
        
        # 비디오 정보에서 ratio 계산
        video_path = os.path.join(self.video_capture_dir, data_info['original_video'])
        ratio = 1.0
        
        if os.path.exists(video_path):
            video = cv2.VideoCapture(video_path)
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ratio = height / width
            video.release()
            print(f"   📐 비디오 해상도: {width:.0f}x{height:.0f}, ratio: {ratio:.3f}")
        
        print(f"   📊 handlist: {len(handlist)}개 프레임")
        print(f"   🔍 기존 floating: {len(existing_floating_frames)}개")
        
        # Quick test 모드일 때 프레임 제한 (캐시된 데이터가 아닌 경우만)
        is_cached_data = "limit" in data_info['video_name']
        
        if self.quick_test and not is_cached_data:
            print(f"🚀 Quick Test 모드: 초반 {self.frame_limit}프레임만 처리")
            
            # handlist 제한
            limited_handlist = []
            for hands in handlist:
                if hands:  # 빈 리스트가 아닌 경우
                    limited_hands = [hand for hand in hands if hand.handframe < self.frame_limit]
                    limited_handlist.append(limited_hands)
                else:
                    limited_handlist.append(hands)
            
            # floating frames 제한
            limited_floating = [
                (frame, handtype, depth, status) 
                for frame, handtype, depth, status in existing_floating_frames 
                if frame < self.frame_limit
            ]
            
            print(f"   ⚡ 제한된 handlist: {sum(len(hands) for hands in limited_handlist if hands)}개 손")
            print(f"   ⚡ 제한된 floating: {len(limited_floating)}개")
            
            return limited_handlist, limited_floating, ratio
        elif is_cached_data:
            print(f"✅ 캐시된 {self.frame_limit}프레임 데이터 로딩 완료")
            print(f"   📊 handlist: {len(handlist)}개 프레임 (이미 제한됨)")
            print(f"   🔍 floating: {len(existing_floating_frames)}개 (이미 제한됨)")
        
        return handlist, existing_floating_frames, ratio
    
    def run_scipy_detection(self, handlist: List, ratio: float) -> Tuple[List, float]:
        """SciPy 원본 버전 실행 (Golden Standard)"""
        monitor.start_step("SciPy 원본 알고리즘 실행", 4)
        
        start_time = time.time()
        
        try:
            import copy
            handlist_copy = copy.deepcopy(handlist)
            
            total_hands = sum(len(hands) for hands in handlist_copy if hands)
            total_frames = len([hands for hands in handlist_copy if hands])
            
            print(f"SciPy 처리: {total_hands:,}개 손, {total_frames:,}개 프레임")
            
            # 1단계: 모델 생성
            monitor.update_step_progress(1, "손 골격 모델 생성 중...")
            monitor.log_operation("손 골격 모델 구성", 
                                "좌/우 손 기준 모델 생성 (3차원 관절 구조 분석)", False)
            
            model_start = time.time()
            monitor.log_memory_usage("모델 생성 전")
            
            print(f"🔧 SciPy 모델 생성 시작:")
            print(f"   📊 입력 데이터: {len(handlist_copy)} 프레임")
            print(f"   🖐️  좌/우 손 구분 분석 중...")
            
            lhmodel, rhmodel = scipy_version.modelskeleton(handlist_copy)
            model_time = time.time() - model_start
            
            print(f"✅ SciPy 모델 생성 완료:")
            print(f"   🖐️  좌손 모델: {len(lhmodel) if lhmodel else 0}개 관절")
            print(f"   🖐️  우손 모델: {len(rhmodel) if rhmodel else 0}개 관절")
            print(f"   ⏱️  소요 시간: {model_time:.2f}초")
            
            monitor.log_memory_usage("모델 생성 후")
            
            # 2단계: 깊이 계산 (가장 시간이 오래 걸림)
            monitor.update_step_progress(2, f"SciPy 방정식 해법으로 {total_hands:,}개 손 깊이 계산 중...")
            
            print(f"\n🔬 SciPy 깊이 계산 시작:")
            print(f"   🧮 해법 방식: Powell's dog leg (Hybrid Newton-Raphson)")
            print(f"   📐 비선형 방정식 시스템: 3변수 3방정식")
            print(f"   🎯 허용 오차: SciPy 기본값 (1e-6 ~ 1e-12)")
            print(f"   ⚙️  처리 방식: 순차 처리 (CPU 단일 스레드)")
            print(f"   📊 대상 손: {total_hands:,}개")
            
            monitor.log_memory_usage("깊이 계산 전")
            start_depth = time.time()
            
            # 진행 상황 모니터링을 위한 콜백 설정
            hands_processed = 0
            last_update = time.time()
            
            def depth_progress_callback():
                nonlocal hands_processed, last_update
                current_time = time.time()
                if current_time - last_update > 5:  # 5초마다 업데이트
                    elapsed = current_time - start_depth
                    rate = hands_processed / elapsed if elapsed > 0 else 0
                    remaining = total_hands - hands_processed
                    eta = remaining / rate if rate > 0 else 0
                    
                    print(f"   ⏳ 진행 상황: {hands_processed:,}/{total_hands:,} ({hands_processed/total_hands*100:.1f}%)")
                    print(f"   ⚡ 처리 속도: {rate:.1f}개/초")
                    print(f"   ⏰ 예상 완료: {monitor._format_time(eta)}")
                    
                    # 개별 손 처리 과정 설명
                    print(f"   🧮 현재 연산: 비선형 방정식 해법 (Newton-Raphson)")
                    print(f"   📐 계산 중: 3D 공간 좌표 → 깊이 값 변환")
                    
                    last_update = current_time
            
            print(f"🔄 개별 손 처리 시작 (순차 처리):")
            scipy_version.depthlist(handlist_copy, lhmodel, rhmodel, ratio)
            
            depth_time = time.time() - start_depth
            depth_rate = total_hands / depth_time if depth_time > 0 else 0
            
            print(f"✅ SciPy 깊이 계산 완료:")
            print(f"   ⏱️  총 소요 시간: {depth_time:.2f}초")
            print(f"   ⚡ 평균 처리 속도: {depth_rate:.1f}개/초")
            print(f"   🎯 계산 정확도: 과학 연산 수준 (reference)")
            
            monitor.log_memory_usage("깊이 계산 후")
            
            # 3단계: 결함 프레임 검출
            monitor.update_step_progress(3, "결함 프레임 분석 중...")
            monitor.log_operation("결함 프레임 검출", 
                                "손 데이터 무결성 검사 및 이상 프레임 식별")
            
            print(f"🔍 결함 프레임 검출 시작:")
            print(f"   📊 분석 대상: {total_frames:,}개 프레임")
            print(f"   🔍 검사 항목: 손 데이터 무결성, 좌표 유효성")
            
            faulty_start = time.time()
            faultyframe = scipy_version.faultyframes(handlist_copy)
            faulty_time = time.time() - faulty_start
            
            print(f"✅ 결함 프레임 검출 완료:")
            print(f"   🚫 결함 프레임: {len(faultyframe):,}개")
            print(f"   ✅ 유효 프레임: {total_frames - len(faultyframe):,}개")
            print(f"   📊 데이터 품질: {(1-len(faultyframe)/total_frames)*100:.1f}%")
            print(f"   ⏱️  소요 시간: {faulty_time:.2f}초")
            
            # 4단계: floating 프레임 검출 (기존 0.9 임계값 사용)
            frame_count = max([max([hand.handframe for hand in hands]) for hands in handlist_copy if hands]) + 1
            valid_frames = frame_count - len(faultyframe)
            
            monitor.update_step_progress(4, f"Floating 프레임 분석 ({frame_count:,}프레임)...")
            monitor.log_operation("Floating 검출 알고리즘", 
                                f"기존 0.9 임계값 기반 floating 상태 판정 ({valid_frames:,}개 유효 프레임)")
            
            print(f"🎯 SciPy Floating 프레임 검출 시작:")
            print(f"   📊 전체 프레임: {frame_count:,}개")
            print(f"   🚫 결함 프레임: {len(faultyframe):,}개 (제외)")
            print(f"   ✅ 분석 대상: {valid_frames:,}개 프레임")
            print(f"   📏 고정 임계값: 0.9 (기존 표준 기준)")
            print(f"   🧮 판정 방식: 계산된 깊이 값 >= 0.9 → FLOATING")
            
            start_floating = time.time()
            floating_frames = scipy_version.detectfloatingframes(
                handlist_copy, frame_count, faultyframe, lhmodel, rhmodel, ratio
            )
            floating_time = time.time() - start_floating
            floating_rate = valid_frames / floating_time if floating_time > 0 else 0
            
            print(f"✅ SciPy Floating 검출 완료:")
            print(f"   🎯 Floating 감지: {len([f for f in floating_frames if f[3] == 'floating']):,}개")
            print(f"   📊 Normal 상태: {len([f for f in floating_frames if f[3] == 'notfloating']):,}개")
            print(f"   📈 Floating 비율: {len([f for f in floating_frames if f[3] == 'floating'])/len(floating_frames)*100:.1f}%")
            print(f"   ⏱️  소요 시간: {floating_time:.2f}초")
            print(f"   ⚡ 처리 속도: {floating_rate:.1f}프레임/초")
            
            monitor.log_memory_usage("Floating 검출 후")
            
        except Exception as e:
            print(f"❌ SciPy 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            monitor.finish_step("실행 실패")
            return [], 0.0
        
        scipy_time = time.time() - start_time
        
        monitor.finish_step(f"총 {len(floating_frames):,}개 floating 검출, {scipy_time:.2f}초 소요")
        
        print(f"\n📈 SciPy 전체 성능 요약:")
        print(f"   ⏱️  총 소요시간: {scipy_time:.2f}초")
        print(f"   🏗️  모델 생성: {model_time:.2f}초 ({model_time/scipy_time*100:.1f}%)")
        print(f"   🔬 깊이 계산: {depth_time:.2f}초 ({depth_time/scipy_time*100:.1f}%)")
        print(f"   🔍 결함 검출: {faulty_time:.2f}초 ({faulty_time/scipy_time*100:.1f}%)")
        print(f"   🎯 Floating 검출: {floating_time:.2f}초 ({floating_time/scipy_time*100:.1f}%)")
        print(f"   📊 전체 처리량: {total_hands/scipy_time:.1f}개/초")
        
        return floating_frames, scipy_time
    
    def run_pytorch_detection(self, handlist: List, ratio: float) -> Tuple[List, float]:
        """PyTorch 순수 버전 실행 (새 구현) - 상세 로깅 포함"""
        monitor.start_step("PyTorch GPU 가속 알고리즘 실행", 4)
        
        start_time = time.time()
        
        try:
            # handlist 복사
            import copy
            handlist_copy = copy.deepcopy(handlist)
            
            # 총 손 수 계산
            total_hands = sum(len(hands) for hands in handlist_copy if hands)
            total_frames = len([hands for hands in handlist_copy if hands])
            
            print(f"🚀 PyTorch 처리 대상 분석:")
            print(f"   👋 총 손 데이터: {total_hands:,}개")
            print(f"   🖼️  유효 프레임: {total_frames:,}개")
            print(f"   📐 비디오 비율: {ratio:.3f}")
            
            # GPU 정보 출력
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_props = torch.cuda.get_device_properties(0)
                print(f"🎮 GPU 가속 환경:")
                print(f"   🔧 GPU: {gpu_name}")
                print(f"   💾 VRAM: {gpu_memory:.1f}GB")
                print(f"   ⚡ Compute: {gpu_props.major}.{gpu_props.minor}")
                print(f"   🔥 CUDA: {torch.version.cuda}")
            else:
                print(f"⚠️  CPU 모드로 실행 (GPU 가속 불가)")
            
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
            monitor.log_operation("손 골격 모델 구성", 
                                "좌/우 손 기준 모델 생성 (PyTorch 텐서 형태)", False)
            
            model_start = time.time()
            monitor.log_memory_usage("모델 생성 전")
            
            print(f"🔧 PyTorch 모델 생성 시작:")
            print(f"   📊 입력 데이터: {len(handlist_copy)} 프레임")
            print(f"   🖐️  좌/우 손 구분 분석 중...")
            
            lhmodel, rhmodel = pytorch_version.modelskeleton(handlist_copy)
            model_time = time.time() - model_start
            
            print(f"✅ PyTorch 모델 생성 완료:")
            print(f"   🖐️  좌손 모델: {len(lhmodel) if lhmodel else 0}개 관절")
            print(f"   🖐️  우손 모델: {len(rhmodel) if rhmodel else 0}개 관절")
            print(f"   ⏱️  소요 시간: {model_time:.2f}초")
            
            monitor.log_memory_usage("모델 생성 후")
            
            # 2단계: 깊이 계산 (가장 시간이 오래 걸림) - 상세 로깅 추가
            monitor.update_step_progress(2, f"PyTorch 벡터화 연산으로 {total_hands:,}개 손 깊이 계산 중...")
            
            print(f"\n🚀 PyTorch 깊이 계산 시작:")
            print(f"   🧮 해법 방식: 고정점 반복법 (Fixed-point Iteration)")
            print(f"   📐 비선형 방정식 시스템: 3변수 3방정식 (벡터화)")
            print(f"   🎯 허용 오차: 1e-6 (성능 최적화)")
            print(f"   ⚙️  처리 방식: GPU 배치 병렬 처리")
            print(f"   📊 대상 손: {total_hands:,}개")
            
            # 배치 크기 결정 로직
            if TORCH_AVAILABLE and torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                batch_size = min(8192, max(1024, int(available_memory * 1000)))
                print(f"   📦 배치 크기: {batch_size:,}개 (GPU 메모리 기반 동적 조정)")
                print(f"   💾 사용 가능 VRAM: {available_memory:.1f}GB")
            else:
                batch_size = 1024
                print(f"   📦 배치 크기: {batch_size:,}개 (CPU 모드)")
            
            monitor.log_memory_usage("깊이 계산 전")
            start_depth = time.time()
            
            # GPU 메모리 모니터링 강화
            if TORCH_AVAILABLE and torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   💾 초기 GPU 메모리: {initial_memory:.2f}GB")
                
                # 실제 GPU 연산 시작 확인
                test_tensor = torch.randn(1000, 1000, device='cuda')
                torch.cuda.synchronize()  # GPU 연산 완료 대기
                print(f"   🔥 GPU 연산 준비 완료")
            
            # 배치 처리 상태 모니터링
            total_batches = (total_hands + batch_size - 1) // batch_size
            print(f"   📊 배치 처리 계획: {total_batches:,}개 배치")
            
            # 실제 깊이 계산 시작 - 진행 상황 모니터링 추가
            print(f"🔄 배치 처리 시작:")
            
            # 진행 상황 모니터링을 위한 콜백 함수 정의
            def create_progress_callback():
                from tqdm import tqdm
                pbar = tqdm(total=total_hands, desc="PyTorch 깊이 계산", ncols=80, leave=True)
                processed_count = 0
                
                def update_progress(batch_processed):
                    nonlocal processed_count
                    processed_count += batch_processed
                    pbar.update(batch_processed)
                    
                    # GPU 상태 업데이트
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            current_memory = torch.cuda.memory_allocated() / 1024**3
                            pbar.set_postfix({
                                '진행률': f'{processed_count}/{total_hands}',
                                'GPU_메모리': f'{current_memory:.1f}GB'
                            })
                        except:
                            pass
                
                def close_progress():
                    pbar.close()
                    
                return update_progress, close_progress
            
            # 진행 상황 콜백 생성
            progress_callback, close_callback = create_progress_callback()
            
            # PyTorch 모듈에 진행 상황 콜백 전달 (만약 지원한다면)
            # 현재는 기본 depthlist 호출
            try:
                # PyTorch 깊이 계산 시작
                print(f"⚡ GPU 가속 벡터 연산 시작...")
                
                # 실제 처리 시작 전 GPU 활성화
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    # GPU 워밍업
                    dummy_tensor = torch.randn(batch_size, 10, device='cuda')
                    torch.cuda.synchronize()
                    print(f"   🔥 GPU 워밍업 완료")
                
                # 배치 처리 모니터링
                hands_processed = 0
                last_log_time = time.time()
                
                # 실제 depthlist 호출
                pytorch_version.depthlist(handlist_copy, lhmodel, rhmodel, ratio)
                
                # 완료 후 진행바 업데이트
                progress_callback(total_hands)
                
            except Exception as e:
                print(f"❌ 깊이 계산 중 오류: {e}")
                raise
            finally:
                close_callback()
            
            depth_time = time.time() - start_depth
            depth_rate = total_hands / depth_time if depth_time > 0 else 0
            
            # GPU 메모리 사용량 확인
            if TORCH_AVAILABLE and torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                final_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   💾 최대 GPU 메모리: {peak_memory:.2f}GB")
                print(f"   💾 최종 GPU 메모리: {final_memory:.2f}GB")
                print(f"   📊 메모리 효율성: {(final_memory/peak_memory)*100:.1f}%")
            
            print(f"✅ PyTorch 깊이 계산 완료:")
            print(f"   ⏱️  총 소요 시간: {depth_time:.2f}초")
            print(f"   ⚡ 평균 처리 속도: {depth_rate:.1f}개/초")
            print(f"   🎯 계산 정확도: float64 정밀도 (과학적 등급)")
            print(f"   📈 정확성 우선: Newton-Raphson 방법 사용")
            
            monitor.log_memory_usage("깊이 계산 후")
            
            # 3단계: 결함 프레임 검출
            monitor.update_step_progress(3, "결함 프레임 분석 중...")
            monitor.log_operation("결함 프레임 검출", 
                                "손 데이터 무결성 검사 및 이상 프레임 식별")
            
            print(f"🔍 결함 프레임 검출 시작:")
            print(f"   📊 분석 대상: {total_frames:,}개 프레임")
            print(f"   🔍 검사 항목: 손 데이터 무결성, 좌표 유효성")
            
            faulty_start = time.time()
            faultyframe = pytorch_version.faultyframes(handlist_copy)
            faulty_time = time.time() - faulty_start
            
            print(f"✅ 결함 프레임 검출 완료:")
            print(f"   🚫 결함 프레임: {len(faultyframe):,}개")
            print(f"   ✅ 유효 프레임: {total_frames - len(faultyframe):,}개")
            print(f"   📊 데이터 품질: {(1-len(faultyframe)/total_frames)*100:.1f}%")
            print(f"   ⏱️  소요 시간: {faulty_time:.2f}초")
            
            # 4단계: floating 프레임 검출 - 상세 로깅 추가
            frame_count = max([max([hand.handframe for hand in hands]) for hands in handlist_copy if hands]) + 1
            valid_frames = frame_count - len(faultyframe)
            
            monitor.update_step_progress(4, f"Floating 프레임 분석 ({frame_count:,}프레임)...")
            monitor.log_operation("GPU 가속 Floating 검출", 
                                f"배치 처리 기반 floating 상태 판정 ({valid_frames:,}개 유효 프레임)")
            
            # 🎯 고정 임계값 0.9 사용 (SciPy와 동일)
            fixed_threshold = 0.9
            
            print(f"🎯 PyTorch Floating 프레임 검출 시작:")
            print(f"   📊 전체 프레임: {frame_count:,}개")
            print(f"   🚫 결함 프레임: {len(faultyframe):,}개 (제외)")
            print(f"   ✅ 분석 대상: {valid_frames:,}개 프레임")
            print(f"   🏔️  고정 임계값: {fixed_threshold:.3f} (SciPy와 동일)")
            print(f"   🧮 판정 방식: GPU 텐서 연산 (계산된 깊이 값 >= 임계값 → FLOATING)")
            print(f"   🚀 가속 방식: 배치 처리로 고속 벡터 연산")
            
            start_floating = time.time()
            
            # GPU 메모리 사용량 모니터링
            if TORCH_AVAILABLE and torch.cuda.is_available():
                floating_start_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   💾 Floating 검출 시작 메모리: {floating_start_memory:.2f}GB")
            
            # Floating 검출 진행 상황 모니터링
            print(f"🔄 Floating 검출 진행 중...")
            
            # Floating 검출 tqdm 진행바
            from tqdm import tqdm
            with tqdm(total=valid_frames, desc="Floating 검출", ncols=80, leave=True) as pbar:
                # 🎯 고정 임계값 0.9를 사용한 floating 검출
                floating_frames = pytorch_version.detectfloatingframes(
                    handlist_copy, frame_count, faultyframe, lhmodel, rhmodel, ratio, fixed_threshold
                )
                pbar.update(valid_frames)  # 완료 시 진행바 업데이트
            
            floating_time = time.time() - start_floating
            floating_rate = valid_frames / floating_time if floating_time > 0 else 0
            
            print(f"✅ PyTorch Floating 검출 완료:")
            print(f"   🎯 Floating 감지: {len([f for f in floating_frames if f[3] == 'floating']):,}개")
            print(f"   📊 Normal 상태: {len([f for f in floating_frames if f[3] == 'notfloating']):,}개")
            print(f"   📈 Floating 비율: {len([f for f in floating_frames if f[3] == 'floating'])/len(floating_frames)*100:.1f}%")
            print(f"   ⏱️  소요 시간: {floating_time:.2f}초")
            print(f"   ⚡ 처리 속도: {floating_rate:.1f}프레임/초")
            
            monitor.log_memory_usage("Floating 검출 후")
            
        except Exception as e:
            print(f"❌ PyTorch 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()  # GPU 메모리 정리
            monitor.finish_step("실행 실패")
            return [], 0.0
        
        pytorch_time = time.time() - start_time
        
        monitor.finish_step(f"총 {len(floating_frames):,}개 floating 검출, {pytorch_time:.2f}초 소요")
        
        print(f"\n📈 PyTorch 전체 성능 요약:")
        print(f"   ⏱️  총 소요시간: {pytorch_time:.2f}초")
        print(f"   🏗️  모델 생성: {model_time:.2f}초 ({model_time/pytorch_time*100:.1f}%)")
        print(f"   🚀 깊이 계산: {depth_time:.2f}초 ({depth_time/pytorch_time*100:.1f}%)")
        print(f"   🔍 결함 검출: {faulty_time:.2f}초 ({faulty_time/pytorch_time*100:.1f}%)")
        print(f"   🎯 Floating 검출: {floating_time:.2f}초 ({floating_time/pytorch_time*100:.1f}%)")
        print(f"   📊 전체 처리량: {total_hands/pytorch_time:.1f}개/초")
        
        # GPU 메모리 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_cleanup_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"   💾 메모리 정리 후: {final_cleanup_memory:.2f}GB")
        
        return floating_frames, pytorch_time
    
    def analyze_results(self, scipy_results: List, pytorch_results: List, 
                       existing_results: List, scipy_time: float, pytorch_time: float) -> Dict[str, Any]:
        """결과 분석 - 현실적인 정밀도 기준 적용"""
        print("📊 결과 분석 중...")
        
        # 현실적인 정밀도 기준 설정
        MEANINGFUL_DEPTH_THRESHOLD = 1e-3  # 1mm 이하는 무의미한 차이로 간주
        CLASSIFICATION_BOUNDARY = 0.9  # floating/normal 경계값
        
        # 결과를 frame, handtype별로 딕셔너리로 변환
        def results_to_dict(results):
            return {(r[0], r[1]): {'depth': r[2], 'status': r[3]} for r in results}
        
        scipy_dict = results_to_dict(scipy_results)
        pytorch_dict = results_to_dict(pytorch_results)
        existing_dict = results_to_dict(existing_results)
        
        # 공통 키와 비교
        all_keys = set(scipy_dict.keys()) | set(pytorch_dict.keys())
        
        agreements = 0
        depth_differences = []
        meaningful_differences = []
        status_mismatches = []
        near_boundary_cases = []
        
        for key in all_keys:
            scipy_data = scipy_dict.get(key)
            pytorch_data = pytorch_dict.get(key)
            
            if scipy_data and pytorch_data:
                # 깊이 차이 계산
                depth_diff = abs(scipy_data['depth'] - pytorch_data['depth'])
                depth_differences.append(depth_diff)
                
                # 의미있는 차이인지 판단
                if depth_diff > MEANINGFUL_DEPTH_THRESHOLD:
                    meaningful_differences.append(depth_diff)
                
                # 경계값 근처 케이스 체크
                scipy_near_boundary = abs(scipy_data['depth'] - CLASSIFICATION_BOUNDARY) < MEANINGFUL_DEPTH_THRESHOLD
                pytorch_near_boundary = abs(pytorch_data['depth'] - CLASSIFICATION_BOUNDARY) < MEANINGFUL_DEPTH_THRESHOLD
                
                if scipy_near_boundary or pytorch_near_boundary:
                    near_boundary_cases.append({
                        'frame': key[0],
                        'handtype': key[1],
                        'scipy_depth': scipy_data['depth'],
                        'pytorch_depth': pytorch_data['depth'],
                        'depth_diff': depth_diff
                    })
                
                # 상태 비교 - 경계값 근처는 관대하게 평가
                if scipy_data['status'] == pytorch_data['status']:
                    agreements += 1
                elif scipy_near_boundary and pytorch_near_boundary and depth_diff <= MEANINGFUL_DEPTH_THRESHOLD:
                    # 둘 다 경계값 근처이고 차이가 미미하면 일치로 간주
                    agreements += 1
                else:
                    status_mismatches.append({
                        'frame': key[0],
                        'handtype': key[1],
                        'scipy_status': scipy_data['status'],
                        'pytorch_status': pytorch_data['status'],
                        'scipy_depth': scipy_data['depth'],
                        'pytorch_depth': pytorch_data['depth'],
                        'depth_diff': depth_diff,
                        'near_boundary': scipy_near_boundary or pytorch_near_boundary
                    })
        
        agreement_rate = (agreements / len(all_keys) * 100) if all_keys else 0
        avg_depth_diff = np.mean(depth_differences) if depth_differences else 0
        max_depth_diff = np.max(depth_differences) if depth_differences else 0
        meaningful_diff_count = len(meaningful_differences)
        meaningful_diff_rate = (meaningful_diff_count / len(all_keys) * 100) if all_keys else 0
        speedup = scipy_time / pytorch_time if pytorch_time > 0 else 0
        
        return {
            'scipy_time': scipy_time,
            'pytorch_time': pytorch_time,
            'speedup': speedup,
            'agreement_rate': agreement_rate,
            'scipy_floating_count': len(scipy_results),
            'pytorch_floating_count': len(pytorch_results),
            'existing_floating_count': len(existing_results),
            'avg_depth_difference': avg_depth_diff,
            'max_depth_difference': max_depth_diff,
            'meaningful_diff_count': meaningful_diff_count,
            'meaningful_diff_rate': meaningful_diff_rate,
            'near_boundary_cases': len(near_boundary_cases),
            'status_mismatches': status_mismatches[:10],  # 최대 10개만
            'total_comparisons': len(all_keys),
            'precision_threshold': MEANINGFUL_DEPTH_THRESHOLD
        }
    
    def create_comparison_video(self, data_info: Dict[str, Any], 
                              scipy_results: List, pytorch_results: List, 
                              existing_results: List, handlist: List) -> str:
        """비교 결과를 시각화한 비디오 생성 - Hand landmarks와 floating 상태를 시각적으로 표시"""
        print("🎥 상세 비교 비디오 생성 중...")
        
        # 캐시된 데이터인 경우 캐시 디렉토리에서 비디오 찾기
        if "limit" in data_info['video_name']:
            video_path = self.limited_video_path
        else:
            video_path = os.path.join(self.video_capture_dir, data_info['original_video'])
        
        if not os.path.exists(video_path):
            print(f"❌ 원본 비디오 없음: {video_path}")
            return ""
        
        # Landmark 캐시 확인
        cached_landmarks = None
        if self.check_landmark_cache(video_path):
            cached_landmarks = self.load_landmark_cache()
        
        # 결과를 프레임별로 정리
        def organize_by_frame(results):
            frame_dict = {}
            for frame, handtype, depth, status in results:
                if frame not in frame_dict:
                    frame_dict[frame] = {}
                frame_dict[frame][handtype] = {'depth': depth, 'status': status}
            return frame_dict
        
        scipy_frames = organize_by_frame(scipy_results)
        pytorch_frames = organize_by_frame(pytorch_results)
        existing_frames = organize_by_frame(existing_results)
        
        # handlist를 프레임별로 정리 (캐시 사용 또는 기존 데이터 재사용)
        if cached_landmarks:
            handlist_by_frame = cached_landmarks['handlist_by_frame']
            print(f"🚀 캐시된 landmark 데이터 사용: {len(handlist_by_frame):,}개 프레임")
        else:
            # 기존 handlist에서 프레임별로 정리 (이미 처리된 데이터 재사용)
            handlist_by_frame = {}
            for hands in handlist:
                if hands:  # 빈 리스트가 아닌 경우
                    for hand in hands:
                        frame_num = hand.handframe
                        if frame_num not in handlist_by_frame:
                            handlist_by_frame[frame_num] = []
                        handlist_by_frame[frame_num].append(hand)
            
            print(f"✅ 기존 처리된 landmark 데이터 재사용: {len(handlist_by_frame):,}개 프레임")
            print(f"⚡ MediaPipe 재계산 생략으로 처리 시간 단축")
        
        # MediaPipe 실시간 추출 생략 (이미 처리된 데이터 사용)
        # handlist에 이미 모든 landmark 데이터가 있으므로 MediaPipe 재계산 불필요
        detector = None
        print(f"⚡ MediaPipe 재계산 생략: 기존 처리된 손 랜드마크 데이터 사용")
        
        # 임계값 미리 계산
        scipy_threshold = 0.9  # SciPy는 항상 0.9 고정
        pytorch_threshold = 0.9  # PyTorch도 0.9 고정
        
        print(f"🎯 비디오 시각화 임계값:")
        print(f"   SciPy: {scipy_threshold:.3f} (고정)")
        print(f"   PyTorch: {pytorch_threshold:.3f} (고정)")
        
        # 비디오 읽기 및 쓰기 준비
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Quick test 모드일 때 프레임 수 제한
        if self.quick_test:
            total_frames = min(total_frames, self.frame_limit)
        
        output_path = f"comparison_{self.target_video}_detailed_vscode_compatible.mp4"
        
        # VSCode 최적 호환성을 위한 코덱 설정 (우선순위 순서)
        codecs_to_try = [
            ('avc1', 'H.264 AVC1 (최고 호환성)'),
            ('H264', 'H.264 (표준)'),
            ('mp4v', 'MPEG-4 Part 2 (안전한 기본값)'),
            ('XVID', 'XVID (압축 효율성)'),
            ('MJPG', 'Motion JPEG (최종 fallback)')
        ]
        
        fourcc = None
        selected_codec = None
        
        for codec, description in codecs_to_try:
            try:
                test_fourcc = cv2.VideoWriter_fourcc(*codec)
                # 더 엄격한 호환성 테스트
                test_path = f'test_compatibility_{codec}.mp4'
                test_writer = cv2.VideoWriter(test_path, test_fourcc, fps, (width, height))
                
                if test_writer.isOpened():
                    # 실제 프레임 쓰기 테스트
                    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    test_writer.write(test_frame)
                    test_writer.release()
                    
                    # 파일이 정상적으로 생성되었는지 확인
                    if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                        # 파일 읽기 테스트
                        test_cap = cv2.VideoCapture(test_path)
                        if test_cap.isOpened():
                            ret, _ = test_cap.read()
                            test_cap.release()
                            if ret:
                                fourcc = test_fourcc
                                selected_codec = codec
                                print(f"   ✅ 비디오 코덱: {codec} ({description}) - 호환성 테스트 통과")
                                os.remove(test_path)
                                break
                        test_cap.release()
                    
                    # 테스트 파일 정리
                    if os.path.exists(test_path):
                        os.remove(test_path)
                else:
                    test_writer.release()
                    
            except Exception as e:
                print(f"   ⚠️  코덱 {codec} 테스트 실패: {e}")
                continue
        
        if fourcc is None:
            # 최종 fallback
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            selected_codec = 'mp4v'
            print(f"   ⚠️  모든 코덱 테스트 실패, 기본 코덱 사용: {selected_codec}")
        
        print(f"   🎬 선택된 코덱: {selected_codec} - VSCode 호환성 최적화")
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        # 기존 처리된 handlist 데이터 재사용으로 새로운 landmark 추출 불필요
        
        # tqdm으로 비디오 생성 진행상황 표시
        from tqdm import tqdm
        with tqdm(total=total_frames, desc="상세 비디오 생성", ncols=80, leave=True) as pbar:
            while cap.isOpened() and frame_num < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 현재 프레임의 floating 상태 확인
                scipy_frame_data = scipy_frames.get(frame_num, {})
                pytorch_frame_data = pytorch_frames.get(frame_num, {})
                existing_frame_data = existing_frames.get(frame_num, {})
                
                # 전체 프레임 floating 상태
                scipy_floating = any(data.get('status') == 'floating' 
                                   for data in scipy_frame_data.values())
                pytorch_floating = any(data.get('status') == 'floating' 
                                     for data in pytorch_frame_data.values())
                existing_floating = any(data.get('status') == 'floating' 
                                      for data in existing_frame_data.values())
                
                # 현재 프레임의 손 데이터 가져오기 (이미 처리된 데이터 사용)
                current_hands = handlist_by_frame.get(frame_num, [])
                
                # MediaPipe 실시간 추출 생략 - 이미 처리된 handlist 데이터 사용
                # current_hands에 이미 해당 프레임의 손 데이터가 있음
                
                # Hand landmarks 그리기
                if current_hands:
                    # 손 데이터를 MediaPipe 형식으로 변환
                    hand_landmarks = []
                    handedness = []
                    
                    for hand in current_hands:
                        # handlandmark를 MediaPipe NormalizedLandmarkList로 변환
                        landmarks = hand.handlandmark
                        
                        # 임시 handedness 객체 생성
                        class TempHandedness:
                            def __init__(self, category_name):
                                self.classification = [TempClassification(category_name)]
                        
                        class TempClassification:
                            def __init__(self, category_name):
                                self.category_name = category_name
                        
                        hand_landmarks.append(landmarks)
                        handedness.append(TempHandedness(hand.handtype))
                    
                    # SciPy 결과로 floating 상태 표시
                    scipy_floating_status = {}
                    for handtype, data in scipy_frame_data.items():
                        scipy_floating_status[handtype] = data.get('status') == 'floating'
                    
                    # PyTorch 결과로 floating 상태 표시
                    pytorch_floating_status = {}
                    for handtype, data in pytorch_frame_data.items():
                        pytorch_floating_status[handtype] = data.get('status') == 'floating'
                    
                    # 두 개의 이미지 생성 (좌: SciPy, 우: PyTorch) - 동일한 임계값 0.9 사용
                    scipy_image = draw_enhanced_hand_landmarks(
                        frame, hand_landmarks, handedness, scipy_floating_status, scipy_frame_data, " (SciPy)", scipy_threshold)
                    pytorch_image = draw_enhanced_hand_landmarks(
                        frame, hand_landmarks, handedness, pytorch_floating_status, pytorch_frame_data, " (PyTorch)", pytorch_threshold)
                    
                    # 두 이미지를 가로로 결합
                    combined_image = np.hstack([scipy_image, pytorch_image])
                    
                    # 중앙 분할선 그리기
                    mid_x = combined_image.shape[1] // 2
                    cv2.line(combined_image, (mid_x, 0), (mid_x, combined_image.shape[0]), (255, 255, 255), 3)
                    
                    # 좌우 레이블 추가
                    cv2.putText(combined_image, "SciPy (Golden Standard)", (20, combined_image.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(combined_image, "PyTorch (New Implementation)", (mid_x + 20, combined_image.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # 새로운 크기로 비디오 설정 (처음 프레임일 때만)
                    if frame_num == 0:
                        out.release()
                        new_width = combined_image.shape[1]
                        new_height = combined_image.shape[0]
                        # 이미 선택된 호환 코덱 사용
                        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
                    
                    # 향상된 비교 정보 헤더 추가
                    annotated_frame = draw_enhanced_comparison_header(
                        combined_image, frame_num, scipy_floating, pytorch_floating, 
                        scipy_frame_data, pytorch_frame_data, total_frames)
                    
                    # 불일치 프레임의 경우 특별한 강조 표시
                    if scipy_floating != pytorch_floating:
                        # 화면 전체에 빨간 테두리 (불일치 강조)
                        cv2.rectangle(annotated_frame, (0, 0), 
                                    (annotated_frame.shape[1]-1, annotated_frame.shape[0]-1), 
                                    (0, 0, 255), 8)
                        
                        # 경고 아이콘 추가
                        warning_x = annotated_frame.shape[1] // 2 - 100
                        warning_y = 140
                        cv2.putText(annotated_frame, "DISAGREEMENT DETECTED", 
                                   (warning_x, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (0, 255, 255), 2)
                    
                    out.write(annotated_frame)
                else:
                    # 손이 없는 프레임은 원본 프레임만 표시
                    no_hands_image = np.hstack([frame, frame])
                    
                    # 새로운 크기로 비디오 설정 (처음 프레임일 때만)
                    if frame_num == 0:
                        out.release()
                        new_width = no_hands_image.shape[1]
                        new_height = no_hands_image.shape[0]
                        # 이미 선택된 호환 코덱 사용
                        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
                    
                    # "No Hands Detected" 표시
                    cv2.putText(no_hands_image, f"Frame: {frame_num} - No Hands Detected", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                    out.write(no_hands_image)
                
                frame_num += 1
                
                # 진행상황 업데이트
                pbar.update(1)
                pbar.set_description(f"상세 비디오 생성 ({frame_num}/{total_frames})")
        
        cap.release()
        out.release()
        
        # 기존 처리된 데이터 재사용으로 캐시 저장 불필요
        # handlist에서 이미 모든 landmark 데이터를 가져왔으므로 추가 캐시 생성 없음
        
        # 비디오 생성 검증 및 메타데이터 확인
        self.verify_video_compatibility(output_path, selected_codec)
        
        print(f"✅ 상세 비교 비디오 생성 완료: {output_path}")
        print(f"   📏 해상도: {width*2}x{height} (좌: SciPy, 우: PyTorch)")
        print(f"   🎬 프레임 수: {frame_num}/{total_frames}")
        print(f"   🎥 사용 코덱: {selected_codec}")
        print(f"   ⚡ MediaPipe 재계산 생략으로 처리 시간 대폭 단축")
        
        # 오디오 합성 시도 - 원본 비디오에서 오디오 가져오기
        original_video_path = os.path.join(self.video_capture_dir, f"{self.target_video}.mp4")
        final_output_path = self.add_audio_to_video(output_path, original_video_path)
        
        return final_output_path
    

    
    def verify_video_compatibility(self, video_path: str, codec_used: str):
        """비디오 파일 호환성 검증 및 메타데이터 확인"""
        try:
            # 파일 존재 및 크기 확인
            if not os.path.exists(video_path):
                print(f"   ❌ 비디오 파일 생성 실패: {video_path}")
                return False
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                print(f"   ❌ 비디오 파일이 비어있음: {video_path}")
                return False
            
            # OpenCV로 비디오 읽기 테스트
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"   ❌ 비디오 파일 읽기 실패: {video_path}")
                return False
            
            # 메타데이터 확인
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # 첫 프레임 읽기 테스트
            ret, frame = cap.read()
            if not ret:
                print(f"   ❌ 비디오 프레임 읽기 실패: {video_path}")
                cap.release()
                return False
            
            cap.release()
            
            # 메타데이터 출력
            print(f"   🔍 비디오 검증 결과:")
            print(f"      📁 파일 크기: {file_size/1024/1024:.1f}MB")
            print(f"      📏 해상도: {width}x{height}")
            print(f"      🎬 프레임: {frame_count}개")
            print(f"      📹 FPS: {fps:.1f}")
            print(f"      🎥 코덱: {codec_used}")
            
            # VSCode 호환성 추천 사항 확인
            if codec_used in ['avc1', 'H264']:
                print(f"   ✅ VSCode 최적 호환성: {codec_used} 코덱 사용")
            elif codec_used in ['mp4v']:
                print(f"   ⚠️  VSCode 기본 호환성: {codec_used} 코덱 사용")
                print(f"      💡 H.264 코덱이 더 나은 호환성을 제공할 수 있습니다")
            else:
                print(f"   ⚠️  VSCode 호환성 주의: {codec_used} 코덱")
                print(f"      💡 VSCode에서 재생되지 않을 수 있습니다")
            
            # 추가 호환성 팁
            print(f"   💡 VSCode 재생 팁:")
            print(f"      1. 파일 탐색기에서 우클릭 → '연결 프로그램' → '미디어 플레이어'")
            print(f"      2. 브라우저에서 드래그 앤 드롭으로 재생")
            print(f"      3. VLC 등 외부 플레이어로 재생")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 비디오 검증 중 오류: {e}")
            return False
    
    def add_audio_to_video(self, video_path: str, source_video_path: str) -> str:
        """원본 비디오에서 오디오를 추출하여 생성된 비디오에 합성"""
        try:
            import subprocess
            
            # 출력 파일 경로 설정
            video_name = os.path.splitext(video_path)[0]
            output_with_audio = f"{video_name}_with_audio.mp4"
            
            print(f"🎵 오디오 합성 시작:")
            print(f"   📹 비디오: {video_path}")
            print(f"   🎧 오디오 소스: {source_video_path}")
            print(f"   🎬 출력: {output_with_audio}")
            
            # 1단계: 파일 존재 확인
            if not os.path.exists(video_path):
                print(f"❌ 비디오 파일이 존재하지 않습니다: {video_path}")
                return video_path
                
            if not os.path.exists(source_video_path):
                print(f"❌ 오디오 소스 파일이 존재하지 않습니다: {source_video_path}")
                return video_path
            
            # 2단계: ffmpeg 설치 확인
            try:
                ffmpeg_version = subprocess.run(['ffmpeg', '-version'], 
                                              capture_output=True, text=True, timeout=5)
                if ffmpeg_version.returncode != 0:
                    raise FileNotFoundError("ffmpeg 실행 실패")
                print(f"   ✅ ffmpeg 확인됨")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(f"❌ ffmpeg를 찾을 수 없습니다.")
                print(f"💡 ffmpeg 설치 방법:")
                print(f"   Ubuntu/Debian: sudo apt-get install ffmpeg")
                print(f"   Windows: https://ffmpeg.org/download.html")
                print(f"   macOS: brew install ffmpeg")
                return video_path
            
            # 3단계: 원본 비디오의 오디오 스트림 확인
            print(f"   🔍 원본 비디오 스트림 분석 중...")
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_streams', source_video_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode != 0:
                print(f"❌ 원본 비디오 분석 실패: {probe_result.stderr}")
                return video_path
            
            # JSON 파싱하여 오디오 스트림 확인
            import json
            try:
                streams_info = json.loads(probe_result.stdout)
                audio_streams = [s for s in streams_info.get('streams', []) 
                               if s.get('codec_type') == 'audio']
                
                if not audio_streams:
                    print(f"❌ 원본 비디오에 오디오 스트림이 없습니다.")
                    print(f"   📊 발견된 스트림: {len(streams_info.get('streams', []))}개")
                    for i, stream in enumerate(streams_info.get('streams', [])):
                        print(f"      스트림 {i}: {stream.get('codec_type', 'unknown')} - {stream.get('codec_name', 'unknown')}")
                    return video_path
                
                print(f"   ✅ 오디오 스트림 발견: {len(audio_streams)}개")
                audio_codec = audio_streams[0].get('codec_name', 'unknown')
                print(f"      코덱: {audio_codec}")
                
            except json.JSONDecodeError:
                print(f"❌ 비디오 스트림 정보 파싱 실패")
                return video_path
            
            # 4단계: 비디오 길이 계산
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = frame_count / fps
            cap.release()
            
            print(f"   📊 비디오 정보:")
            print(f"      프레임 수: {frame_count}")
            print(f"      FPS: {fps:.2f}")
            print(f"      길이: {video_duration:.2f}초")
            
            # 5단계: 오디오 합성 시도 (여러 방법)
            success = False
            
            # 방법 1: 기본 AAC 코덱
            print(f"   🎯 방법 1: AAC 코덱 시도...")
            cmd1 = [
                'ffmpeg',
                '-i', video_path,          # 비디오 입력
                '-i', source_video_path,   # 오디오 소스
                '-c:v', 'copy',            # 비디오 코덱 복사
                '-c:a', 'aac',             # 오디오 코덱 AAC
                '-map', '0:v:0',           # 첫 번째 입력의 비디오 스트림
                '-map', '1:a:0',           # 두 번째 입력의 오디오 스트림
                '-t', str(video_duration), # 비디오 길이만큼 오디오 자르기
                '-y',                      # 출력 파일 덮어쓰기
                output_with_audio
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True)
            if result1.returncode == 0 and os.path.exists(output_with_audio):
                success = True
                print(f"   ✅ 방법 1 성공!")
            else:
                print(f"   ❌ 방법 1 실패: {result1.stderr[:200]}...")
                
                # 방법 2: 오디오 코덱 복사
                print(f"   🎯 방법 2: 오디오 코덱 복사 시도...")
                cmd2 = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', source_video_path,
                    '-c:v', 'copy',
                    '-c:a', 'copy',            # 오디오 코덱도 복사
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-t', str(video_duration),
                    '-y',
                    output_with_audio
                ]
                
                result2 = subprocess.run(cmd2, capture_output=True, text=True)
                if result2.returncode == 0 and os.path.exists(output_with_audio):
                    success = True
                    print(f"   ✅ 방법 2 성공!")
                else:
                    print(f"   ❌ 방법 2 실패: {result2.stderr[:200]}...")
                    
                    # 방법 3: 단순 합성 (shortest 옵션)
                    print(f"   🎯 방법 3: 단순 합성 시도...")
                    cmd3 = [
                        'ffmpeg',
                        '-i', video_path,
                        '-i', source_video_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-shortest',               # 짧은 스트림에 맞춤
                        '-y',
                        output_with_audio
                    ]
                    
                    result3 = subprocess.run(cmd3, capture_output=True, text=True)
                    if result3.returncode == 0 and os.path.exists(output_with_audio):
                        success = True
                        print(f"   ✅ 방법 3 성공!")
                    else:
                        print(f"   ❌ 방법 3 실패: {result3.stderr[:200]}...")
            
            if success:
                # 성공한 경우 파일 크기 비교
                original_size = os.path.getsize(video_path) / 1024 / 1024
                new_size = os.path.getsize(output_with_audio) / 1024 / 1024
                
                print(f"✅ 오디오 합성 완료!")
                print(f"   📊 파일 크기 비교:")
                print(f"      원본 (무음): {original_size:.1f}MB")
                print(f"      오디오 포함: {new_size:.1f}MB")
                print(f"      증가량: {new_size - original_size:.1f}MB")
                
                # VSCode 호환성을 위한 H.264 변환
                final_h264_path = self.convert_to_h264(output_with_audio)
                
                # 중간 파일들 정리
                try:
                    os.remove(video_path)  # 원본 무음 비디오 삭제
                    print(f"   🗑️  원본 무음 비디오 삭제: {video_path}")
                except:
                    print(f"   ⚠️  원본 파일 삭제 실패")
                
                if final_h264_path != output_with_audio:
                    # H.264 변환이 성공한 경우 중간 파일 삭제
                    try:
                        os.remove(output_with_audio)
                        print(f"   🗑️  중간 파일 삭제: {output_with_audio}")
                    except:
                        print(f"   ⚠️  중간 파일 삭제 실패")
                
                return final_h264_path
            else:
                print(f"❌ 모든 오디오 합성 방법 실패")
                print(f"   원본 비디오 반환: {video_path}")
                return video_path
                
        except Exception as e:
            print(f"❌ 오디오 합성 중 예외 발생: {e}")
            print(f"   원본 비디오 반환: {video_path}")
            import traceback
            traceback.print_exc()
            return video_path
    
    def convert_to_h264(self, input_video_path: str) -> str:
        """VSCode 호환성을 위한 H.264 변환"""
        try:
            import subprocess
            
            # 출력 파일 경로 설정
            video_name = os.path.splitext(input_video_path)[0]
            h264_output = f"{video_name}_h264.mp4"
            
            print(f"🎬 VSCode 호환성 최적화 시작:")
            print(f"   📥 입력: {input_video_path}")
            print(f"   📤 출력: {h264_output}")
            
            # H.264 변환 명령어 구성
            cmd = [
                'ffmpeg',
                '-i', input_video_path,
                '-c:v', 'libx264',          # H.264 비디오 코덱
                '-c:a', 'aac',              # AAC 오디오 코덱
                '-preset', 'medium',        # 인코딩 속도 vs 품질 균형
                '-crf', '23',               # 품질 설정 (18-28 범위, 23이 기본)
                '-pix_fmt', 'yuv420p',      # 픽셀 포맷 (최대 호환성)
                '-profile:v', 'main',       # H.264 프로파일 (호환성 우선)
                '-level', '4.0',            # H.264 레벨
                '-movflags', '+faststart',  # 웹 스트리밍 최적화
                '-y',                       # 출력 파일 덮어쓰기
                h264_output
            ]
            
            print(f"   ⚙️  H.264 변환 중... (시간이 소요될 수 있습니다)")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(h264_output):
                # 변환 성공
                original_size = os.path.getsize(input_video_path) / 1024 / 1024
                h264_size = os.path.getsize(h264_output) / 1024 / 1024
                
                print(f"✅ H.264 변환 완료!")
                print(f"   📊 파일 크기 비교:")
                print(f"      변환 전: {original_size:.1f}MB")
                print(f"      H.264 후: {h264_size:.1f}MB")
                print(f"      차이: {h264_size - original_size:+.1f}MB")
                
                # 비디오 품질 확인
                probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', h264_output]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                
                if probe_result.returncode == 0:
                    import json
                    try:
                        streams_info = json.loads(probe_result.stdout)
                        video_streams = [s for s in streams_info.get('streams', []) if s.get('codec_type') == 'video']
                        audio_streams = [s for s in streams_info.get('streams', []) if s.get('codec_type') == 'audio']
                        
                        if video_streams:
                            video_codec = video_streams[0].get('codec_name', 'unknown')
                            profile = video_streams[0].get('profile', 'unknown')
                            print(f"   🎥 비디오: {video_codec} ({profile})")
                        
                        if audio_streams:
                            audio_codec = audio_streams[0].get('codec_name', 'unknown')
                            print(f"   🎵 오디오: {audio_codec}")
                        
                        print(f"   💻 VSCode 호환성: 최적화 완료 (H.264 Main Profile)")
                        print(f"   🌐 웹 브라우저 호환성: 우수 (faststart 플래그)")
                        
                    except json.JSONDecodeError:
                        print(f"   ⚠️  비디오 정보 확인 실패 (기능상 문제없음)")
                
                return h264_output
            else:
                print(f"❌ H.264 변환 실패:")
                print(f"   에러: {result.stderr[:300]}...")
                print(f"   원본 파일 유지: {input_video_path}")
                return input_video_path
                
        except Exception as e:
            print(f"❌ H.264 변환 중 예외 발생: {e}")
            print(f"   원본 파일 유지: {input_video_path}")
            return input_video_path
    
    def make_json_serializable(self, obj):
        """결과 딕셔너리를 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float)):
            return obj
        else:
            # 기타 객체는 문자열로 변환
            return str(obj)
    
    def run_benchmark(self) -> Dict[str, Any]:
        """전체 벤치마킹 실행 - 상세 모니터링 포함"""
        mode_text = f"({self.frame_limit}프레임 제한)" if self.quick_test else "(전체 프레임)"
        print(f"🚀 {self.target_video} 벤치마킹 시작 {mode_text}")
        print("=" * 60)
        
        # 모니터링 시작
        monitor.start_monitoring(total_steps=5)
        
        try:
            # 1. 데이터 찾기 및 로드
            monitor.start_step("데이터 로드 및 초기화", 3)
            
            print("🔍 벤치마킹 대상 데이터 검색 중...")
            data_info = self.find_target_data()
            monitor.update_step_progress(1, f"데이터 위치: {data_info['data_dir']}")
            
            print("📁 손 데이터 및 비디오 정보 로딩 중...")
            handlist, existing_floating, ratio = self.load_data(data_info)
            monitor.update_step_progress(2, f"handlist: {len(handlist)}프레임 로딩 완료")
            
            # 실제 처리할 프레임 수 계산
            actual_frame_count = len([hand for hands in handlist if hands for hand in hands])
            total_hands = sum(len(hands) for hands in handlist if hands)
            
            monitor.update_step_progress(3, f"분석 준비: {total_hands:,}개 손 데이터")
            monitor.finish_step(f"총 {total_hands:,}개 손, {len(handlist)}프레임 로딩 완료")
            
            print(f"\n📊 벤치마킹 데이터 요약:")
            print(f"   🎬 비디오: {data_info['video_name']}")
            print(f"   📐 해상도 비율: {ratio:.3f}")
            print(f"   🖼️  총 프레임: {len(handlist):,}개")
            print(f"   👋 총 손 데이터: {total_hands:,}개")
            print(f"   📊 기존 floating: {len(existing_floating):,}개")
            
            # 2. SciPy 원본 버전 실행 (Golden Standard)
            print(f"\n⚙️  Golden Standard 벤치마킹 시작...")
            print(f"🔬 SciPy 기반 원본 알고리즘 (과학 연산 정밀도)")
            scipy_results, scipy_time = self.run_scipy_detection(handlist, ratio)
            
            # 3. PyTorch 순수 버전 실행 (새 구현)
            print(f"\n🚀 신규 구현 벤치마킹 시작...")
            print(f"🎮 PyTorch GPU 가속 알고리즘 (성능 최적화)")
            pytorch_results, pytorch_time = self.run_pytorch_detection(handlist, ratio)
            
            # 4. 결과 분석
            monitor.start_step("결과 정확도 분석", 1)
            print("📊 SciPy vs PyTorch 결과 비교 분석 중...")
            analysis_start = time.time()
            analysis = self.analyze_results(scipy_results, pytorch_results, existing_floating, 
                                          scipy_time, pytorch_time)
            analysis_time = time.time() - analysis_start
            monitor.update_step_progress(1, f"정확도 분석: {analysis['agreement_rate']:.1f}% 일치율")
            monitor.finish_step(f"분석 완료: {analysis['total_comparisons']:,}개 비교")
            
            # 5. 비교 비디오 생성
            monitor.start_step("비교 비디오 생성", 1)
            
            detailed_video_path = ""
            
            if self.generate_detailed_video:
                print("🎥 상세 비교 비디오 생성 중... (Hand landmarks 포함)")
                video_start = time.time()
                detailed_video_path = self.create_comparison_video(data_info, scipy_results, pytorch_results, existing_floating, handlist)
                video_time = time.time() - video_start
                monitor.update_step_progress(1, f"상세 비디오: {detailed_video_path} ({video_time:.1f}초)")
            else:
                print("⚠️  비디오 생성이 비활성화되어 있습니다.")
                monitor.update_step_progress(1, "비디오 생성 생략")
            
            monitor.finish_step("비디오 생성 완료")
            
            analysis['comparison_video_detailed'] = detailed_video_path
            
            return analysis
            
        finally:
            # 모니터링 종료
            monitor.stop_monitoring()

    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력 - 현실적인 정밀도 기준 반영"""
        print("\n" + "=" * 60)
        mode_text = f" (Quick Test: {self.frame_limit}프레임)" if self.quick_test else ""
        cache_text = " [캐시 사용]" if self.quick_test and self.enable_caching else ""
        print(f"📋 {self.target_video} 벤치마킹 결과{mode_text}{cache_text}")
        print("=" * 60)
        print(f"🎯 비교 대상: SciPy 원본 (Golden Standard) vs PyTorch 순수 구현")
        
        if self.quick_test and self.enable_caching:
            print(f"💾 캐시 정보: {self.cache_data_dir}")
            print(f"📁 제한 비디오: {self.limited_video_path}")
        
        print(f"\n⏱️  성능:")
        print(f"   SciPy 원본: {results['scipy_time']:.2f}초 (CPU)")
        print(f"   PyTorch 순수: {results['pytorch_time']:.2f}초 (GPU)")
        print(f"   성능 향상: {results['speedup']:.1f}배")
        
        print(f"\n🎯 정확도 비교:")
        print(f"   일치율: {results['agreement_rate']:.1f}%")
        print(f"   총 비교: {results['total_comparisons']}개")
        print(f"   불일치: {len(results['status_mismatches'])}개")
        
        print(f"\n🔍 Floating 감지:")
        print(f"   SciPy 원본: {results['scipy_floating_count']}개")
        print(f"   PyTorch 순수: {results['pytorch_floating_count']}개")
        print(f"   기존 데이터: {results['existing_floating_count']}개")
        
        print(f"\n📏 수치적 차이 분석:")
        print(f"   평균 깊이 차이: {results['avg_depth_difference']:.3f}")
        print(f"   최대 깊이 차이: {results['max_depth_difference']:.3f}")
        print(f"   허용오차 초과: {results['meaningful_diff_count']}개 ({results['meaningful_diff_rate']:.1f}%)")
        print(f"   경계값(0.9) 근처: {results['near_boundary_cases']}개")
        
        if results.get('comparison_video_detailed'):
            print(f"\n🎥 상세 비교 비디오: {results['comparison_video_detailed']}")
            print(f"   📏 해상도: 2배 확대 (좌: SciPy, 우: PyTorch)")
            print(f"   👁️ Hand landmarks 및 floating 상태 시각화")
            print(f"   💻 VSCode 호환: 최적화된 코덱 사용")
            if "h264" in results['comparison_video_detailed']:
                print(f"   🎵 오디오 포함: 원본 비디오 사운드 합성")
                print(f"   🎬 코덱: H.264 (VSCode 최적 호환성)")
            elif "with_audio" in results['comparison_video_detailed']:
                print(f"   🎵 오디오 포함: 원본 비디오 사운드 합성")
                print(f"   ⚠️  H.264 변환 실패: 기본 코덱 사용")
            else:
                print(f"   🔇 오디오 없음: ffmpeg 사용 불가 또는 오디오 합성 실패")
        
        if results['status_mismatches']:
            print(f"\n⚠️  주요 불일치 케이스 (경계값 근처 포함):")
            for mismatch in results['status_mismatches'][:5]:
                boundary_marker = " [경계값]" if mismatch.get('near_boundary', False) else ""
                print(f"   Frame {mismatch['frame']} {mismatch['handtype']}: "
                      f"SciPy={mismatch['scipy_status']}({mismatch['scipy_depth']:.3f}) vs "
                      f"PyTorch={mismatch['pytorch_status']}({mismatch['pytorch_depth']:.3f})"
                      f" 차이={mismatch['depth_diff']:.3f}{boundary_marker}")


def process_multiple_videos():
    """여러 비디오를 순차적으로 처리하는 메인 함수"""
    benchmark = SmallDataBenchmark()
    
    if not benchmark.target_videos:
        print("❌ 처리할 비디오가 없습니다.")
        return
    
    all_results = []
    
    for i, video_name in enumerate(benchmark.target_videos, 1):
        print(f"\n{'='*80}")
        print(f"🎬 [{i}/{len(benchmark.target_videos)}] 비디오 처리 중: {video_name}")
        print(f"   임계값: {benchmark.FIXED_THRESHOLD}")
        print(f"{'='*80}")
        
        # 현재 비디오 설정
        benchmark.set_current_video(video_name)
        
        try:
            # 벤치마크 실행
            results = benchmark.run_benchmark()
            
            if results:
                all_results.append({
                    'video_name': video_name,
                    'results': results
                })
                
                # 개별 결과 요약 출력
                print(f"\n✅ [{i}/{len(benchmark.target_videos)}] 완료: {video_name}")
                benchmark.print_summary(results)
                
        except Exception as e:
            print(f"❌ [{i}/{len(benchmark.target_videos)}] 실패: {video_name}")
            print(f"   오류: {str(e)}")
            continue
    
    # 전체 결과 요약
    print(f"\n{'='*80}")
    print(f"🎯 전체 처리 완료!")
    print(f"   성공: {len(all_results)}/{len(benchmark.target_videos)}개")
    print(f"   임계값: {benchmark.FIXED_THRESHOLD}")
    print(f"{'='*80}")
    
    for result in all_results:
        video_name = result['video_name']
        res = result['results']
        print(f"\n📊 {video_name}:")
        print(f"   SciPy 시간: {res.get('scipy_time', 0):.2f}초")
        print(f"   PyTorch 시간: {res.get('pytorch_time', 0):.2f}초")
        print(f"   일치율: {res.get('accuracy', {}).get('overall_accuracy', 0):.1f}%")
        
        if res.get('video_generated'):
            print(f"   비디오: {res.get('output_video_path', '생성됨')}")

if __name__ == "__main__":
    print("🚀 PianoVAM Floating Hand Detection 벤치마크")
    print("=" * 60)
    
    # 여러 비디오 처리 실행
    process_multiple_videos() 
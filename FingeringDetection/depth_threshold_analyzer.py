#!/usr/bin/env python3
"""
Depth Threshold Analyzer - MIDI-Supervised Version
- PyTorch floating detector에서 생성된 깊이 데이터(JSON) 분석
- MIDI 구간 정보와 매칭하여 최적 임계값 계산
- 왼손/오른손 별도 분석
- MIDI 기반 지도학습 기법만 사용 (미세한 차이에 최적화)
- ROC 최적화, Fisher 판별 분석 등 머신러닝 방법 적용
- 개선된 성능, 에러 처리, 검증 기능
- 참고: 비디오 파일(.mp4)은 사용하지 않음, JSON과 MIDI만 사용
- 통계적 분위수/평균 방법들은 제거됨 (건반 누름의 미세한 차이에서 무효)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
from dataclasses import dataclass
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import re

warnings.filterwarnings('ignore')

# MIDI 처리를 위한 라이브러리
try:
    import mido
    MIDO_AVAILABLE = True
    print("✅ MIDI processing module loaded successfully")
except ImportError:
    print("⚠️ MIDI module not available: pip install mido")
    MIDO_AVAILABLE = False

# 고급 분석을 위한 라이브러리
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from scipy import stats
    # find_peaks 제거됨 - valley detection 방법이 유효하지 않음
    from scipy.optimize import minimize_scalar
    SKLEARN_AVAILABLE = True
    print("✅ Advanced analysis modules loaded successfully")
except ImportError:
    print("⚠️ Advanced analysis modules not available: pip install scikit-learn scipy")
    SKLEARN_AVAILABLE = False

@dataclass
class AnalysisConfig:
    """분석 설정 클래스"""
    # 경로 설정
    depth_data_dir: str = "depth_data"
    midi_dir: Optional[str] = "/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert"
    output_dir: str = "threshold_analysis"
    
    # 비디오 설정
    target_fps: int = 20
    
    # MIDI 설정
    hand_split_note: int = 60  # C4 기준
    # tempo_bpm 제거됨 - 실제 MIDI 파일에서 템포를 읽어서 사용
    
    # 분석 설정
    depth_range: Tuple[float, float] = (0.5, 1.5)
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # 시각화 설정
    figure_dpi: int = 300
    figure_format: str = 'png'
    color_palette: str = 'default'
    
    # 검증 설정
    min_midi_segments: int = 10  # 최소 MIDI 구간 수
    max_time_drift: float = 0.1  # 최대 시간 동기화 오차 (초)
    
    def validate(self) -> bool:
        """설정 검증"""
        try:
            assert self.target_fps > 0, "FPS는 양수여야 합니다"
            assert 0 <= self.hand_split_note <= 127, "MIDI 음표 번호는 0-127 범위여야 합니다"
            assert self.depth_range[0] < self.depth_range[1], "깊이 범위가 올바르지 않습니다"
            assert self.max_workers > 0, "Worker 수는 양수여야 합니다"
            return True
        except AssertionError as e:
            print(f"❌ 설정 오류: {e}")
            return False

class DataValidator:
    """데이터 품질 검증 클래스"""
    
    @staticmethod
    def validate_depth_data(depth_data: Dict) -> Dict[str, Any]:
        """깊이 데이터 검증"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        try:
            # 기본 구조 검증
            required_fields = ['video_name', 'total_hands', 'frame_data']
            for field in required_fields:
                if field not in depth_data:
                    validation_result['errors'].append(f"필수 필드 누락: {field}")
                    validation_result['is_valid'] = False
            
            if not validation_result['is_valid']:
                return validation_result
            
            # 데이터 품질 검증
            frame_data = depth_data.get('frame_data', [])
            if not frame_data:
                validation_result['warnings'].append("프레임 데이터가 비어있습니다")
                return validation_result
            
            # 깊이 값 분포 검증
            all_depths = []
            for frame in frame_data:
                for hand in frame.get('hands', []):
                    depth = hand.get('depth')
                    if depth is not None:
                        all_depths.append(depth)
            
            if all_depths:
                depths_array = np.array(all_depths)
                validation_result['statistics'] = {
                    'total_samples': len(all_depths),
                    'mean_depth': float(np.mean(depths_array)),
                    'std_depth': float(np.std(depths_array)),
                    'min_depth': float(np.min(depths_array)),
                    'max_depth': float(np.max(depths_array)),
                    'outlier_ratio': float(np.sum((depths_array < 0.1) | (depths_array > 3.0)) / len(depths_array))
                }
                
                # 이상값 경고
                if validation_result['statistics']['outlier_ratio'] > 0.1:
                    validation_result['warnings'].append(f"이상값 비율이 높습니다: {validation_result['statistics']['outlier_ratio']:.1%}")
            
        except Exception as e:
            validation_result['errors'].append(f"검증 중 오류: {str(e)}")
            validation_result['is_valid'] = False
        
        return validation_result
    
    @staticmethod
    def validate_midi_data(midi_data: Dict, config: AnalysisConfig) -> Dict[str, Any]:
        """MIDI 데이터 검증"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            if not midi_data or 'note_segments' not in midi_data:
                validation_result['errors'].append("MIDI 데이터가 비어있거나 올바르지 않습니다")
                validation_result['is_valid'] = False
                return validation_result
            
            # 음표 구간 검증
            note_segments = midi_data['note_segments']
            total_segments = sum(len(segments) for segments in note_segments.values())
            
            if total_segments < config.min_midi_segments:
                validation_result['warnings'].append(f"MIDI 구간이 너무 적습니다: {total_segments}개")
            
            # 시간 범위 검증
            all_times = []
            for hand_segments in note_segments.values():
                for segment in hand_segments:
                    all_times.extend([segment['start_time'], segment['end_time']])
            
            if all_times:
                validation_result['statistics'] = {
                    'total_segments': total_segments,
                    'duration': max(all_times) - min(all_times),
                    'avg_segment_duration': np.mean([seg['duration'] for segments in note_segments.values() for seg in segments])
                }
        
        except Exception as e:
            validation_result['errors'].append(f"MIDI 검증 중 오류: {str(e)}")
            validation_result['is_valid'] = False
        
        return validation_result

class EnhancedDepthThresholdAnalyzer:
    """개선된 깊이 데이터 기반 임계값 분석기"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        분석기 초기화
        
        Args:
            config: 분석 설정 객체
        """
        self.config = config or AnalysisConfig()
        
        if not self.config.validate():
            raise ValueError("설정이 올바르지 않습니다")
        
        # 경로 설정
        self.depth_data_dir = Path(self.config.depth_data_dir)
        self.midi_dir = Path(self.config.midi_dir) if self.config.midi_dir else None
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self._setup_logging()
        
        # 데이터 저장
        self.depth_datasets: List[Dict] = []
        self.midi_datasets: List[Dict] = []
        self.combined_data: Dict[str, Dict] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        # 검증기
        self.validator = DataValidator()
        
        # 캐시
        self._midi_cache: Dict[str, Dict] = {}
        
        self.logger.info("Enhanced Depth Threshold Analyzer 초기화 완료")
    
    def _setup_logging(self):
        """로깅 설정"""
        log_file = self.output_dir / 'analysis.log'
        
        # 로거 설정
        self.logger = logging.getLogger('DepthAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # 핸들러가 이미 있으면 제거 (중복 방지)
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포매터
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def load_depth_data(self) -> List[Dict]:
        """개선된 깊이 데이터 로딩"""
        self.logger.info(f"깊이 데이터 로딩 시작: {self.depth_data_dir}")
        print(f"📊 Enhanced depth data loading: {self.depth_data_dir}")
        
        depth_files = list(self.depth_data_dir.glob("*_depth_data.json"))
        print(f"   Found files: {len(depth_files)}")
        
        datasets = []
        validation_summary = {'valid': 0, 'invalid': 0, 'warnings': 0}
        
        # 병렬 처리로 파일 로딩
        if self.config.parallel_processing and len(depth_files) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(self._load_single_depth_file, depth_files))
        else:
            results = [self._load_single_depth_file(file_path) for file_path in depth_files]
        
        for result in results:
            if result is not None:
                data, validation_result = result
                datasets.append(data)
                
                if validation_result['is_valid']:
                    validation_summary['valid'] += 1
                else:
                    validation_summary['invalid'] += 1
                
                if validation_result['warnings']:
                    validation_summary['warnings'] += 1
        
        self.depth_datasets = datasets
        
        print(f"📊 Loading completed: {len(datasets)} datasets")
        print(f"   ✅ Valid: {validation_summary['valid']}")
        print(f"   ❌ Invalid: {validation_summary['invalid']}")
        print(f"   ⚠️ With warnings: {validation_summary['warnings']}")
        
        self.logger.info(f"깊이 데이터 로딩 완료: {len(datasets)}개")
        return datasets
    
    def _load_single_depth_file(self, file_path: Path) -> Optional[Tuple[Dict, Dict]]:
        """단일 깊이 데이터 파일 로딩"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 검증
            validation_result = self.validator.validate_depth_data(data)
            
            if not validation_result['is_valid']:
                print(f"   ❌ {file_path.name}: 검증 실패")
                for error in validation_result['errors']:
                    print(f"      {error}")
                return None
            
            # 비디오 이름 추출
            video_name = data.get('video_name', file_path.stem.replace('_depth_data', ''))  # 실제로는 세션 식별자 (비디오 파일 사용 안 함)
            
            print(f"   ✅ {video_name}: {data.get('total_hands', 0)} hands")
            
            # 경고 출력
            for warning in validation_result['warnings']:
                print(f"   ⚠️ {video_name}: {warning}")
            
            return data, validation_result
            
        except Exception as e:
            print(f"   ❌ {file_path.name}: {e}")
            self.logger.error(f"파일 로딩 실패 {file_path}: {e}")
            return None
    
    def find_midi_files(self) -> List[Path]:
        """개선된 MIDI 파일 탐색"""
        if not MIDO_AVAILABLE:
            print("⚠️ MIDI analysis unavailable: mido module required")
            return []
        
        # MIDI 디렉토리 자동 탐색 (기존 로직 유지하되 개선)
        if self.midi_dir is None:
            possible_dirs = [
                Path("/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert"),
                Path(".").parent / "midiconvert",
                Path(".") / "midiconvert", 
                Path(".").parent / "midi",
                Path(".") / "midi",
                Path(".").parent / "videocapture",
                Path(".") / "videocapture",
                Path(".").parent,
                Path(".")
            ]
            
            for dir_path in possible_dirs:
                if dir_path.exists():
                    midi_files = list(dir_path.glob("*.mid")) + list(dir_path.glob("*.midi"))
                    if midi_files:
                        self.midi_dir = dir_path
                        print(f"🎵 MIDI directory found: {dir_path}")
                        break
        
        if self.midi_dir is None or not self.midi_dir.exists():
            print("⚠️ No MIDI files found")
            return []
        
        midi_files = list(self.midi_dir.glob("*.mid")) + list(self.midi_dir.glob("*.midi"))
        print(f"🎵 Found {len(midi_files)} MIDI files")
        
        return midi_files
    
    def load_midi_data(self, midi_file: Path) -> Dict:
        """개선된 MIDI 데이터 로딩 (캐싱 지원)"""
        if not MIDO_AVAILABLE:
            return {}
        
        # 캐시 확인
        cache_key = str(midi_file)
        if self.config.enable_caching and cache_key in self._midi_cache:
            return self._midi_cache[cache_key]
        
        try:
            start_time = time.time()
            mid = mido.MidiFile(midi_file)
            
            # 개선된 MIDI 파싱
            result = self._parse_midi_file(mid, midi_file)
            
            # 검증
            validation_result = self.validator.validate_midi_data(result, self.config)
            if not validation_result['is_valid']:
                self.logger.warning(f"MIDI 검증 실패 {midi_file.name}: {validation_result['errors']}")
            
            # 캐싱
            if self.config.enable_caching:
                self._midi_cache[cache_key] = result
            
            load_time = time.time() - start_time
            self.logger.info(f"MIDI 로딩 완료 {midi_file.name}: {load_time:.2f}초")
            
            return result
            
        except Exception as e:
            print(f"❌ MIDI loading failed {midi_file.name}: {e}")
            self.logger.error(f"MIDI 로딩 실패 {midi_file}: {e}")
            return {}
    
    def _parse_midi_file(self, mid: mido.MidiFile, midi_file: Path) -> Dict:
        """개선된 MIDI 파일 파싱 - 실제 MIDI 템포 사용"""
        # 활성 음표 추적용 (hand별, note별)
        active_notes = {'Left': {}, 'Right': {}}
        note_segments = {'Left': [], 'Right': []}
        
        # MIDI 파일에서 실제 템포 추출
        current_tempo = 500000  # 기본값 (120 BPM)
        
        # 첫 번째 패스: 템포 정보 수집
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    current_tempo = msg.tempo
                    print(f"   🎵 MIDI 템포 발견: {current_tempo} 마이크로초/박자 ({60000000/current_tempo:.1f} BPM)")
                    break  # 첫 번째 템포만 사용 (단순화)
            if current_tempo != 500000:  # 템포를 찾았으면 중단
                break
        
        if current_tempo == 500000:
            print(f"   ⚠️ 템포 정보 없음, 기본값 사용: 120 BPM")
        
        # 트랙별 처리
        for track_idx, track in enumerate(mid.tracks):
            track_time = 0
            
            for msg in track:
                track_time += msg.time
                
                if msg.type in ['note_on', 'note_off']:
                    # 실제 MIDI 템포를 사용한 시간 변환
                    time_seconds = mido.tick2second(track_time, mid.ticks_per_beat, current_tempo)
                    
                    # 개선된 손 구분 (설정 가능한 분할점)
                    hand_type = 'Right' if msg.note >= self.config.hand_split_note else 'Left'
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # 음표 시작
                        active_notes[hand_type][msg.note] = {
                            'start_time': time_seconds,
                            'velocity': msg.velocity,
                            'track': track_idx
                        }
                    else:
                        # 음표 종료
                        if msg.note in active_notes[hand_type]:
                            start_info = active_notes[hand_type].pop(msg.note)
                            duration = time_seconds - start_info['start_time']
                            
                            # 너무 짧은 음표 필터링
                            if duration > 0.01:  # 10ms 이상
                                note_segments[hand_type].append({
                                    'start_time': start_info['start_time'],
                                    'end_time': time_seconds,
                                    'note': msg.note,
                                    'velocity': start_info['velocity'],
                                    'duration': duration,
                                    'track': start_info['track']
                                })
        
        # 시간 순 정렬
        for hand_type in note_segments:
            note_segments[hand_type].sort(key=lambda x: x['start_time'])
        
        # 통계 계산
        total_segments = sum(len(segments) for segments in note_segments.values())
        max_time = 0
        
        for segments in note_segments.values():
            if segments:
                max_time = max(max_time, max(seg['end_time'] for seg in segments))
        
        return {
            'file': midi_file.name,
            'note_segments': note_segments,
            'total_segments': total_segments,
            'duration': max_time,
            'hand_distribution': {
                'Left': len(note_segments['Left']),
                'Right': len(note_segments['Right'])
            }
        }

    def match_depth_and_midi(self) -> Dict[str, Dict]:
        """개선된 깊이-MIDI 데이터 매칭"""
        print(f"🔗 Enhanced depth-MIDI data matching")
        self.logger.info("깊이-MIDI 데이터 매칭 시작")
        
        midi_files = self.find_midi_files()
        combined_data = {}
        
        matching_stats = {
            'perfect_matches': 0,
            'partial_matches': 0,
            'no_matches': 0
        }
        
        for depth_data in self.depth_datasets:
            video_name = depth_data['video_name']
            
            # 개선된 매칭 알고리즘
            matching_midi, match_quality = self._find_best_midi_match(video_name, midi_files)
            
            if matching_midi and matching_midi.get('total_segments', 0) > 0:
                print(f"   ✅ {video_name}: MIDI matched ({matching_midi['total_segments']} segments, quality: {match_quality})")
                combined_data[video_name] = {
                    'depth_data': depth_data,
                    'midi_data': matching_midi,
                    'has_midi': True,
                    'match_quality': match_quality
                }
                
                if match_quality == 'perfect':
                    matching_stats['perfect_matches'] += 1
                else:
                    matching_stats['partial_matches'] += 1
            else:
                print(f"   ⚠️ {video_name}: No MIDI match (depth data only)")
                combined_data[video_name] = {
                    'depth_data': depth_data,
                    'midi_data': None,
                    'has_midi': False,
                    'match_quality': 'none'
                }
                matching_stats['no_matches'] += 1
        
        self.combined_data = combined_data
        
        print(f"🔗 Matching completed:")
        print(f"   Perfect matches: {matching_stats['perfect_matches']}")
        print(f"   Partial matches: {matching_stats['partial_matches']}")  
        print(f"   No matches: {matching_stats['no_matches']}")
        
        self.logger.info(f"매칭 완료: {matching_stats}")
        return combined_data
    
    def _find_best_midi_match(self, video_name: str, midi_files: List[Path]) -> Tuple[Optional[Dict], str]:
        """간단한 파일명 기반 MIDI 매칭"""
        # 비디오명에서 날짜-시간 패턴 추출 (YYYY-MM-DD_HH-MM-SS)
        video_pattern = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', video_name)
        
        if not video_pattern:
            print(f"   ⚠️ {video_name}: 날짜-시간 패턴을 찾을 수 없음")
            return None, 'none'
        
        target_pattern = video_pattern.group(1)
        
        # 동일한 패턴을 가진 MIDI 파일 찾기
        for midi_file in midi_files:
            midi_name = midi_file.stem
            
            # 정확한 날짜-시간 패턴 매칭
            if target_pattern in midi_name:
                print(f"   ✅ {video_name}: 정확한 MIDI 매칭 발견 - {midi_file.name}")
                
                # MIDI 데이터 로드
                midi_data = self.load_midi_data(midi_file)
                if midi_data and midi_data.get('total_segments', 0) > 0:
                    return midi_data, 'perfect'
                else:
                    print(f"   ⚠️ {midi_file.name}: MIDI 데이터가 비어있음")
        
        print(f"   ❌ {video_name}: 일치하는 MIDI 파일을 찾을 수 없음 (패턴: {target_pattern})")
        return None, 'none'
    
    def _calculate_video_duration(self, video_name: str) -> float:
        """비디오 길이 계산 (프레임 오프셋 보정 포함)"""
        for depth_data in self.depth_datasets:
            if depth_data.get('video_name') == video_name:
                frame_data = depth_data.get('frame_data', [])
                if frame_data:
                    max_frame = max(frame['frame'] for frame in frame_data)
                    min_frame = min(frame['frame'] for frame in frame_data)
                    # 실제 프레임 수로 길이 계산 (오프셋 보정)
                    actual_frames = max_frame - min_frame
                    return actual_frames / self.config.target_fps
        return 0
    
    # 불필요한 복잡한 매칭 함수들 제거됨 (단순 파일명 매칭으로 대체)
    
    def create_labeled_dataset(self) -> pd.DataFrame:
        """개선된 레이블된 데이터셋 생성"""
        print(f"🏷️ Enhanced labeled dataset creation")
        self.logger.info("레이블된 데이터셋 생성 시작")
        
        start_time = time.time()
        labeled_data = []
        
        # 데이터 수집 통계
        stats = {
            'total_frames': 0,
            'frames_with_hands': 0,
            'midi_matched_frames': 0,
            'processing_errors': 0
        }
        
        for video_name, data in self.combined_data.items():
            try:
                depth_data = data['depth_data']
                midi_data = data['midi_data']
                match_quality = data.get('match_quality', 'none')
                
                # 품질이 너무 낮은 매칭은 제외 (더 엄격한 기준)
                if match_quality in ['poor', 'none']:
                    print(f"   ⚠️ {video_name}: 매칭 품질이 낮아 제외됨 (품질: {match_quality})")
                    continue
                
                # 추가 검증: 길이 차이가 너무 크면 제외
                if data['has_midi'] and midi_data:
                    video_duration = self._calculate_video_duration(video_name)
                    midi_duration = midi_data.get('duration', 0)
                    if video_duration > 0 and midi_duration > 0:
                        duration_ratio = min(video_duration, midi_duration) / max(video_duration, midi_duration)
                        if duration_ratio < 0.7:  # 70% 미만 일치면 제외
                            print(f"   ⚠️ {video_name}: 길이 차이가 커서 제외됨 (비율: {duration_ratio:.2f})")
                            continue
                
                # 프레임별 데이터 처리 (벡터화 최적화)
                frame_data = depth_data.get('frame_data', [])
                stats['total_frames'] += len(frame_data)
                
                # 프레임 오프셋 계산 (첫 프레임이 0이 아닐 수 있음)
                first_frame = min(frame['frame'] for frame in frame_data) if frame_data else 0
                
                for frame_info in frame_data:
                    frame_idx = frame_info['frame']
                    # 프레임 오프셋 보정: 첫 프레임을 0으로 만듦
                    adjusted_frame = frame_idx - first_frame
                    frame_time = adjusted_frame / self.config.target_fps
                    hands = frame_info.get('hands', [])
                    
                    if hands:
                        stats['frames_with_hands'] += 1
                    
                    for hand_info in hands:
                        hand_type = hand_info['type']
                        depth = hand_info['depth']
                        
                        # 깊이 값 범위 검증
                        if not (self.config.depth_range[0] <= depth <= self.config.depth_range[1]):
                            continue  # 범위 밖 데이터 제외
                        
                        # 개선된 MIDI 구간 기반 레이블 생성
                        is_playing, active_notes_count = self._determine_playing_status(
                            frame_time, hand_type, midi_data, data['has_midi']
                        )
                        
                        if data['has_midi']:
                            stats['midi_matched_frames'] += 1
                        
                        labeled_data.append({
                            'video_name': video_name,
                            'frame': frame_idx,
                            'time': frame_time,
                            'hand_type': hand_type,
                            'depth': depth,
                            'is_playing': is_playing,
                            'has_midi': data['has_midi'],
                            'match_quality': match_quality,
                            'active_notes': active_notes_count
                        })
                        
            except Exception as e:
                stats['processing_errors'] += 1
                self.logger.error(f"데이터 처리 오류 {video_name}: {e}")
                print(f"   ❌ {video_name}: 처리 오류")
        
        # DataFrame 생성 및 최적화
        df = pd.DataFrame(labeled_data)
        
        if len(df) > 0:
            # 데이터 타입 최적화
            df['frame'] = df['frame'].astype('int32')
            df['time'] = df['time'].astype('float32')
            df['depth'] = df['depth'].astype('float32')
            df['is_playing'] = df['is_playing'].astype('bool')
            df['has_midi'] = df['has_midi'].astype('bool')
            df['active_notes'] = df['active_notes'].astype('int16')
            
            # 결과 통계
            processing_time = time.time() - start_time
            
            print(f"📊 Dataset creation completed ({processing_time:.2f}s):")
            print(f"   Total samples: {len(df):,}")
            print(f"   Left hand: {len(df[df['hand_type'] == 'Left']):,}")
            print(f"   Right hand: {len(df[df['hand_type'] == 'Right']):,}")
            print(f"   With MIDI: {len(df[df['has_midi'] == True]):,}")
            
            if 'is_playing' in df.columns:
                playing_count = len(df[df['is_playing'] == True])
                playing_ratio = playing_count / len(df) * 100
                print(f"   Playing: {playing_count:,} ({playing_ratio:.1f}%)")
                print(f"   Not playing: {len(df) - playing_count:,} ({100-playing_ratio:.1f}%)")
            
            # 품질별 분포
            quality_counts = df['match_quality'].value_counts()
            print(f"   Match quality distribution: {dict(quality_counts)}")
            
            self.logger.info(f"데이터셋 생성 완료: {len(df)}개 샘플, {processing_time:.2f}초")
        
        return df
    
    def _determine_playing_status(self, frame_time: float, hand_type: str, 
                                 midi_data: Optional[Dict], has_midi: bool) -> Tuple[bool, int]:
        """개선된 연주 상태 판정"""
        if not has_midi or not midi_data:
            return False, 0
        
        note_segments = midi_data['note_segments'].get(hand_type, [])
        active_notes_count = 0
        
        # 현재 시간에 활성화된 모든 음표 확인
        for segment in note_segments:
            if segment['start_time'] <= frame_time <= segment['end_time']:
                active_notes_count += 1
        
        # 다성 연주 고려: 하나 이상의 음표가 활성화되면 연주 중
        is_playing = active_notes_count > 0
        
        return is_playing, active_notes_count
    
    def analyze_depth_distribution(self, df: pd.DataFrame):
        """깊이 데이터 분포를 분석합니다"""
        print(f"📈 깊이 데이터 분포 분석")
        
        # 기본 통계
        stats_results = {}
        
        for hand_type in ['Left', 'Right']:
            hand_data = df[df['hand_type'] == hand_type]['depth']
            if len(hand_data) == 0:
                continue
                
            stats_results[hand_type] = {
                'count': len(hand_data),
                'mean': hand_data.mean(),
                'std': hand_data.std(),
                'min': hand_data.min(),
                'max': hand_data.max(),
                'median': hand_data.median(),
                'q25': hand_data.quantile(0.25),
                'q75': hand_data.quantile(0.75),
                'q90': hand_data.quantile(0.90),
                'q95': hand_data.quantile(0.95)
            }
        
        # MIDI 있는 데이터의 연주/미연주 분포
        midi_stats = {}
        midi_df = df[df['has_midi'] == True]
        
        if len(midi_df) > 0:
            for hand_type in ['Left', 'Right']:
                hand_data = midi_df[midi_df['hand_type'] == hand_type]
                if len(hand_data) == 0:
                    continue
                
                playing_data = hand_data[hand_data['is_playing'] == True]['depth']
                not_playing_data = hand_data[hand_data['is_playing'] == False]['depth']
                
                midi_stats[hand_type] = {
                    'playing': {
                        'count': len(playing_data),
                        'mean': playing_data.mean() if len(playing_data) > 0 else 0,
                        'std': playing_data.std() if len(playing_data) > 0 else 0,
                        'median': playing_data.median() if len(playing_data) > 0 else 0
                    },
                    'not_playing': {
                        'count': len(not_playing_data),
                        'mean': not_playing_data.mean() if len(not_playing_data) > 0 else 0,
                        'std': not_playing_data.std() if len(not_playing_data) > 0 else 0,
                        'median': not_playing_data.median() if len(not_playing_data) > 0 else 0
                    }
                }
        
        self.analysis_results['distribution'] = {
            'overall_stats': stats_results,
            'midi_based_stats': midi_stats
        }
        
        # 결과 출력
        print(f"\n📊 전체 깊이 분포:")
        for hand_type, stats in stats_results.items():
            print(f"   {hand_type}손:")
            print(f"     평균: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"     중앙값: {stats['median']:.3f}")
            print(f"     범위: {stats['min']:.3f} ~ {stats['max']:.3f}")
            print(f"     90분위: {stats['q90']:.3f}")
        
        if midi_stats:
            print(f"\n🎵 MIDI 기반 분포:")
            for hand_type, stats in midi_stats.items():
                playing = stats['playing']
                not_playing = stats['not_playing']
                print(f"   {hand_type}손:")
                print(f"     연주 중: {playing['mean']:.3f} ± {playing['std']:.3f} ({playing['count']}개)")
                print(f"     미연주: {not_playing['mean']:.3f} ± {not_playing['std']:.3f} ({not_playing['count']}개)")
        
        return stats_results, midi_stats
    
    def calculate_optimal_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """MIDI 기반 지도학습으로 최적 임계값을 계산합니다 (미세한 차이에 최적화)"""
        print(f"🎯 MIDI 기반 최적 임계값 계산")
        
        threshold_results = {}
        
        for hand_type in ['Left', 'Right']:
            hand_data = df[df['hand_type'] == hand_type]
            if len(hand_data) == 0:
                continue
            
            hand_thresholds = {}
            
            # MIDI 기반 최적화만 사용 (미세한 차이에서 유효한 방법들)
            midi_hand_data = hand_data[hand_data['has_midi'] == True]
            if len(midi_hand_data) > 0 and 'is_playing' in midi_hand_data.columns:
                playing_depths = midi_hand_data[midi_hand_data['is_playing'] == True]['depth'].values
                not_playing_depths = midi_hand_data[midi_hand_data['is_playing'] == False]['depth'].values
                
                if len(playing_depths) > 0 and len(not_playing_depths) > 0:
                    # ROC 커브 기반 최적 임계값
                    if SKLEARN_AVAILABLE:
                        try:
                            y_true = midi_hand_data['is_playing'].values
                            y_scores = midi_hand_data['depth'].values
                            
                            # ROC 커브 계산
                            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                            
                            # Youden's J statistic 최대화
                            j_scores = tpr - fpr
                            best_idx = np.argmax(j_scores)
                            hand_thresholds['roc_optimal'] = thresholds[best_idx]
                            
                            # AUC 계산
                            auc_score = auc(fpr, tpr)
                            hand_thresholds['auc_score'] = auc_score
                            
                        except:
                            pass
                    
                    # 두 그룹의 평균값 사이
                    playing_mean = np.mean(playing_depths)
                    not_playing_mean = np.mean(not_playing_depths)
                    hand_thresholds['midi_mean_between'] = (playing_mean + not_playing_mean) / 2
                    
                    # Fisher's discriminant ratio 최대화
                    hand_thresholds['fisher_optimal'] = self._calculate_fisher_threshold(
                        playing_depths, not_playing_depths
                    )
            
            # Valley detection 제거됨 - 미세한 차이에서는 명확한 골짜기가 없음
            
            threshold_results[hand_type] = hand_thresholds
        
        # 결과 출력
        print(f"\n🎯 계산된 임계값들:")
        for hand_type, thresholds in threshold_results.items():
            print(f"\n   {hand_type}손:")
            for method, value in thresholds.items():
                if method == 'auc_score':
                    print(f"     AUC 점수: {value:.3f}")
                else:
                    print(f"     {method}: {value:.3f}")
        
        self.analysis_results['thresholds'] = threshold_results
        return threshold_results
    
    def _calculate_fisher_threshold(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Fisher's linear discriminant 기반 최적 임계값 계산"""
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1), np.var(group2)
            
            # Fisher's threshold
            threshold = (var2 * mean1 + var1 * mean2) / (var1 + var2)
            return threshold
        except:
            return (np.mean(group1) + np.mean(group2)) / 2
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create analysis result visualizations"""
        print(f"📊 Creating enhanced visualizations")
        
        # 1. Comprehensive analysis dashboard
        self._plot_comprehensive_dashboard(df)
        
        # 2. MIDI-based detailed analysis (if available)
        midi_df = df[df['has_midi'] == True]
        if len(midi_df) > 0:
            self._plot_midi_detailed_analysis(midi_df)
        
        # 3. ROC curve and performance analysis
        if len(midi_df) > 0:
            self._plot_roc_analysis(midi_df)
        
        # 4. Threshold effects visualization
        if 'thresholds' in self.analysis_results:
            self._plot_threshold_effects(df)
        
        print(f"📊 Enhanced visualizations completed: {self.output_dir}")
    
    def _plot_comprehensive_dashboard(self, df: pd.DataFrame):
        """Comprehensive Analysis Dashboard"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 12))
        
        # Default font settings
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        for hand_idx, hand_type in enumerate(['Left', 'Right']):
            hand_data = df[df['hand_type'] == hand_type]['depth']
            
            if len(hand_data) == 0:
                continue
            
            # 1. 전체 분포 (밀도 + 히스토그램)
            ax1 = plt.subplot(3, 4, hand_idx*2 + 1)
            
            # 히스토그램 (0.5-1.5 범위)
            bins = np.linspace(0.5, 1.5, 50)
            n, bins, patches = ax1.hist(hand_data, bins=bins, alpha=0.6, 
                                       color='lightblue', edgecolor='navy', density=True)
            
            # Density curve
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(hand_data)
            x_range = np.linspace(0.5, 1.5, 100)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density Curve')
            
            # X축 범위 고정
            ax1.set_xlim(0.5, 1.5)
            
            # Statistics values
            mean_val = hand_data.mean()
            median_val = hand_data.median()
            q90_val = hand_data.quantile(0.9)
            
            ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_val:.3f}')
            ax1.axvline(median_val, color='green', linestyle='--', linewidth=2,
                       label=f'Median: {median_val:.3f}')
            ax1.axvline(q90_val, color='orange', linestyle='--', linewidth=2,
                       label=f'90th: {q90_val:.3f}')
            
            ax1.set_title(f'{hand_type} Hand Depth Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Depth Value', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Statistics info text box
            stats_text = f'Samples: {len(hand_data):,}\nStd Dev: {hand_data.std():.3f}\nRange: {hand_data.min():.3f}~{hand_data.max():.3f}'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=9)
            
            # 2. MIDI 기반 비교 (있는 경우)
            midi_hand_data = df[(df['hand_type'] == hand_type) & (df['has_midi'] == True)]
            if len(midi_hand_data) > 0:
                ax2 = plt.subplot(3, 4, hand_idx*2 + 2)
                
                playing_data = midi_hand_data[midi_hand_data['is_playing'] == True]['depth']
                not_playing_data = midi_hand_data[midi_hand_data['is_playing'] == False]['depth']
                
                # Density plot comparison (0.5-1.5 범위)
                x_range = np.linspace(0.5, 1.5, 100)
                
                if len(playing_data) > 0:
                    kde_playing = gaussian_kde(playing_data)
                    ax2.plot(x_range, kde_playing(x_range), 'r-', linewidth=3, 
                            label=f'Playing ({len(playing_data):,} samples)')
                    ax2.fill_between(x_range, kde_playing(x_range), alpha=0.3, color='red')
                
                if len(not_playing_data) > 0:
                    kde_not_playing = gaussian_kde(not_playing_data)
                    ax2.plot(x_range, kde_not_playing(x_range), 'b-', linewidth=3,
                            label=f'Not Playing ({len(not_playing_data):,} samples)')
                    ax2.fill_between(x_range, kde_not_playing(x_range), alpha=0.3, color='blue')
                
                # X축 범위 고정
                ax2.set_xlim(0.5, 1.5)
                
                ax2.set_title(f'{hand_type} Hand MIDI-based Comparison', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Depth Value', fontsize=12)
                ax2.set_ylabel('Density', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # Mean difference display
                if len(playing_data) > 0 and len(not_playing_data) > 0:
                    diff = not_playing_data.mean() - playing_data.mean()
                    ax2.text(0.02, 0.98, f'Mean Diff: {diff:.3f}', 
                            transform=ax2.transAxes,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                            verticalalignment='top', fontsize=10)
            
            # 3. 임계값 비교 (하단)
            if 'thresholds' in self.analysis_results and hand_type in self.analysis_results['thresholds']:
                ax3 = plt.subplot(3, 2, 3 + hand_idx)
                
                thresholds = self.analysis_results['thresholds'][hand_type]
                
                # 히스토그램 배경 (0.5-1.5 범위)
                bins = np.linspace(0.5, 1.5, 40)
                ax3.hist(hand_data, bins=bins, alpha=0.4, color='lightgray', density=True)
                
                # X축 범위 고정
                ax3.set_xlim(0.5, 1.5)
                
                # Display thresholds with color coding
                threshold_colors = {
                    'roc_optimal': 'red',
                    'fisher_optimal': 'blue', 
                    'q90': 'orange',
                    'q85': 'green',
                    'valley_detection': 'purple',
                    'midi_mean_between': 'brown'
                }
                
                threshold_names = {
                    'roc_optimal': 'ROC Optimal',
                    'fisher_optimal': 'Fisher LDA',
                    'q90': '90th Percentile',
                    'q85': '85th Percentile', 
                    'valley_detection': 'Valley Detection',
                    'midi_mean_between': 'MIDI Mean'
                }
                
                for method, value in thresholds.items():
                    if method in threshold_colors:
                        color = threshold_colors[method]
                        name = threshold_names.get(method, method)
                        ax3.axvline(value, color=color, linestyle='--', linewidth=3,
                                   label=f'{name}: {value:.3f}')
                
                ax3.set_title(f'{hand_type} Hand Threshold Comparison', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Depth Value', fontsize=12)
                ax3.set_ylabel('Density', fontsize=12)
                ax3.legend(fontsize=9, loc='upper right')
                ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Depth Data Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_midi_detailed_analysis(self, midi_df: pd.DataFrame):
        """MIDI-based Detailed Analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Default font settings
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        for hand_idx, hand_type in enumerate(['Left', 'Right']):
            hand_data = midi_df[midi_df['hand_type'] == hand_type]
            
            if len(hand_data) == 0:
                continue
            
            playing_data = hand_data[hand_data['is_playing'] == True]['depth']
            not_playing_data = hand_data[hand_data['is_playing'] == False]['depth']
            
            # 1. Detailed histogram (0.5-1.5 범위)
            ax1 = axes[hand_idx, 0]
            bins = np.linspace(0.5, 1.5, 40)
            
            ax1.hist(not_playing_data, bins=bins, alpha=0.6, color='blue', 
                    label=f'Not Playing ({len(not_playing_data):,})', density=True)
            ax1.hist(playing_data, bins=bins, alpha=0.6, color='red',
                    label=f'Playing ({len(playing_data):,})', density=True)
            
            # X축 범위 고정
            ax1.set_xlim(0.5, 1.5)
            
            ax1.set_title(f'{hand_type} Hand Detailed Distribution', fontweight='bold')
            ax1.set_xlabel('Depth Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Boxplot + Violin plot
            ax2 = axes[hand_idx, 1]
            data_for_plot = [not_playing_data.dropna(), playing_data.dropna()]
            labels = ['Not Playing', 'Playing']
            
            # Violin plot
            violin_parts = ax2.violinplot(data_for_plot, positions=[1, 2], showmeans=True)
            
            # Color settings
            violin_parts['bodies'][0].set_facecolor('blue')
            violin_parts['bodies'][0].set_alpha(0.6)
            violin_parts['bodies'][1].set_facecolor('red') 
            violin_parts['bodies'][1].set_alpha(0.6)
            
            ax2.set_xticks([1, 2])
            ax2.set_xticklabels(labels)
            ax2.set_title(f'{hand_type} Hand Distribution Shape', fontweight='bold')
            ax2.set_ylabel('Depth Value')
            ax2.grid(True, alpha=0.3)
            
            # 3. Statistics comparison table
            ax3 = axes[hand_idx, 2]
            ax3.axis('off')
            
            # Statistics table data
            stats_data = []
            if len(playing_data) > 0:
                stats_data.append(['Playing', f'{len(playing_data):,}', 
                                 f'{playing_data.mean():.3f}', f'{playing_data.std():.3f}',
                                 f'{playing_data.median():.3f}'])
            if len(not_playing_data) > 0:
                stats_data.append(['Not Playing', f'{len(not_playing_data):,}',
                                 f'{not_playing_data.mean():.3f}', f'{not_playing_data.std():.3f}',
                                 f'{not_playing_data.median():.3f}'])
            
            if stats_data:
                table = ax3.table(cellText=stats_data,
                                colLabels=['Status', 'Count', 'Mean', 'Std Dev', 'Median'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0.3, 1, 0.4])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                
                # Header styling
                for i in range(5):
                    table[(0, i)].set_facecolor('#40466e')
                    table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax3.set_title(f'{hand_type} Hand Statistics Summary', fontweight='bold')
        
        plt.suptitle('MIDI-based Detailed Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'midi_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_analysis(self, midi_df: pd.DataFrame):
        """ROC Curve and Performance Analysis"""
        if not SKLEARN_AVAILABLE:
            print("⚠️ scikit-learn is required for ROC analysis")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Default font settings
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        for hand_idx, hand_type in enumerate(['Left', 'Right']):
            hand_data = midi_df[midi_df['hand_type'] == hand_type]
            
            if len(hand_data) == 0 or 'is_playing' not in hand_data.columns:
                continue
            
            y_true = hand_data['is_playing'].values
            y_scores = hand_data['depth'].values
            
            if len(np.unique(y_true)) < 2:
                continue
            
            try:
                # ROC 커브
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                ax1 = axes[0, hand_idx]
                ax1.plot(fpr, tpr, color='darkorange', linewidth=2,
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax1.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
                
                # 최적점 표시
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_threshold = thresholds[best_idx]
                
                ax1.plot(fpr[best_idx], tpr[best_idx], marker='o', markersize=10, 
                        color='red', label=f'Optimal Point (Threshold={best_threshold:.3f})')
                
                ax1.set_xlim([0.0, 1.0])
                ax1.set_ylim([0.0, 1.05])
                ax1.set_xlabel('False Positive Rate')
                ax1.set_ylabel('True Positive Rate') 
                ax1.set_title(f'{hand_type} Hand ROC Curve', fontweight='bold')
                ax1.legend(loc="lower right")
                ax1.grid(True, alpha=0.3)
                
                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
                pr_auc = auc(recall, precision)
                
                ax2 = axes[1, hand_idx]
                ax2.plot(recall, precision, color='blue', linewidth=2,
                        label=f'PR curve (AUC = {pr_auc:.3f})')
                
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
                ax2.set_title(f'{hand_type} Hand Precision-Recall Curve', fontweight='bold')
                ax2.legend(loc="lower left")
                ax2.grid(True, alpha=0.3)
                
                # Performance metrics text
                info_text = f'ROC AUC: {roc_auc:.3f}\nPR AUC: {pr_auc:.3f}\nOptimal Threshold: {best_threshold:.3f}'
                ax1.text(0.6, 0.2, info_text, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                print(f"ROC analysis error ({hand_type} hand): {e}")
        
        plt.suptitle('ROC and Precision-Recall Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_effects(self, df: pd.DataFrame):
        """Threshold Effects Visualization"""
        if 'thresholds' not in self.analysis_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Default font settings
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        for hand_idx, hand_type in enumerate(['Left', 'Right']):
            if hand_type not in self.analysis_results['thresholds']:
                continue
                
            hand_data = df[df['hand_type'] == hand_type]['depth']
            thresholds = self.analysis_results['thresholds'][hand_type]
            
            # 1. Threshold classification effects
            ax1 = axes[0, hand_idx]
            
            # Histogram background (0.5-1.5 범위)
            bins = np.linspace(0.5, 1.5, 50)
            n, bins, patches = ax1.hist(hand_data, bins=bins, alpha=0.3, color='lightgray', density=True)
            
            # X축 범위 고정
            ax1.set_xlim(0.5, 1.5)
            
            # Show only MIDI-based valid thresholds
            important_thresholds = {
                'roc_optimal': ('ROC Optimal', 'red', '-'),
                'fisher_optimal': ('Fisher LDA', 'blue', '--'),
                'midi_mean_between': ('MIDI Mean', 'green', '-.'),
                'current': ('Current (0.9)', 'purple', ':')
            }
            
            # Add current threshold
            if 'current' not in thresholds:
                thresholds['current'] = 0.9
            
            for method, value in thresholds.items():
                if method in important_thresholds:
                    name, color, style = important_thresholds[method]
                    ax1.axvline(value, color=color, linestyle=style, linewidth=3,
                               label=f'{name}: {value:.3f}')
                    
                    # Calculate exceed ratio
                    exceed_ratio = (hand_data > value).mean() * 100
                    ax1.text(value, ax1.get_ylim()[1]*0.8, f'{exceed_ratio:.1f}%',
                            rotation=90, ha='center', va='bottom', color=color, fontweight='bold')
            
            ax1.set_title(f'{hand_type} Hand Threshold Effects Comparison', fontweight='bold')
            ax1.set_xlabel('Depth Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Floating ratio by threshold
            ax2 = axes[1, hand_idx]
            
            threshold_values = []
            floating_ratios = []
            threshold_labels = []
            
            for method, value in thresholds.items():
                if method != 'auc_score' and method in important_thresholds:
                    threshold_values.append(value)
                    floating_ratio = (hand_data > value).mean() * 100
                    floating_ratios.append(floating_ratio)
                    threshold_labels.append(important_thresholds[method][0])
            
            # Bar chart
            bars = ax2.bar(threshold_labels, floating_ratios, 
                          color=['red', 'blue', 'green', 'purple'][:len(threshold_labels)],
                          alpha=0.7)
            
            # Value display
            for bar, ratio in zip(bars, floating_ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title(f'{hand_type} Hand Floating Ratio by Threshold', fontweight='bold')
            ax2.set_ylabel('Floating Ratio (%)')
            ax2.set_ylim(0, max(floating_ratios) * 1.2)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Rotate X-axis labels
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Threshold Effects Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """분석 결과를 바탕으로 추천 임계값을 생성합니다"""
        print(f"💡 추천 임계값 생성")
        
        recommendations = {}
        
        if 'thresholds' in self.analysis_results:
            for hand_type, thresholds in self.analysis_results['thresholds'].items():
                hand_recommendations = {}
                
                # 1. ROC 기반 (가장 우선)
                if 'roc_optimal' in thresholds:
                    hand_recommendations['primary'] = {
                        'value': thresholds['roc_optimal'],
                        'method': 'ROC Optimal (Youden\'s J)',
                        'confidence': 'High',
                        'reason': 'MIDI 온셋 기반 최적화'
                    }
                
                # 2. Fisher discriminant (두 번째 우선)
                elif 'fisher_optimal' in thresholds:
                    hand_recommendations['primary'] = {
                        'value': thresholds['fisher_optimal'],
                        'method': 'Fisher Discriminant',
                        'confidence': 'High',
                        'reason': 'MIDI 기반 그룹 분리 최적화'
                    }
                
                # 3. 다른 MIDI 기반 방법
                elif 'midi_mean_between' in thresholds:
                    hand_recommendations['primary'] = {
                        'value': thresholds['midi_mean_between'],
                        'method': 'MIDI Mean Between',
                        'confidence': 'Medium',
                        'reason': 'MIDI 기반 연주/비연주 평균값'
                    }
                
                # 대안 추천
                alternatives = []
                for method, value in thresholds.items():
                    if method not in ['auc_score'] and (
                        'primary' not in hand_recommendations or 
                        value != hand_recommendations['primary']['value']
                    ):
                        alternatives.append({
                            'value': value,
                            'method': method
                        })
                
                # 값 기준으로 정렬
                alternatives.sort(key=lambda x: x['value'])
                hand_recommendations['alternatives'] = alternatives[:3]  # 상위 3개
                
                recommendations[hand_type] = hand_recommendations
        
        # 최종 추천값 출력
        print(f"\n💡 추천 임계값:")
        for hand_type, rec in recommendations.items():
            if 'primary' in rec:
                primary = rec['primary']
                print(f"\n   {hand_type}손:")
                print(f"     🎯 추천값: {primary['value']:.3f}")
                print(f"     📊 방법: {primary['method']}")
                print(f"     ✅ 신뢰도: {primary['confidence']}")
                print(f"     💭 근거: {primary['reason']}")
                
                if rec.get('alternatives'):
                    print(f"     📋 대안:")
                    for alt in rec['alternatives']:
                        print(f"        {alt['method']}: {alt['value']:.3f}")
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def save_analysis_report(self):
        """분석 결과를 종합 보고서로 저장합니다"""
        print(f"📄 분석 보고서 저장")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'configuration': {
                'depth_data_dir': str(self.config.depth_data_dir),
                'midi_dir': str(self.config.midi_dir) if self.config.midi_dir else None,
                'target_fps': self.config.target_fps
            },
            'datasets': {
                'total_videos': len(self.combined_data),
                'videos_with_midi': len([d for d in self.combined_data.values() if d['has_midi']]),
                'videos_without_midi': len([d for d in self.combined_data.values() if not d['has_midi']])
            },
            'analysis_results': self.analysis_results
        }
        
        # JSON 저장
        report_path = self.output_dir / 'threshold_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 텍스트 보고서 저장
        self._save_text_report()
        
        print(f"📄 보고서 저장 완료:")
        print(f"   JSON: {report_path}")
        print(f"   텍스트: {self.output_dir / 'threshold_analysis_summary.txt'}")
    
    def _save_text_report(self):
        """텍스트 형태의 요약 보고서 저장"""
        report_path = self.output_dir / 'threshold_analysis_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("깊이 기반 임계값 분석 보고서\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"데이터 경로: {self.config.depth_data_dir}\n")
            f.write(f"MIDI 경로: {self.config.midi_dir}\n\n")
            
            # 데이터셋 정보
            f.write("📊 데이터셋 정보:\n")
            total_videos = len(self.combined_data)
            with_midi = len([d for d in self.combined_data.values() if d['has_midi']])
            f.write(f"   총 비디오: {total_videos}개\n")
            f.write(f"   MIDI 있음: {with_midi}개\n")
            f.write(f"   MIDI 없음: {total_videos - with_midi}개\n\n")
            
            # 추천 임계값
            if 'recommendations' in self.analysis_results:
                f.write("🎯 추천 임계값:\n")
                for hand_type, rec in self.analysis_results['recommendations'].items():
                    if 'primary' in rec:
                        primary = rec['primary']
                        f.write(f"\n   {hand_type}손:\n")
                        f.write(f"     추천값: {primary['value']:.3f}\n")
                        f.write(f"     방법: {primary['method']}\n")
                        f.write(f"     신뢰도: {primary['confidence']}\n")
                        f.write(f"     근거: {primary['reason']}\n")
            
            # 분포 통계
            if 'distribution' in self.analysis_results:
                f.write("\n📈 깊이 분포 통계:\n")
                overall_stats = self.analysis_results['distribution']['overall_stats']
                for hand_type, stats in overall_stats.items():
                    f.write(f"\n   {hand_type}손:\n")
                    f.write(f"     평균: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                    f.write(f"     중앙값: {stats['median']:.3f}\n")
                    f.write(f"     90분위: {stats['q90']:.3f}\n")
                    f.write(f"     범위: {stats['min']:.3f} ~ {stats['max']:.3f}\n")
    
    def run_full_analysis(self) -> Optional[Dict[str, Any]]:
        """개선된 전체 분석 파이프라인"""
        start_time = time.time()
        
        print(f"🚀 Enhanced depth threshold analysis pipeline")
        print("=" * 70)
        
        analysis_steps = [
            ("Data Loading", self.load_depth_data),
            ("MIDI Matching", self.match_depth_and_midi), 
            ("Dataset Creation", self.create_labeled_dataset),
            ("Distribution Analysis", None),  # 특별 처리
            ("Threshold Calculation", None),  # 특별 처리
            ("Visualization", None),  # 특별 처리
            ("Recommendations", self.generate_recommendations),
            ("Report Generation", self.save_analysis_report)
        ]
        
        results = {
            'success': False,
            'execution_time': 0,
            'steps_completed': 0,
            'total_steps': len(analysis_steps),
            'data': {}
        }
        
        try:
            df = None
            
            for step_idx, (step_name, step_func) in enumerate(analysis_steps, 1):
                step_start = time.time()
                print(f"\n{step_idx}️⃣ {step_name}")
                self.logger.info(f"분석 단계 시작: {step_name}")
                
                try:
                    if step_name == "Data Loading":
                        datasets = step_func()
                        if not datasets:
                            raise ValueError("No valid depth data found")
                        results['data']['datasets_count'] = len(datasets)
                        
                    elif step_name == "MIDI Matching":
                        combined_data = step_func()
                        if not combined_data:
                            raise ValueError("No data available for matching")
                        
                        midi_count = len([d for d in combined_data.values() if d['has_midi']])
                        results['data']['midi_matches'] = midi_count
                        
                    elif step_name == "Dataset Creation":
                        df = step_func()
                        if len(df) == 0:
                            raise ValueError("No valid samples in dataset")
                        
                        results['data']['total_samples'] = len(df)
                        results['data']['midi_samples'] = len(df[df['has_midi'] == True])
                        
                    elif step_name == "Distribution Analysis":
                        if df is None:
                            raise ValueError("No dataset available")
                        distribution_stats, midi_stats = self.analyze_depth_distribution(df)
                        results['data']['distribution_stats'] = distribution_stats
                        
                    elif step_name == "Threshold Calculation":
                        if df is None:
                            raise ValueError("No dataset available")
                        threshold_results = self.calculate_optimal_thresholds(df)
                        results['data']['thresholds'] = threshold_results
                        
                    elif step_name == "Visualization":
                        if df is None:
                            raise ValueError("No dataset available")
                        self.create_visualizations(df)
                        results['data']['visualizations_created'] = True
                        
                    elif step_name == "Recommendations":
                        recommendations = step_func()
                        results['data']['recommendations'] = recommendations
                        
                    elif step_name == "Report Generation":
                        step_func()
                        results['data']['report_generated'] = True
                    
                    step_time = time.time() - step_start
                    results['steps_completed'] += 1
                    
                    print(f"   ✅ {step_name} completed ({step_time:.2f}s)")
                    self.logger.info(f"단계 완료: {step_name} ({step_time:.2f}초)")
                    
                except Exception as step_error:
                    step_time = time.time() - step_start
                    print(f"   ❌ {step_name} failed: {step_error}")
                    self.logger.error(f"단계 실패: {step_name}: {step_error}")
                    
                    # 일부 단계는 실패해도 계속 진행 가능
                    if step_name in ["Visualization", "Report Generation"]:
                        print(f"   ⚠️ Continuing despite {step_name} failure")
                        continue
                    else:
                        raise step_error
            
            # 성공적 완료
            total_time = time.time() - start_time
            results['success'] = True
            results['execution_time'] = total_time
            
            print(f"\n🎉 Analysis pipeline completed successfully!")
            print(f"⏱️ Total execution time: {total_time:.2f}s")
            print(f"📊 Steps completed: {results['steps_completed']}/{results['total_steps']}")
            print(f"📂 Results saved to: {self.output_dir}")
            
            # 핵심 결과 요약
            if 'recommendations' in results['data']:
                print(f"\n🎯 Key Results:")
                for hand_type, rec in results['data']['recommendations'].items():
                    if 'primary' in rec:
                        primary = rec['primary']
                        print(f"   {hand_type} hand: {primary['value']:.3f} ({primary['method']})")
            
            self.logger.info(f"전체 분석 완료: {total_time:.2f}초")
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            results['execution_time'] = total_time
            
            print(f"\n❌ Analysis pipeline failed: {e}")
            print(f"⏱️ Time until failure: {total_time:.2f}s")
            print(f"📊 Steps completed: {results['steps_completed']}/{results['total_steps']}")
            
            self.logger.error(f"전체 분석 실패: {e}")
            
            # 디버깅 정보 출력
            if self.logger.handlers:
                print(f"📋 Check log file: {self.output_dir}/analysis.log")
            
            import traceback
            traceback.print_exc()
            return results

def create_analysis_config(
    depth_dir: str = "depth_data",
    midi_dir: str = "/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert",
    output_dir: str = "threshold_analysis",
    **kwargs
) -> AnalysisConfig:
    """분석 설정 생성 헬퍼 함수"""
    config = AnalysisConfig(
        depth_data_dir=depth_dir,
        midi_dir=midi_dir,
        output_dir=output_dir
    )
    
    # 추가 설정 적용
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def run_enhanced_analysis(config: Optional[AnalysisConfig] = None):
    """개선된 분석 실행"""
    if config is None:
        config = create_analysis_config()
    
    print("🎯 Enhanced Depth Threshold Analyzer")
    print("=" * 70)
    print("📊 Purpose: Find optimal thresholds for PyTorch floating detector")
    print("🎵 Method: MIDI segment-based analysis + Statistical methods")
    print(f"🎵 MIDI path: {config.midi_dir}")
    print(f"📁 Output: {config.output_dir}")
    print("=" * 70)
    
    try:
        # 분석기 초기화
        analyzer = EnhancedDepthThresholdAnalyzer(config=config)
        
        # 전체 분석 실행
        results = analyzer.run_full_analysis()
        
        if results:
            print("\n✅ Analysis completed successfully!")
            print(f"📂 Results saved to: {config.output_dir}")
            print(f"📊 Check the generated visualizations and reports")
            return results
        else:
            print("\n❌ Analysis failed or produced no results")
            return None
            
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Depth Threshold Analyzer')
    parser.add_argument('--depth-dir', default="depth_data", 
                       help='Depth data directory (default: depth_data)')
    parser.add_argument('--midi-dir', default="/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert",
                       help='MIDI files directory')
    parser.add_argument('--output-dir', default="threshold_analysis",
                       help='Output directory (default: threshold_analysis)')
    parser.add_argument('--fps', type=int, default=20,
                       help='Target FPS (default: 20)')
    parser.add_argument('--hand-split', type=int, default=60,
                       help='MIDI note for hand split (default: 60/C4)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel processing')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = create_analysis_config(
        depth_dir=args.depth_dir,
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        target_fps=args.fps,
        hand_split_note=args.hand_split,
        parallel_processing=args.parallel,
        enable_caching=not args.no_cache,
        max_workers=args.workers
    )
    
    # 분석 실행
    run_enhanced_analysis(config)

if __name__ == "__main__":
    main() 
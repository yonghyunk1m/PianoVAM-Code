#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI 기반 Floating 값 분포 분석기
MIDI가 연주될 때 vs 조용할 때의 손 깊이값 분포 비교
"""

import mido
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
import re
from scipy import stats
from scipy.stats import shapiro, anderson, normaltest
import warnings
warnings.filterwarnings('ignore')


class MIDIFloatingDistributionAnalyzer:
    """MIDI 연주 상태에 따른 floating 값 분포 분석기"""
    
    def __init__(self, 
                 depth_data_dir: str = "depth_data",
                 midi_dir: str = "midiconvert",
                 target_fps: int = 20,
                 output_dir: str = "analysis_results"):
        self.depth_data_dir = Path(depth_data_dir)
        self.midi_dir = Path(midi_dir)
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.hand_split_note = 60  # Middle C
        
        # 출력 폴더 생성
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 분석 결과 저장 폴더: {self.output_dir}")
        
    def analyze_floating_distribution(self):
        """메인 분석 함수 - 모든 depth_data 통합 분석"""
        print("🎵 MIDI 기반 Floating 분포 분석 시작 (모든 depth_data)")
        print("=" * 60)
        
        # 1. 모든 depth_data JSON 파일 로드
        all_data = self._load_all_depth_data()
        if not all_data:
            return None
        
        # 2. 각 데이터셋에서 분석 데이터 수집
        combined_analysis_data = []
        successful_matches = 0
        
        for data_info in all_data:
            video_name = data_info['video_name']
            frame_data = data_info['frame_data']
            
            print(f"\n🔍 분석 중: {video_name}")
            
            # MIDI 데이터 로드
            midi_data = self._load_midi_data(video_name)
            if not midi_data:
                print(f"   ⚠️ MIDI 없음 - 깊이값만 수집")
                # MIDI 없어도 깊이값은 수집
                depth_only_data = self._extract_depth_from_frame_data(frame_data, video_name)
                combined_analysis_data.extend(depth_only_data)
                continue
            
            # MIDI 타임라인 생성
            midi_timeline = self._create_simple_timeline(midi_data)
            
            # 프레임 데이터와 MIDI 상태 매칭 (정확한 시간 정렬)
            analysis_data = self._match_frame_data_with_midi(frame_data, midi_timeline, video_name)
            combined_analysis_data.extend(analysis_data)
            successful_matches += 1
            
            print(f"   ✅ 매칭 완료: {len(analysis_data):,}개 데이터")
        
        print(f"\n📊 통합 분석 준비 완료:")
        print(f"   성공한 MIDI 매칭: {successful_matches}개")
        print(f"   총 분석 데이터: {len(combined_analysis_data):,}개")
        
        # 3. 기본 통계량 및 분포 분석
        basic_results = self._analyze_distributions(combined_analysis_data, "all_combined")
        
        # 4. 상세한 통계량 및 분포 형태 분석
        detailed_results = self._create_comprehensive_distribution_analysis(combined_analysis_data, "All Combined Data")
        
        return {
            'basic_analysis': basic_results,
            'detailed_analysis': detailed_results
        }
    
    def _load_all_depth_data(self):
        """모든 사용 가능한 depth_data JSON 파일 로드"""
        print("📁 모든 depth_data JSON 파일 로드 중...")
        
        # 사용 가능한 JSON 파일들 찾기
        json_files = list(self.depth_data_dir.glob("*_depth_data.json"))
        
        if not json_files:
            print("❌ depth_data JSON 파일을 찾을 수 없습니다.")
            return None
        
        print(f"   발견된 파일: {len(json_files)}개")
        
        # 모든 depth_data와 해당 video_name을 저장
        all_data = []
        total_files_loaded = 0
        total_depth_points = 0
        
        for json_file in json_files:
            try:
                print(f"   로딩: {json_file.name}")
                
                # JSON 로드
                with open(json_file, 'r') as f:
                    depth_data = json.load(f)
                
                # video_name 추출
                video_name = json_file.stem.replace('_depth_data', '')
                
                # 프레임 데이터 추출 (정확한 시간 정렬 위해)
                frame_data = depth_data.get('frame_data', [])
                
                # 레거시 depth_data도 가져오기 (참고용)
                left_depths = depth_data.get('depth_data', {}).get('Left', [])
                right_depths = depth_data.get('depth_data', {}).get('Right', [])
                
                total_points = len(frame_data)
                
                if total_points > 0:  # 데이터가 있는 경우만 추가
                    all_data.append({
                        'video_name': video_name,
                        'frame_data': frame_data,  # 정확한 프레임별 데이터
                        'left_depths': left_depths,  # 레거시 지원
                        'right_depths': right_depths,  # 레거시 지원
                        'total_points': total_points,
                        'metadata': {
                            'total_hands': depth_data.get('total_hands', 0),
                            'total_frames': depth_data.get('total_frames', 0),
                            'depth_range': depth_data.get('depth_range', [0, 1])
                        }
                    })
                    total_files_loaded += 1
                    total_depth_points += total_points
                    
                    # 손 타입별 통계 계산
                    left_count = sum(1 for frame in frame_data 
                                   for hand in frame.get('hands', []) 
                                   if hand.get('type') == 'Left')
                    right_count = sum(1 for frame in frame_data 
                                    for hand in frame.get('hands', []) 
                                    if hand.get('type') == 'Right')
                    
                    print(f"     ✅ {total_points:,}개 프레임 (L:{left_count}, R:{right_count})")
                else:
                    print(f"     ⚠️ 빈 데이터, 건너뜀")
                    
            except Exception as e:
                print(f"     ❌ 로드 실패: {e}")
                continue
        
        print(f"\n✅ 로드 완료:")
        print(f"   성공한 파일: {total_files_loaded}개")
        print(f"   총 깊이 데이터: {total_depth_points:,}개")
        
        return all_data if all_data else None
    
    def _load_midi_data(self, video_name: str):
        """MIDI 데이터 로드"""
        print("🎵 MIDI 데이터 로드 중...")
        
        # 매칭되는 MIDI 파일 찾기
        pattern = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', video_name)
        if not pattern:
            print(f"   ⚠️ 날짜 패턴 없음: {video_name}")
            return None
        
        target_pattern = pattern.group(1)
        midi_files = list(self.midi_dir.glob("*.mid"))
        
        for midi_file in midi_files:
            if target_pattern in midi_file.stem:
                print(f"   ✅ MIDI 매칭: {midi_file.name}")
                return self._parse_midi_simple(midi_file)
        
        print(f"   ❌ MIDI 파일 없음: {target_pattern}")
        return None
    
    def _parse_midi_simple(self, midi_file: Path):
        """간단한 MIDI 파싱"""
        try:
            mid = mido.MidiFile(midi_file)
            
            # 템포 추출
            tempo = 500000
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                        break
                if tempo != 500000:
                    break
            
            # 음표 세그먼트 추출
            note_segments = {'Left': [], 'Right': []}
            active_notes = {'Left': {}, 'Right': {}}
            
            for track in mid.tracks:
                track_time = 0
                for msg in track:
                    track_time += msg.time
                    
                    if msg.type in ['note_on', 'note_off']:
                        time_seconds = mido.tick2second(track_time, mid.ticks_per_beat, tempo)
                        hand_type = 'Right' if msg.note >= self.hand_split_note else 'Left'
                        
                        if msg.type == 'note_on' and msg.velocity > 0:
                            active_notes[hand_type][msg.note] = time_seconds
                        else:
                            if msg.note in active_notes[hand_type]:
                                start_time = active_notes[hand_type].pop(msg.note)
                                if time_seconds - start_time > 0.01:
                                    note_segments[hand_type].append({
                                        'start': start_time,
                                        'end': time_seconds
                                    })
            
            total_notes = sum(len(segs) for segs in note_segments.values())
            duration = max([seg['end'] for segs in note_segments.values() for seg in segs]) if total_notes > 0 else 0
            
            print(f"   🎵 파싱 완료: {total_notes}개 음표, {duration:.1f}초")
            
            return {
                'segments': note_segments,
                'duration': duration,
                'total_notes': total_notes
            }
            
        except Exception as e:
            print(f"   ❌ MIDI 파싱 실패: {e}")
            return None
    
    def _create_simple_timeline(self, midi_data: Dict):
        """간단한 MIDI 타임라인 생성"""
        duration = midi_data['duration']
        resolution = 0.05  # 50ms
        points = int(duration / resolution) + 1
        
        timeline = {
            'Left': np.zeros(points, dtype=bool),
            'Right': np.zeros(points, dtype=bool),
            'resolution': resolution,
            'duration': duration
        }
        
        for hand_type in ['Left', 'Right']:
            for segment in midi_data['segments'][hand_type]:
                start_idx = int(segment['start'] / resolution)
                end_idx = int(segment['end'] / resolution)
                
                start_idx = max(0, start_idx)
                end_idx = min(points - 1, end_idx)
                timeline[hand_type][start_idx:end_idx + 1] = True
        
        return timeline
    
    def _match_frame_data_with_midi(self, frame_data: List[dict], 
                                   midi_timeline: Dict, video_name: str):
        """프레임 데이터와 MIDI 상태 매칭 - 정확한 시간 정렬"""
        analysis_data = []
        
        for frame_info in frame_data:
            frame_number = frame_info.get('frame', 0)
            frame_time = frame_number / self.target_fps  # 20fps 기준 정확한 시간
            
            # 해당 프레임의 모든 손 처리
            for hand in frame_info.get('hands', []):
                hand_type = hand.get('type', 'Unknown')
                depth_value = hand.get('depth', 1.0)
                
                # MIDI 연주 상태 확인
                time_idx = int(frame_time / midi_timeline['resolution'])
                is_playing = False
                
                if hand_type in midi_timeline and 0 <= time_idx < len(midi_timeline[hand_type]):
                    is_playing = bool(midi_timeline[hand_type][time_idx])
                
                analysis_data.append({
                    'video_name': video_name,
                    'frame': frame_number,  # 정확한 프레임 번호
                    'time': frame_time,     # 정확한 시간
                    'hand_type': hand_type,
                    'depth': depth_value,
                    'is_playing': is_playing,
                    'midi_status': 'playing' if is_playing else 'quiet'
                })
        
        return analysis_data
    
    def _extract_depth_from_frame_data(self, frame_data: List[dict], video_name: str):
        """MIDI 없는 경우 프레임 데이터에서 직접 추출"""
        analysis_data = []
        
        for frame_info in frame_data:
            frame_number = frame_info.get('frame', 0)
            frame_time = frame_number / self.target_fps  # 20fps 기준 정확한 시간
            
            # 해당 프레임의 모든 손 처리
            for hand in frame_info.get('hands', []):
                hand_type = hand.get('type', 'Unknown')
                depth_value = hand.get('depth', 1.0)
                
                analysis_data.append({
                    'video_name': video_name,
                    'frame': frame_number,  # 정확한 프레임 번호
                    'time': frame_time,     # 정확한 시간
                    'hand_type': hand_type,
                    'depth': depth_value,
                    'is_playing': None,  # MIDI 정보 없음
                    'midi_status': 'unknown'
                })
        
        return analysis_data
    
    def _calculate_detailed_statistics(self, data: List[float], group_name: str) -> Dict:
        """상세한 기술통계량 계산"""
        if not data:
            return {}
        
        arr = np.array(data)
        
        return {
            'group': group_name,
            'count': len(arr),
            'mean': np.mean(arr),
            'median': np.median(arr),
            'std': np.std(arr, ddof=1),
            'var': np.var(arr, ddof=1),
            'min': np.min(arr),
            'max': np.max(arr),
            'q1': np.percentile(arr, 25),
            'q3': np.percentile(arr, 75),
            'iqr': np.percentile(arr, 75) - np.percentile(arr, 25),
            'range': np.max(arr) - np.min(arr),
            'cv': np.std(arr, ddof=1) / np.mean(arr) if np.mean(arr) != 0 else 0,  # 변동계수
            'skewness': stats.skew(arr),  # 왜도
            'kurtosis': stats.kurtosis(arr),  # 첨도
            'se_mean': np.std(arr, ddof=1) / np.sqrt(len(arr))  # 평균의 표준오차
        }
    
    def _test_normality(self, data: List[float], group_name: str) -> Dict:
        """정규성 검정"""
        if len(data) < 3:
            return {'group': group_name, 'error': 'Insufficient data'}
        
        arr = np.array(data)
        results = {'group': group_name}
        
        # Shapiro-Wilk 검정 (표본 크기 5000 이하 권장)
        if len(arr) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(arr)
                results['shapiro_stat'] = shapiro_stat
                results['shapiro_p'] = shapiro_p
                results['shapiro_normal'] = shapiro_p > 0.05
            except:
                results['shapiro_stat'] = None
                results['shapiro_normal'] = None
        
        # Anderson-Darling 검정
        try:
            anderson_result = anderson(arr, dist='norm')
            results['anderson_stat'] = anderson_result.statistic
            results['anderson_critical'] = anderson_result.critical_values[2]  # 5% 유의수준
            results['anderson_normal'] = anderson_result.statistic < anderson_result.critical_values[2]
        except:
            results['anderson_stat'] = None
            results['anderson_normal'] = None
        
        return results
    
    def _calculate_kde(self, data: np.ndarray, num_points: int = 1000):
        """KDE 계산"""
        from scipy.stats import gaussian_kde
        
        if len(data) < 2:
            return np.array([]), np.array([])
        
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), num_points)
        density = kde(x_range)
        
        return x_range, density
    
    def _create_statistics_table(self, stats_results: List[Dict], title: str = "Statistics Summary"):
        """통계량 비교 테이블 생성"""
        if not stats_results:
            return
        
        # 그래프 설정
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{title} - Statistical Comparison', fontsize=16, fontweight='bold')
        
        # 1. 기본 통계량 테이블
        ax1.axis('tight')
        ax1.axis('off')
        
        # 테이블 데이터 준비
        table_data = []
        headers = ['Statistic', 'Left Playing', 'Left Quiet', 'Right Playing', 'Right Quiet']
        
        stats_to_show = ['count', 'mean', 'median', 'std', 'cv', 'skewness', 'kurtosis', 'iqr']
        stat_names = ['Count', 'Mean', 'Median', 'Std Dev', 'CV', 'Skewness', 'Kurtosis', 'IQR']
        
        for stat, name in zip(stats_to_show, stat_names):
            row = [name]
            for group in ['Left_playing', 'Left_quiet', 'Right_playing', 'Right_quiet']:
                group_data = next((x for x in stats_results if x['group'] == group), {})
                if stat in group_data:
                    if stat == 'count':
                        row.append(f"{group_data[stat]:,}")
                    else:
                        row.append(f"{group_data[stat]:.4f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        table = ax1.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center', fontsize=10)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.0)
        
        # 헤더 스타일링
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('Detailed Statistics', fontweight='bold', pad=20)
        
        # 2. 주요 지표 비교 바차트
        groups = [x['group'].replace('_', ' ').title() for x in stats_results]
        means = [x.get('mean', 0) for x in stats_results]
        stds = [x.get('std', 0) for x in stats_results]
        
        x_pos = np.arange(len(groups))
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=['#FF6B6B', '#4ECDC4', '#FF6B6B', '#4ECDC4'],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Groups', fontweight='bold')
        ax2.set_ylabel('Depth Value', fontweight='bold')
        ax2.set_title('Mean Depth Values with Standard Deviation', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(groups, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = self.output_dir / "01_statistical_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 통계량 비교 테이블 저장: {filename}")
        plt.close()
    
    def _create_comprehensive_distribution_analysis(self, combined_data: List[Dict], dataset_name: str):
        """종합적인 분포 분석 시각화"""
        
        # 데이터 분리
        data_groups = {
            'Left_playing': [x['depth'] for x in combined_data if x['hand_type'] == 'Left' and x['is_playing'] == True],
            'Left_quiet': [x['depth'] for x in combined_data if x['hand_type'] == 'Left' and x['is_playing'] == False],
            'Right_playing': [x['depth'] for x in combined_data if x['hand_type'] == 'Right' and x['is_playing'] == True],
            'Right_quiet': [x['depth'] for x in combined_data if x['hand_type'] == 'Right' and x['is_playing'] == False]
        }
        
        print(f"\n📊 데이터 그룹 크기:")
        for group, data in data_groups.items():
            print(f"   {group}: {len(data):,}개")
        
        # 빈 그룹 제거
        data_groups = {k: v for k, v in data_groups.items() if len(v) > 0}
        
        # 1. 기본 통계량 계산
        print(f"\n📈 기본 통계량 분석 중...")
        stats_results = []
        normality_results = []
        
        for group_name, data in data_groups.items():
            stats_results.append(self._calculate_detailed_statistics(data, group_name))
            normality_results.append(self._test_normality(data, group_name))
        
        # 통계량 테이블 출력
        self._create_statistics_table(stats_results, dataset_name)
        
        # 2. 손별 분리된 종합 분포 분석
        self._create_hand_specific_analysis(data_groups, dataset_name, 'Left')
        self._create_hand_specific_analysis(data_groups, dataset_name, 'Right')
        
        # 3. 기존 종합 분포 분석 시각화 (참고용)
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'{dataset_name} - Overall Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 색상 설정
        colors = {'playing': '#FF6B6B', 'quiet': '#4ECDC4'}
        
        # 전체 데이터 범위 계산
        all_data = [val for data_list in data_groups.values() for val in data_list]
        global_min, global_max = min(all_data), max(all_data)
        
        # 2-1. 히스토그램 오버레이 (Left)
        ax1 = plt.subplot(3, 4, 1)
        if 'Left_playing' in data_groups:
            ax1.hist(data_groups['Left_playing'], bins=50, alpha=0.7, 
                    color=colors['playing'], label='Playing', density=True, range=(global_min, global_max))
        if 'Left_quiet' in data_groups:
            ax1.hist(data_groups['Left_quiet'], bins=50, alpha=0.7, 
                    color=colors['quiet'], label='Quiet', density=True, range=(global_min, global_max))
        ax1.set_title('Left Hand - Histogram Overlay', fontweight='bold')
        ax1.set_xlabel('Depth Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2-2. 히스토그램 오버레이 (Right)
        ax2 = plt.subplot(3, 4, 2)
        if 'Right_playing' in data_groups:
            ax2.hist(data_groups['Right_playing'], bins=50, alpha=0.7, 
                    color=colors['playing'], label='Playing', density=True, range=(global_min, global_max))
        if 'Right_quiet' in data_groups:
            ax2.hist(data_groups['Right_quiet'], bins=50, alpha=0.7, 
                    color=colors['quiet'], label='Quiet', density=True, range=(global_min, global_max))
        ax2.set_title('Right Hand - Histogram Overlay', fontweight='bold')
        ax2.set_xlabel('Depth Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 2-3. 박스플롯 종합
        ax3 = plt.subplot(3, 4, 3)
        box_data = []
        box_labels = []
        box_colors = []
        
        for group_name in ['Left_playing', 'Left_quiet', 'Right_playing', 'Right_quiet']:
            if group_name in data_groups:
                box_data.append(data_groups[group_name])
                box_labels.append(group_name.replace('_', '\n'))
                if 'playing' in group_name:
                    box_colors.append(colors['playing'])
                else:
                    box_colors.append(colors['quiet'])
        
        if box_data:
            bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(box_colors[i])
                patch.set_alpha(0.7)
        ax3.set_title('Box Plot - All Groups', fontweight='bold')
        ax3.set_ylabel('Depth Value')
        ax3.set_ylim(global_min, global_max)
        ax3.grid(True, alpha=0.3)
        
        # 2-4. KDE 비교 (Left)
        ax4 = plt.subplot(3, 4, 4)
        if 'Left_playing' in data_groups:
            left_playing = np.array(data_groups['Left_playing'])
            x_range, density = self._calculate_kde(left_playing)
            if len(x_range) > 0:
                ax4.plot(x_range, density, color=colors['playing'], linewidth=2, label='Playing')
        if 'Left_quiet' in data_groups:
            left_quiet = np.array(data_groups['Left_quiet'])
            x_range, density = self._calculate_kde(left_quiet)
            if len(x_range) > 0:
                ax4.plot(x_range, density, color=colors['quiet'], linewidth=2, label='Quiet')
        ax4.set_title('Left Hand - KDE Comparison', fontweight='bold')
        ax4.set_xlabel('Depth Value')
        ax4.set_ylabel('Density')
        ax4.set_xlim(global_min, global_max)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 2-5. KDE 비교 (Right)
        ax5 = plt.subplot(3, 4, 5)
        if 'Right_playing' in data_groups:
            right_playing = np.array(data_groups['Right_playing'])
            x_range, density = self._calculate_kde(right_playing)
            if len(x_range) > 0:
                ax5.plot(x_range, density, color=colors['playing'], linewidth=2, label='Playing')
        if 'Right_quiet' in data_groups:
            right_quiet = np.array(data_groups['Right_quiet'])
            x_range, density = self._calculate_kde(right_quiet)
            if len(x_range) > 0:
                ax5.plot(x_range, density, color=colors['quiet'], linewidth=2, label='Quiet')
        ax5.set_title('Right Hand - KDE Comparison', fontweight='bold')
        ax5.set_xlabel('Depth Value')
        ax5.set_ylabel('Density')
        ax5.set_xlim(global_min, global_max)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 2-6. Q-Q Plot (Left Playing)
        ax6 = plt.subplot(3, 4, 6)
        if 'Left_playing' in data_groups:
            stats.probplot(data_groups['Left_playing'], dist="norm", plot=ax6)
            ax6.set_title('Left Playing - Q-Q Plot', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 2-7. Q-Q Plot (Left Quiet)
        ax7 = plt.subplot(3, 4, 7)
        if 'Left_quiet' in data_groups:
            stats.probplot(data_groups['Left_quiet'], dist="norm", plot=ax7)
            ax7.set_title('Left Quiet - Q-Q Plot', fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # 2-8. 누적분포함수 (Left)
        ax8 = plt.subplot(3, 4, 8)
        if 'Left_playing' in data_groups:
            left_playing_sorted = np.sort(data_groups['Left_playing'])
            left_playing_cdf = np.arange(1, len(left_playing_sorted) + 1) / len(left_playing_sorted)
            ax8.plot(left_playing_sorted, left_playing_cdf, color=colors['playing'], 
                    linewidth=2, label='Playing')
        if 'Left_quiet' in data_groups:
            left_quiet_sorted = np.sort(data_groups['Left_quiet'])
            left_quiet_cdf = np.arange(1, len(left_quiet_sorted) + 1) / len(left_quiet_sorted)
            ax8.plot(left_quiet_sorted, left_quiet_cdf, color=colors['quiet'], 
                    linewidth=2, label='Quiet')
        ax8.set_title('Left Hand - CDF Comparison', fontweight='bold')
        ax8.set_xlabel('Depth Value')
        ax8.set_ylabel('Cumulative Probability')
        ax8.set_xlim(global_min, global_max)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 2-9. 누적분포함수 (Right)
        ax9 = plt.subplot(3, 4, 9)
        if 'Right_playing' in data_groups:
            right_playing_sorted = np.sort(data_groups['Right_playing'])
            right_playing_cdf = np.arange(1, len(right_playing_sorted) + 1) / len(right_playing_sorted)
            ax9.plot(right_playing_sorted, right_playing_cdf, color=colors['playing'], 
                     linewidth=2, label='Playing')
        if 'Right_quiet' in data_groups:
            right_quiet_sorted = np.sort(data_groups['Right_quiet'])
            right_quiet_cdf = np.arange(1, len(right_quiet_sorted) + 1) / len(right_quiet_sorted)
            ax9.plot(right_quiet_sorted, right_quiet_cdf, color=colors['quiet'], 
                     linewidth=2, label='Quiet')
        ax9.set_title('Right Hand - CDF Comparison', fontweight='bold')
        ax9.set_xlabel('Depth Value')
        ax9.set_ylabel('Cumulative Probability')
        ax9.set_xlim(global_min, global_max)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 2-10. 바이올린 플롯 (종합)
        ax10 = plt.subplot(3, 4, 10)
        violin_data = []
        violin_labels = []
        violin_colors = []
        for group, data in data_groups.items():
            if data:
                violin_data.append(data)
                violin_labels.append(group.replace('_', '\n'))
                if 'playing' in group:
                    violin_colors.append(colors['playing'])
                else:
                    violin_colors.append(colors['quiet'])
        
        if violin_data:
            parts = ax10.violinplot(violin_data, showmeans=True, showmedians=True)
            for i, part in enumerate(parts['bodies']):
                part.set_facecolor(violin_colors[i])
                part.set_alpha(0.7)
        ax10.set_title('Violin Plot - All Groups', fontweight='bold')
        ax10.set_xticks(range(1, len(violin_labels) + 1))
        ax10.set_xticklabels(violin_labels)
        ax10.set_ylabel('Depth Value')
        ax10.set_ylim(global_min, global_max)
        ax10.grid(True, alpha=0.3)
        
        # 2-11. 정규성 검정 결과 요약
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        # 정규성 검정 결과 텍스트
        text_content = "Normality Test Results\n" + "="*25 + "\n\n"
        for result in normality_results:
            if 'error' not in result:
                group = result['group'].replace('_', ' ').title()
                text_content += f"{group}:\n"
                if 'shapiro_p' in result and result['shapiro_p'] is not None:
                    normal_status = "Normal" if result['shapiro_normal'] else "Not Normal"
                    text_content += f"  Shapiro: p={result['shapiro_p']:.4f}\n  Status: {normal_status}\n"
                if 'anderson_normal' in result and result['anderson_normal'] is not None:
                    normal_status = "Normal" if result['anderson_normal'] else "Not Normal"
                    text_content += f"  Anderson: {normal_status}\n"
                text_content += "\n"
        
        ax11.text(0.05, 0.95, text_content, transform=ax11.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 2-12. 통계 요약
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        summary_text = "Statistical Summary\n" + "="*20 + "\n\n"
        for result in stats_results:
            group = result['group'].replace('_', ' ').title()
            summary_text += f"{group}:\n"
            summary_text += f"  Mean: {result['mean']:.4f}\n"
            summary_text += f"  Std: {result['std']:.4f}\n"
            summary_text += f"  Skew: {result['skewness']:.4f}\n"
            summary_text += f"  Kurt: {result['kurtosis']:.4f}\n\n"
        
        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = self.output_dir / "02_overall_distribution_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 전체 종합 분포 분석 저장: {filename}")
        plt.close()
        
        # 3. 정규성 검정 결과 요약 출력
        print(f"\n🔬 정규성 검정 결과:")
        for result in normality_results:
            if 'error' not in result:
                group = result['group'].replace('_', ' ').title()
                print(f"\n   {group}:")
                if 'shapiro_p' in result and result['shapiro_p'] is not None:
                    status = "정규분포" if result['shapiro_normal'] else "비정규분포"
                    print(f"     Shapiro-Wilk: p={result['shapiro_p']:.6f} → {status}")
                if 'anderson_normal' in result and result['anderson_normal'] is not None:
                    status = "정규분포" if result['anderson_normal'] else "비정규분포"
                    print(f"     Anderson-Darling: → {status}")
        
        # 4. 통계 결과를 텍스트 파일로 저장
        self._save_statistics_report(stats_results, normality_results, data_groups, dataset_name)
        
        # 5. 각 플롯별 상세 분석 리포트 저장
        self._save_plot_analysis_report(stats_results, normality_results, data_groups, dataset_name)
        
        return {
            'stats_results': stats_results,
            'normality_results': normality_results,
            'data_groups': data_groups
        }
    
    def _save_statistics_report(self, stats_results, normality_results, data_groups, dataset_name):
        """통계 분석 결과를 텍스트 파일로 저장"""
        
        filename = self.output_dir / "04_statistics_report.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Playing vs Quiet 통계 분석 보고서\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # 1. 데이터 요약
            f.write("1. 데이터 요약\n")
            f.write("-" * 30 + "\n")
            for group, data in data_groups.items():
                f.write(f"{group.replace('_', ' ').title()}: {len(data):,}개 데이터\n")
            f.write(f"\n총 데이터: {sum(len(data) for data in data_groups.values()):,}개\n\n")
            
            # 2. 기본 통계량
            f.write("2. 기본 통계량\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Group':<15} {'Count':<10} {'Mean':<10} {'Std':<10} {'Median':<10} {'CV':<8} {'Skew':<8} {'Kurt':<8}\n")
            f.write("-" * 85 + "\n")
            
            for result in stats_results:
                group = result['group'].replace('_', ' ')
                f.write(f"{group:<15} {result['count']:<10} {result['mean']:<10.4f} {result['std']:<10.4f} "
                       f"{result['median']:<10.4f} {result['cv']:<8.4f} {result['skewness']:<8.4f} {result['kurtosis']:<8.4f}\n")
            
            f.write("\n")
            
            # 3. 상세 통계량
            f.write("3. 상세 통계량\n")
            f.write("-" * 30 + "\n")
            for result in stats_results:
                group = result['group'].replace('_', ' ').title()
                f.write(f"\n{group}:\n")
                f.write(f"  Count: {result['count']:,}\n")
                f.write(f"  Mean: {result['mean']:.6f}\n")
                f.write(f"  Median: {result['median']:.6f}\n")
                f.write(f"  Std Dev: {result['std']:.6f}\n")
                f.write(f"  Variance: {result['var']:.6f}\n")
                f.write(f"  Min: {result['min']:.6f}\n")
                f.write(f"  Max: {result['max']:.6f}\n")
                f.write(f"  Range: {result['range']:.6f}\n")
                f.write(f"  Q1: {result['q1']:.6f}\n")
                f.write(f"  Q3: {result['q3']:.6f}\n")
                f.write(f"  IQR: {result['iqr']:.6f}\n")
                f.write(f"  CV (변동계수): {result['cv']:.6f}\n")
                f.write(f"  Skewness (왜도): {result['skewness']:.6f}\n")
                f.write(f"  Kurtosis (첨도): {result['kurtosis']:.6f}\n")
                f.write(f"  SE Mean (평균의 표준오차): {result['se_mean']:.6f}\n")
            
            # 4. 정규성 검정
            f.write("\n\n4. 정규성 검정 결과\n")
            f.write("-" * 30 + "\n")
            for result in normality_results:
                if 'error' not in result:
                    group = result['group'].replace('_', ' ').title()
                    f.write(f"\n{group}:\n")
                    if 'shapiro_p' in result and result['shapiro_p'] is not None:
                        status = "정규분포" if result['shapiro_normal'] else "비정규분포"
                        f.write(f"  Shapiro-Wilk Test:\n")
                        f.write(f"    Statistic: {result['shapiro_stat']:.6f}\n")
                        f.write(f"    P-value: {result['shapiro_p']:.6f}\n")
                        f.write(f"    결과: {status} (α=0.05)\n")
                    if 'anderson_normal' in result and result['anderson_normal'] is not None:
                        status = "정규분포" if result['anderson_normal'] else "비정규분포"
                        f.write(f"  Anderson-Darling Test:\n")
                        f.write(f"    Statistic: {result['anderson_stat']:.6f}\n")
                        f.write(f"    Critical Value (5%): {result['anderson_critical']:.6f}\n")
                        f.write(f"    결과: {status} (α=0.05)\n")
            
            # 5. 그룹 간 비교
            f.write("\n\n5. 그룹 간 비교\n")
            f.write("-" * 30 + "\n")
            
            # Left: Playing vs Quiet
            left_playing = next((x for x in stats_results if x['group'] == 'Left_playing'), None)
            left_quiet = next((x for x in stats_results if x['group'] == 'Left_quiet'), None)
            
            if left_playing and left_quiet:
                f.write(f"\nLeft Hand (Playing vs Quiet):\n")
                f.write(f"  평균 차이: {left_quiet['mean'] - left_playing['mean']:+.6f}\n")
                f.write(f"  표준편차 차이: {left_quiet['std'] - left_playing['std']:+.6f}\n")
                f.write(f"  Playing 평균: {left_playing['mean']:.6f} ± {left_playing['std']:.6f}\n")
                f.write(f"  Quiet 평균: {left_quiet['mean']:.6f} ± {left_quiet['std']:.6f}\n")
            
            # Right: Playing vs Quiet
            right_playing = next((x for x in stats_results if x['group'] == 'Right_playing'), None)
            right_quiet = next((x for x in stats_results if x['group'] == 'Right_quiet'), None)
            
            if right_playing and right_quiet:
                f.write(f"\nRight Hand (Playing vs Quiet):\n")
                f.write(f"  평균 차이: {right_quiet['mean'] - right_playing['mean']:+.6f}\n")
                f.write(f"  표준편차 차이: {right_quiet['std'] - right_playing['std']:+.6f}\n")
                f.write(f"  Playing 평균: {right_playing['mean']:.6f} ± {right_playing['std']:.6f}\n")
                f.write(f"  Quiet 평균: {right_quiet['mean']:.6f} ± {right_quiet['std']:.6f}\n")
            
            # 6. 결론
            f.write("\n\n6. 결론\n")
            f.write("-" * 30 + "\n")
            f.write("이 분석은 MIDI Playing vs Quiet 상황에서 손의 깊이값 분포 차이를 통계적으로 검증합니다.\n")
            f.write("- 평균값의 차이가 클수록 MIDI 기반 floating 판정이 유효함을 의미합니다.\n")
            f.write("- 정규성 검정 결과는 추후 모수/비모수 검정 선택의 기준이 됩니다.\n")
            f.write("- 분포 형태 분석을 통해 각 그룹의 특성을 파악할 수 있습니다.\n")
        
        print(f"📋 통계 분석 보고서 저장: {filename}")
    
    def _save_plot_analysis_report(self, stats_results, normality_results, data_groups, dataset_name):
        """각 플롯별 상세 분석 리포트 저장"""
        
        filename = self.output_dir / "05_plot_analysis_report.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Playing vs Quiet 플롯별 상세 분석 리포트\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("📊 각 그래프 해석 가이드\n")
            f.write("=" * 40 + "\n\n")
            
            # 1. 히스토그램 오버레이 분석
            f.write("1. 히스토그램 오버레이 (Histogram Overlay)\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: Playing vs Quiet 상황의 분포 형태와 중첩 정도 확인\n\n")
            f.write("📈 해석 방법:\n")
            f.write("• 두 분포가 완전히 분리되어 있으면 → 구분이 매우 용이\n")
            f.write("• 분포가 많이 겹치면 → 구분이 어려움\n")
            f.write("• 피크(최빈값) 위치 차이 → 전형적인 값의 차이\n")
            f.write("• 분포의 폭 차이 → 변동성(일관성) 차이\n\n")
            
            # 실제 데이터 분석
            self._analyze_histogram_data(f, stats_results, data_groups)
            
            # 2. 박스플롯 분석
            f.write("\n2. 박스플롯 (Box Plot)\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: 중앙값, 사분위수, 이상치 비교\n\n")
            f.write("📈 해석 방법:\n")
            f.write("• 상자의 위치 → 중앙값(median) 비교\n")
            f.write("• 상자의 크기 → IQR(사분위범위), 변동성 비교\n")
            f.write("• 수염의 길이 → 데이터 범위, 극값 비교\n")
            f.write("• 점들(◦) → 이상치(outliers) 존재\n\n")
            
            self._analyze_boxplot_data(f, stats_results, data_groups)
            
            # 3. KDE 비교 분석
            f.write("\n3. KDE 비교 (Kernel Density Estimation)\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: 연속적인 확률밀도함수로 분포 형태 비교\n\n")
            f.write("📈 해석 방법:\n")
            f.write("• 곡선의 피크 → 가장 확률이 높은 값\n")
            f.write("• 곡선의 폭 → 분포의 산포도\n")
            f.write("• 곡선의 모양 → 분포의 대칭성, 치우침\n")
            f.write("• 다중 피크 → 다중 모달 분포 (여러 패턴 존재)\n\n")
            
            self._analyze_kde_data(f, stats_results, data_groups)
            
            # 4. Q-Q Plot 분석
            f.write("\n4. Q-Q Plot (Quantile-Quantile Plot)\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: 데이터가 정규분포를 따르는지 확인\n\n")
            f.write("📈 해석 방법:\n")
            f.write("• 점들이 직선에 가까우면 → 정규분포에 가까움\n")
            f.write("• S자 곡선 → 꼬리가 두꺼운 분포 (heavy-tailed)\n")
            f.write("• 역S자 곡선 → 꼬리가 얇은 분포 (light-tailed)\n")
            f.write("• 아래로 볼록 → 오른쪽 치우침 (right-skewed)\n")
            f.write("• 위로 볼록 → 왼쪽 치우침 (left-skewed)\n\n")
            
            self._analyze_qqplot_data(f, normality_results, stats_results)
            
            # 5. CDF 비교 분석
            f.write("\n5. CDF 비교 (Cumulative Distribution Function)\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: 누적확률로 분포 차이를 명확히 비교\n\n")
            f.write("📈 해석 방법:\n")
            f.write("• 곡선이 왼쪽에 있으면 → 더 작은 값들이 많음\n")
            f.write("• 곡선이 오른쪽에 있으면 → 더 큰 값들이 많음\n")
            f.write("• 곡선 간 최대 거리 → 분포 차이의 크기\n")
            f.write("• 가파른 상승 → 특정 구간에 데이터 집중\n\n")
            
            self._analyze_cdf_data(f, stats_results, data_groups)
            
            # 6. 바이올린 플롯 분석
            f.write("\n6. 바이올린 플롯 (Violin Plot)\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: 박스플롯 + KDE의 결합, 완전한 분포 정보\n\n")
            f.write("📈 해석 방법:\n")
            f.write("• 바이올린의 폭 → 해당 값의 데이터 밀도\n")
            f.write("• 바이올린의 모양 → 분포의 형태 (대칭, 치우침, 다중모달)\n")
            f.write("• 중앙의 박스 → 중앙값과 사분위수\n")
            f.write("• 전체적인 크기 → 데이터의 범위\n\n")
            
            self._analyze_violin_data(f, stats_results, data_groups)
            
            # 7. 통계 요약 해석
            f.write("\n7. 통계 요약 해석\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: 수치적 지표로 분포 특성 정량화\n\n")
            
            self._analyze_statistics_summary(f, stats_results, data_groups)
            
            # 8. 정규성 검정 해석
            f.write("\n8. 정규성 검정 해석\n")
            f.write("-" * 50 + "\n")
            f.write("🎯 목적: 통계적 가설검정 방법 선택의 기준\n\n")
            f.write("📈 Shapiro-Wilk Test 해석:\n")
            f.write("• H0 (귀무가설): 데이터가 정규분포를 따른다\n")
            f.write("• H1 (대립가설): 데이터가 정규분포를 따르지 않는다\n")
            f.write("• p-value > 0.05 → 정규분포 가정 (모수검정 가능)\n")
            f.write("• p-value ≤ 0.05 → 비정규분포 (비모수검정 권장)\n\n")
            
            self._analyze_normality_results(f, normality_results)
            
            # 9. 종합 해석 및 권장사항
            f.write("\n9. 종합 해석 및 권장사항\n")
            f.write("-" * 50 + "\n")
            
            self._provide_overall_interpretation(f, stats_results, normality_results, data_groups)
        
        print(f"📊 플롯별 분석 리포트 저장: {filename}")
    
    def _analyze_histogram_data(self, f, stats_results, data_groups):
        """히스토그램 데이터 분석"""
        f.write("🔍 현재 데이터 분석:\n")
        for hand in ['Left', 'Right']:
            playing_key = f'{hand}_playing'
            quiet_key = f'{hand}_quiet'
            
            if playing_key in data_groups and quiet_key in data_groups:
                playing_stats = next((x for x in stats_results if x['group'] == playing_key), None)
                quiet_stats = next((x for x in stats_results if x['group'] == quiet_key), None)
                
                if playing_stats and quiet_stats:
                    mean_diff = quiet_stats['mean'] - playing_stats['mean']
                    overlap_estimate = min(playing_stats['mean'] + 2*playing_stats['std'], 
                                         quiet_stats['mean'] + 2*quiet_stats['std']) - \
                                     max(playing_stats['mean'] - 2*playing_stats['std'], 
                                         quiet_stats['mean'] - 2*quiet_stats['std'])
                    total_range = max(playing_stats['mean'] + 2*playing_stats['std'], 
                                    quiet_stats['mean'] + 2*quiet_stats['std']) - \
                                min(playing_stats['mean'] - 2*playing_stats['std'], 
                                    quiet_stats['mean'] - 2*quiet_stats['std'])
                    
                    f.write(f"\n{hand} Hand:\n")
                    f.write(f"  평균 차이: {mean_diff:+.4f}\n")
                    if mean_diff > 0:
                        f.write(f"  → Quiet이 더 깊음 (floating 경향)\n")
                    else:
                        f.write(f"  → Playing이 더 깊음 (예상과 반대)\n")
                    
                    if total_range > 0:
                        overlap_ratio = overlap_estimate / total_range
                        f.write(f"  예상 겹침 정도: {overlap_ratio:.1%}\n")
                        if overlap_ratio < 0.3:
                            f.write(f"  → 분포가 잘 분리됨 (구분 용이)\n")
                        elif overlap_ratio < 0.7:
                            f.write(f"  → 중간 정도 겹침 (구분 가능)\n")
                        else:
                            f.write(f"  → 많이 겹침 (구분 어려움)\n")
    
    def _analyze_boxplot_data(self, f, stats_results, data_groups):
        """박스플롯 데이터 분석"""
        f.write("🔍 현재 데이터 분석:\n")
        for hand in ['Left', 'Right']:
            playing_key = f'{hand}_playing'
            quiet_key = f'{hand}_quiet'
            
            if playing_key in data_groups and quiet_key in data_groups:
                playing_stats = next((x for x in stats_results if x['group'] == playing_key), None)
                quiet_stats = next((x for x in stats_results if x['group'] == quiet_key), None)
                
                if playing_stats and quiet_stats:
                    f.write(f"\n{hand} Hand:\n")
                    f.write(f"  중앙값 비교: Playing {playing_stats['median']:.4f} vs Quiet {quiet_stats['median']:.4f}\n")
                    f.write(f"  IQR 비교: Playing {playing_stats['iqr']:.4f} vs Quiet {quiet_stats['iqr']:.4f}\n")
                    
                    if quiet_stats['median'] > playing_stats['median']:
                        f.write(f"  → Quiet의 중앙값이 더 큼 (floating 경향)\n")
                    else:
                        f.write(f"  → Playing의 중앙값이 더 큼\n")
                    
                    if quiet_stats['iqr'] > playing_stats['iqr']:
                        f.write(f"  → Quiet이 더 변동적\n")
                    else:
                        f.write(f"  → Playing이 더 변동적\n")
    
    def _analyze_kde_data(self, f, stats_results, data_groups):
        """KDE 데이터 분석"""
        f.write("🔍 현재 데이터 분석:\n")
        for hand in ['Left', 'Right']:
            playing_key = f'{hand}_playing'
            quiet_key = f'{hand}_quiet'
            
            if playing_key in data_groups and quiet_key in data_groups:
                playing_stats = next((x for x in stats_results if x['group'] == playing_key), None)
                quiet_stats = next((x for x in stats_results if x['group'] == quiet_key), None)
                
                if playing_stats and quiet_stats:
                    f.write(f"\n{hand} Hand:\n")
                    f.write(f"  분포 형태 - Playing: 왜도 {playing_stats['skewness']:.3f}, 첨도 {playing_stats['kurtosis']:.3f}\n")
                    f.write(f"  분포 형태 - Quiet: 왜도 {quiet_stats['skewness']:.3f}, 첨도 {quiet_stats['kurtosis']:.3f}\n")
                    
                    # 왜도 해석
                    for group_name, stats in [('Playing', playing_stats), ('Quiet', quiet_stats)]:
                        if abs(stats['skewness']) < 0.5:
                            skew_desc = "대칭적"
                        elif stats['skewness'] > 0.5:
                            skew_desc = "오른쪽 치우침"
                        else:
                            skew_desc = "왼쪽 치우침"
                        
                        if abs(stats['kurtosis']) < 0.5:
                            kurt_desc = "정규분포와 유사한 꼬리"
                        elif stats['kurtosis'] > 0.5:
                            kurt_desc = "두꺼운 꼬리"
                        else:
                            kurt_desc = "얇은 꼬리"
                        
                        f.write(f"  → {group_name}: {skew_desc}, {kurt_desc}\n")
    
    def _analyze_qqplot_data(self, f, normality_results, stats_results):
        """Q-Q 플롯 데이터 분석"""
        f.write("🔍 현재 데이터 분석:\n")
        for result in normality_results:
            if 'error' not in result:
                group = result['group'].replace('_', ' ').title()
                f.write(f"\n{group}:\n")
                
                # 왜도/첨도를 이용한 Q-Q plot 예측
                stats_data = next((x for x in stats_results if x['group'] == result['group']), None)
                if stats_data:
                    skewness = stats_data['skewness']
                    kurtosis = stats_data['kurtosis']
                    
                    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                        f.write(f"  예상 Q-Q plot: 직선에 가까움 (정규분포 유사)\n")
                    elif skewness > 0.5:
                        f.write(f"  예상 Q-Q plot: 아래로 볼록한 곡선 (오른쪽 치우침)\n")
                    elif skewness < -0.5:
                        f.write(f"  예상 Q-Q plot: 위로 볼록한 곡선 (왼쪽 치우침)\n")
                    
                    if kurtosis > 0.5:
                        f.write(f"  예상 꼬리 부분: S자 형태 (두꺼운 꼬리)\n")
                    elif kurtosis < -0.5:
                        f.write(f"  예상 꼬리 부분: 역S자 형태 (얇은 꼬리)\n")
    
    def _analyze_cdf_data(self, f, stats_results, data_groups):
        """CDF 데이터 분석"""
        f.write("🔍 현재 데이터 분석:\n")
        for hand in ['Left', 'Right']:
            playing_key = f'{hand}_playing'
            quiet_key = f'{hand}_quiet'
            
            if playing_key in data_groups and quiet_key in data_groups:
                playing_stats = next((x for x in stats_results if x['group'] == playing_key), None)
                quiet_stats = next((x for x in stats_results if x['group'] == quiet_key), None)
                
                if playing_stats and quiet_stats:
                    f.write(f"\n{hand} Hand:\n")
                    mean_diff = quiet_stats['mean'] - playing_stats['mean']
                    f.write(f"  평균 차이: {mean_diff:+.4f}\n")
                    
                    if mean_diff > 0:
                        f.write(f"  → Quiet의 CDF가 더 오른쪽 (더 큰 값들)\n")
                        f.write(f"  → Quiet hands가 더 floating한 경향\n")
                    else:
                        f.write(f"  → Playing의 CDF가 더 오른쪽\n")
                    
                    # 분산 차이를 이용한 CDF 기울기 예측
                    var_ratio = quiet_stats['var'] / playing_stats['var'] if playing_stats['var'] > 0 else 1
                    if var_ratio > 1.2:
                        f.write(f"  → Quiet의 CDF가 더 완만한 기울기 (더 분산됨)\n")
                    elif var_ratio < 0.8:
                        f.write(f"  → Quiet의 CDF가 더 가파른 기울기 (더 집중됨)\n")
                    else:
                        f.write(f"  → 유사한 분산도\n")
    
    def _analyze_violin_data(self, f, stats_results, data_groups):
        """바이올린 플롯 데이터 분석"""
        f.write("🔍 현재 데이터 분석:\n")
        for hand in ['Left', 'Right']:
            playing_key = f'{hand}_playing'
            quiet_key = f'{hand}_quiet'
            
            if playing_key in data_groups and quiet_key in data_groups:
                playing_stats = next((x for x in stats_results if x['group'] == playing_key), None)
                quiet_stats = next((x for x in stats_results if x['group'] == quiet_key), None)
                
                if playing_stats and quiet_stats:
                    f.write(f"\n{hand} Hand:\n")
                    f.write(f"  데이터 크기: Playing {playing_stats['count']:,} vs Quiet {quiet_stats['count']:,}\n")
                    f.write(f"  범위: Playing {playing_stats['range']:.4f} vs Quiet {quiet_stats['range']:.4f}\n")
                    
                    # 분포 형태 예측
                    playing_shape = "대칭적" if abs(playing_stats['skewness']) < 0.5 else \
                                  ("오른쪽 치우침" if playing_stats['skewness'] > 0 else "왼쪽 치우침")
                    quiet_shape = "대칭적" if abs(quiet_stats['skewness']) < 0.5 else \
                                ("오른쪽 치우침" if quiet_stats['skewness'] > 0 else "왼쪽 치우침")
                    
                    f.write(f"  예상 바이올린 모양 - Playing: {playing_shape}\n")
                    f.write(f"  예상 바이올린 모양 - Quiet: {quiet_shape}\n")
    
    def _analyze_statistics_summary(self, f, stats_results, data_groups):
        """통계 요약 분석"""
        f.write("📈 주요 통계 지표 해석:\n\n")
        f.write("• 평균 (Mean): 전체적인 중심 경향\n")
        f.write("• 표준편차 (Std): 변동성의 크기\n") 
        f.write("• 변동계수 (CV): 상대적 변동성 (CV = Std/Mean)\n")
        f.write("• 왜도 (Skewness): 분포의 비대칭성\n")
        f.write("  - 0에 가까우면 대칭적\n")
        f.write("  - 양수면 오른쪽 치우침 (긴 오른쪽 꼬리)\n")
        f.write("  - 음수면 왼쪽 치우침 (긴 왼쪽 꼬리)\n")
        f.write("• 첨도 (Kurtosis): 분포의 뾰족함\n")
        f.write("  - 0에 가까우면 정규분포와 유사\n")
        f.write("  - 양수면 뾰족한 분포 (두꺼운 꼬리)\n")
        f.write("  - 음수면 평평한 분포 (얇은 꼬리)\n\n")
        
        f.write("🔍 현재 데이터 분석:\n")
        for hand in ['Left', 'Right']:
            playing_key = f'{hand}_playing'
            quiet_key = f'{hand}_quiet'
            
            if playing_key in data_groups and quiet_key in data_groups:
                playing_stats = next((x for x in stats_results if x['group'] == playing_key), None)
                quiet_stats = next((x for x in stats_results if x['group'] == quiet_key), None)
                
                if playing_stats and quiet_stats:
                    f.write(f"\n{hand} Hand 비교:\n")
                    
                    # 평균 차이
                    mean_diff = quiet_stats['mean'] - playing_stats['mean']
                    mean_diff_pct = (mean_diff / playing_stats['mean'] * 100) if playing_stats['mean'] != 0 else 0
                    f.write(f"  평균 차이: {mean_diff:+.4f} ({mean_diff_pct:+.1f}%)\n")
                    
                    # 변동성 비교
                    cv_playing = playing_stats['cv']
                    cv_quiet = quiet_stats['cv']
                    f.write(f"  변동계수: Playing {cv_playing:.3f} vs Quiet {cv_quiet:.3f}\n")
                    
                    if cv_quiet > cv_playing * 1.1:
                        f.write(f"  → Quiet이 더 변동적 (일관성 낮음)\n")
                    elif cv_quiet < cv_playing * 0.9:
                        f.write(f"  → Playing이 더 변동적 (일관성 낮음)\n")
                    else:
                        f.write(f"  → 유사한 변동성\n")
                    
                    # 실용적 의미
                    if abs(mean_diff) > playing_stats['std'] * 0.5:
                        f.write(f"  ⭐ 실용적으로 의미 있는 차이 (효과크기 중간 이상)\n")
                    elif abs(mean_diff) > playing_stats['std'] * 0.2:
                        f.write(f"  ⭐ 실용적으로 작은 차이 (효과크기 작음)\n")
                    else:
                        f.write(f"  ⭐ 실용적으로 무시할 수 있는 차이\n")
    
    def _analyze_normality_results(self, f, normality_results):
        """정규성 검정 결과 분석"""
        f.write("🔍 현재 데이터 분석:\n")
        
        normal_count = 0
        total_count = 0
        
        for result in normality_results:
            if 'error' not in result:
                total_count += 1
                group = result['group'].replace('_', ' ').title()
                f.write(f"\n{group}:\n")
                
                if 'shapiro_p' in result and result['shapiro_p'] is not None:
                    p_val = result['shapiro_p']
                    is_normal = result['shapiro_normal']
                    
                    if is_normal:
                        normal_count += 1
                        f.write(f"  p-value = {p_val:.6f} > 0.05 → 정규분포 가정 가능\n")
                        f.write(f"  → 모수검정 (t-test, ANOVA) 사용 권장\n")
                    else:
                        f.write(f"  p-value = {p_val:.6f} ≤ 0.05 → 비정규분포\n")
                        f.write(f"  → 비모수검정 (Mann-Whitney U, Kruskal-Wallis) 권장\n")
                        
                        if p_val < 0.001:
                            f.write(f"  → 매우 강한 비정규성 (p < 0.001)\n")
                        elif p_val < 0.01:
                            f.write(f"  → 강한 비정규성 (p < 0.01)\n")
                        else:
                            f.write(f"  → 약한 비정규성 (0.01 ≤ p ≤ 0.05)\n")
        
        if total_count > 0:
            normal_ratio = normal_count / total_count
            f.write(f"\n📊 전체 요약:\n")
            f.write(f"  정규분포 그룹: {normal_count}/{total_count} ({normal_ratio:.1%})\n")
            
            if normal_ratio >= 0.75:
                f.write(f"  → 대부분 정규분포, 모수검정 사용 가능\n")
            elif normal_ratio >= 0.5:
                f.write(f"  → 혼재된 상황, 비모수검정 고려\n")
            else:
                f.write(f"  → 대부분 비정규분포, 비모수검정 권장\n")
    
    def _provide_overall_interpretation(self, f, stats_results, normality_results, data_groups):
        """종합 해석 및 권장사항"""
        f.write("🎯 MIDI 기반 Floating Hand 판정 유효성 검증 결과\n\n")
        
        # 각 손별 효과 크기 계산
        significant_differences = []
        
        for hand in ['Left', 'Right']:
            playing_key = f'{hand}_playing'
            quiet_key = f'{hand}_quiet'
            
            if playing_key in data_groups and quiet_key in data_groups:
                playing_stats = next((x for x in stats_results if x['group'] == playing_key), None)
                quiet_stats = next((x for x in stats_results if x['group'] == quiet_key), None)
                
                if playing_stats and quiet_stats:
                    # Cohen's d 계산 (효과 크기)
                    pooled_std = np.sqrt((playing_stats['var'] + quiet_stats['var']) / 2)
                    cohens_d = (quiet_stats['mean'] - playing_stats['mean']) / pooled_std if pooled_std > 0 else 0
                    
                    f.write(f"{hand} Hand 결과:\n")
                    f.write(f"  평균 차이: {quiet_stats['mean'] - playing_stats['mean']:+.4f}\n")
                    f.write(f"  Cohen's d: {cohens_d:.3f}\n")
                    
                    if abs(cohens_d) >= 0.8:
                        effect_size = "큰 효과"
                        significant_differences.append(hand)
                    elif abs(cohens_d) >= 0.5:
                        effect_size = "중간 효과"
                        significant_differences.append(hand)
                    elif abs(cohens_d) >= 0.2:
                        effect_size = "작은 효과"
                    else:
                        effect_size = "무시할 수 있는 효과"
                    
                    f.write(f"  효과 크기: {effect_size}\n")
                    
                    if cohens_d > 0:
                        f.write(f"  → Quiet 상황에서 손이 더 깊음 (floating 경향) ✅\n")
                    else:
                        f.write(f"  → Playing 상황에서 손이 더 깊음 (예상과 반대) ⚠️\n")
                    f.write("\n")
        
        # 종합 결론
        f.write("📋 종합 결론:\n")
        f.write("-" * 30 + "\n")
        
        if len(significant_differences) >= 2:
            f.write("✅ 양손 모두에서 유의미한 차이 발견\n")
            f.write("✅ MIDI 기반 floating hand 판정이 통계적으로 유효함\n")
            recommendation = "MIDI 정보를 활용한 floating hand 감지 시스템 구현 권장"
        elif len(significant_differences) == 1:
            f.write(f"⚠️ {significant_differences[0]} 손에서만 유의미한 차이 발견\n")
            f.write(f"⚠️ 손별로 다른 임계값 또는 알고리즘 필요\n")
            recommendation = "손별 차별화된 floating hand 감지 알고리즘 개발 권장"
        else:
            f.write("❌ 양손 모두에서 유의미한 차이 없음\n")
            f.write("❌ MIDI 기반 floating hand 판정의 효과 제한적\n")
            recommendation = "다른 특성(속도, 가속도, 궤적 등) 탐색 권장"
        
        f.write(f"\n🎯 권장사항: {recommendation}\n")
        
        # 추가 분석 제안
        f.write(f"\n📌 추가 분석 제안:\n")
        f.write(f"1. 통계적 가설검정 수행 (t-test 또는 Mann-Whitney U test)\n")
        f.write(f"2. ROC 곡선 분석으로 최적 임계값 탐색\n")
        f.write(f"3. 시간대별/곡별 세부 분석\n")
        f.write(f"4. 머신러닝 분류기 성능 평가\n")
        f.write(f"5. 실시간 적용 시 성능 검증\n")
    
    def _create_hand_specific_analysis(self, data_groups: Dict, dataset_name: str, hand_type: str):
        """특정 손(Left/Right)에 대한 상세 분포 분석"""
        
        # 해당 손의 데이터만 추출
        hand_playing_key = f'{hand_type}_playing'
        hand_quiet_key = f'{hand_type}_quiet'
        
        if hand_playing_key not in data_groups and hand_quiet_key not in data_groups:
            print(f"⚠️ {hand_type} 손 데이터가 없어 분석을 건너뜁니다.")
            return
        
        playing_data = data_groups.get(hand_playing_key, [])
        quiet_data = data_groups.get(hand_quiet_key, [])
        
        print(f"\n🔍 {hand_type} Hand 상세 분석 중...")
        print(f"   Playing: {len(playing_data):,}개, Quiet: {len(quiet_data):,}개")
        
        # 전체 데이터 범위 계산 (이 손의 데이터만)
        all_hand_data = playing_data + quiet_data
        if not all_hand_data:
            print(f"⚠️ {hand_type} 손의 유효한 데이터가 없습니다.")
            return
            
        global_min, global_max = min(all_hand_data), max(all_hand_data)
        
        # 색상 설정
        colors = {'playing': '#FF6B6B', 'quiet': '#4ECDC4'}
        
        # 3x3 레이아웃으로 구성
        fig = plt.figure(figsize=(18, 15))
        fig.suptitle(f'{hand_type} Hand - Detailed Distribution Analysis\n{dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 히스토그램 오버레이
        ax1 = plt.subplot(3, 3, 1)
        if playing_data:
            ax1.hist(playing_data, bins=50, alpha=0.7, color=colors['playing'], 
                    label=f'Playing (n={len(playing_data):,})', density=True, range=(global_min, global_max))
        if quiet_data:
            ax1.hist(quiet_data, bins=50, alpha=0.7, color=colors['quiet'], 
                    label=f'Quiet (n={len(quiet_data):,})', density=True, range=(global_min, global_max))
        ax1.set_title(f'{hand_type} Hand - Histogram Overlay', fontweight='bold')
        ax1.set_xlabel('Depth Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 박스플롯
        ax2 = plt.subplot(3, 3, 2)
        box_data = []
        box_labels = []
        box_colors = []
        if playing_data:
            box_data.append(playing_data)
            box_labels.append('Playing')
            box_colors.append(colors['playing'])
        if quiet_data:
            box_data.append(quiet_data)
            box_labels.append('Quiet')
            box_colors.append(colors['quiet'])
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(box_colors[i])
                patch.set_alpha(0.7)
        ax2.set_title(f'{hand_type} Hand - Box Plot', fontweight='bold')
        ax2.set_ylabel('Depth Value')
        ax2.set_ylim(global_min, global_max)
        ax2.grid(True, alpha=0.3)
        
        # 3. KDE 비교
        ax3 = plt.subplot(3, 3, 3)
        if playing_data:
            playing_arr = np.array(playing_data)
            x_range, density = self._calculate_kde(playing_arr)
            if len(x_range) > 0:
                ax3.plot(x_range, density, color=colors['playing'], linewidth=2, label='Playing')
        if quiet_data:
            quiet_arr = np.array(quiet_data)
            x_range, density = self._calculate_kde(quiet_arr)
            if len(x_range) > 0:
                ax3.plot(x_range, density, color=colors['quiet'], linewidth=2, label='Quiet')
        ax3.set_title(f'{hand_type} Hand - KDE Comparison', fontweight='bold')
        ax3.set_xlabel('Depth Value')
        ax3.set_ylabel('Density')
        ax3.set_xlim(global_min, global_max)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q Plot (Playing)
        ax4 = plt.subplot(3, 3, 4)
        if playing_data:
            stats.probplot(playing_data, dist="norm", plot=ax4)
            ax4.set_title(f'{hand_type} Playing - Q-Q Plot', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. Q-Q Plot (Quiet)
        ax5 = plt.subplot(3, 3, 5)
        if quiet_data:
            stats.probplot(quiet_data, dist="norm", plot=ax5)
            ax5.set_title(f'{hand_type} Quiet - Q-Q Plot', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. CDF 비교
        ax6 = plt.subplot(3, 3, 6)
        if playing_data:
            playing_sorted = np.sort(playing_data)
            playing_cdf = np.arange(1, len(playing_sorted) + 1) / len(playing_sorted)
            ax6.plot(playing_sorted, playing_cdf, color=colors['playing'], linewidth=2, label='Playing')
        if quiet_data:
            quiet_sorted = np.sort(quiet_data)
            quiet_cdf = np.arange(1, len(quiet_sorted) + 1) / len(quiet_sorted)
            ax6.plot(quiet_sorted, quiet_cdf, color=colors['quiet'], linewidth=2, label='Quiet')
        ax6.set_title(f'{hand_type} Hand - CDF Comparison', fontweight='bold')
        ax6.set_xlabel('Depth Value')
        ax6.set_ylabel('Cumulative Probability')
        ax6.set_xlim(global_min, global_max)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 바이올린 플롯
        ax7 = plt.subplot(3, 3, 7)
        if box_data:
            parts = ax7.violinplot(box_data, showmeans=True, showmedians=True)
            for i, part in enumerate(parts['bodies']):
                part.set_facecolor(box_colors[i])
                part.set_alpha(0.7)
        ax7.set_title(f'{hand_type} Hand - Violin Plot', fontweight='bold')
        ax7.set_xticks(range(1, len(box_labels) + 1))
        ax7.set_xticklabels(box_labels)
        ax7.set_ylabel('Depth Value')
        ax7.set_ylim(global_min, global_max)
        ax7.grid(True, alpha=0.3)
        
        # 8. 통계 요약
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        summary_text = f"{hand_type} Hand Statistics\n" + "="*25 + "\n\n"
        
        if playing_data:
            playing_stats = self._calculate_detailed_statistics(playing_data, f'{hand_type}_playing')
            summary_text += f"Playing (n={len(playing_data):,}):\n"
            summary_text += f"  Mean: {playing_stats['mean']:.4f}\n"
            summary_text += f"  Std: {playing_stats['std']:.4f}\n"
            summary_text += f"  Median: {playing_stats['median']:.4f}\n"
            summary_text += f"  Skew: {playing_stats['skewness']:.4f}\n"
            summary_text += f"  Kurt: {playing_stats['kurtosis']:.4f}\n\n"
        
        if quiet_data:
            quiet_stats = self._calculate_detailed_statistics(quiet_data, f'{hand_type}_quiet')
            summary_text += f"Quiet (n={len(quiet_data):,}):\n"
            summary_text += f"  Mean: {quiet_stats['mean']:.4f}\n"
            summary_text += f"  Std: {quiet_stats['std']:.4f}\n"
            summary_text += f"  Median: {quiet_stats['median']:.4f}\n"
            summary_text += f"  Skew: {quiet_stats['skewness']:.4f}\n"
            summary_text += f"  Kurt: {quiet_stats['kurtosis']:.4f}\n\n"
        
        if playing_data and quiet_data:
            playing_stats = self._calculate_detailed_statistics(playing_data, f'{hand_type}_playing')
            quiet_stats = self._calculate_detailed_statistics(quiet_data, f'{hand_type}_quiet')
            mean_diff = quiet_stats['mean'] - playing_stats['mean']
            summary_text += f"Difference (Quiet - Playing):\n"
            summary_text += f"  Mean Diff: {mean_diff:+.4f}\n"
            if mean_diff > 0:
                summary_text += f"  → Quiet hands are deeper\n"
                summary_text += f"    (more floating)\n"
            else:
                summary_text += f"  → Playing hands are deeper\n"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 9. 정규성 검정 결과
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        normality_text = f"{hand_type} Normality Tests\n" + "="*22 + "\n\n"
        
        if playing_data:
            playing_norm = self._test_normality(playing_data, f'{hand_type}_playing')
            normality_text += f"Playing:\n"
            if 'shapiro_p' in playing_norm and playing_norm['shapiro_p'] is not None:
                status = "Normal" if playing_norm['shapiro_normal'] else "Not Normal"
                normality_text += f"  Shapiro: p={playing_norm['shapiro_p']:.4f}\n  → {status}\n"
            normality_text += "\n"
        
        if quiet_data:
            quiet_norm = self._test_normality(quiet_data, f'{hand_type}_quiet')
            normality_text += f"Quiet:\n"
            if 'shapiro_p' in quiet_norm and quiet_norm['shapiro_p'] is not None:
                status = "Normal" if quiet_norm['shapiro_normal'] else "Not Normal"
                normality_text += f"  Shapiro: p={quiet_norm['shapiro_p']:.4f}\n  → {status}\n"
        
        ax9.text(0.05, 0.95, normality_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = self.output_dir / f"02{hand_type.lower()[0]}_{hand_type.lower()}_hand_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 {hand_type} Hand 상세 분석 저장: {filename}")
        plt.close()
    
    def _analyze_distributions(self, analysis_data: List[Dict], analysis_name: str):
        """분포 분석 및 시각화 - 전체 데이터 driven"""
        print("\n📊 통합 분포 분석 시작")
        print("=" * 40)
        
        df = pd.DataFrame(analysis_data)
        
        # MIDI 정보 유무 분리
        midi_data = df[df['is_playing'].notna()]  # MIDI 있는 데이터
        no_midi_data = df[df['is_playing'].isna()]  # MIDI 없는 데이터
        
        # 기본 통계
        print("📈 기본 통계:")
        print(f"   총 데이터: {len(df):,}개")
        print(f"   MIDI 매칭: {len(midi_data):,}개")
        print(f"   MIDI 없음: {len(no_midi_data):,}개")
        
        if len(midi_data) > 0:
            playing_data = len(midi_data[midi_data['is_playing']])
            quiet_data = len(midi_data) - playing_data
            print(f"   연주 중: {playing_data:,}개 ({playing_data/len(midi_data)*100:.1f}%)")
            print(f"   조용함: {quiet_data:,}개 ({quiet_data/len(midi_data)*100:.1f}%)")
        
        # 전체 데이터 범위 계산 (데이터 driven)
        global_min = df['depth'].min()
        global_max = df['depth'].max()
        print(f"   깊이값 범위: {global_min:.3f} ~ {global_max:.3f}")
        
        # 손별 분포 분석
        results = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, hand_type in enumerate(['Left', 'Right']):
            # MIDI 데이터만 분석 (playing/quiet 구분용)
            hand_midi_data = midi_data[midi_data['hand_type'] == hand_type]
            # 전체 손 데이터 (범위 설정용)
            hand_all_data = df[df['hand_type'] == hand_type]
            
            if len(hand_all_data) == 0:
                continue
            
            playing = hand_midi_data[hand_midi_data['is_playing']]['depth']
            quiet = hand_midi_data[~hand_midi_data['is_playing']]['depth']
            all_depths = hand_all_data['depth']  # 전체 데이터
            
            # 통계 계산
            stats = {
                'playing': {
                    'count': len(playing),
                    'mean': playing.mean() if len(playing) > 0 else 0,
                    'std': playing.std() if len(playing) > 0 else 0,
                    'median': playing.median() if len(playing) > 0 else 0
                },
                'quiet': {
                    'count': len(quiet),
                    'mean': quiet.mean() if len(quiet) > 0 else 0,
                    'std': quiet.std() if len(quiet) > 0 else 0,
                    'median': quiet.median() if len(quiet) > 0 else 0
                },
                'all': {
                    'count': len(all_depths),
                    'mean': all_depths.mean(),
                    'std': all_depths.std(),
                    'range': (all_depths.min(), all_depths.max())
                }
            }
            
            results[hand_type] = stats
            
            # 히스토그램 (전체 데이터 범위 기준)
            ax1 = axes[idx, 0]
            if len(playing) > 0:
                ax1.hist(playing, bins=50, alpha=0.7, label='MIDI Playing', color='blue', 
                        density=True, range=(global_min, global_max))
            if len(quiet) > 0:
                ax1.hist(quiet, bins=50, alpha=0.7, label='MIDI Quiet', color='red', 
                        density=True, range=(global_min, global_max))
            # 전체 데이터 범위로 x축 고정
            ax1.set_xlim(global_min, global_max)
            ax1.set_title(f'{hand_type} Hand Depth Distribution')
            ax1.set_xlabel('Depth Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 박스플롯 (전체 데이터 범위 기준)
            ax2 = axes[idx, 1]
            data_for_box = []
            labels_for_box = []
            if len(playing) > 0:
                data_for_box.append(playing)
                labels_for_box.append('Playing')
            if len(quiet) > 0:
                data_for_box.append(quiet)
                labels_for_box.append('Quiet')
            
            if data_for_box:
                ax2.boxplot(data_for_box, labels=labels_for_box)
                # 전체 데이터 범위로 y축 고정
                ax2.set_ylim(global_min, global_max)
                ax2.set_title(f'{hand_type} Hand Depth Boxplot')
                ax2.set_ylabel('Depth Value')
                ax2.grid(True, alpha=0.3)
            
            # 통계 출력
            print(f"\n👐 {hand_type}손 분석:")
            if len(playing) > 0:
                print(f"   연주 중: 평균 {stats['playing']['mean']:.3f} ± {stats['playing']['std']:.3f} ({stats['playing']['count']}개)")
            if len(quiet) > 0:
                print(f"   조용함: 평균 {stats['quiet']['mean']:.3f} ± {stats['quiet']['std']:.3f} ({stats['quiet']['count']}개)")
            
            if len(playing) > 0 and len(quiet) > 0:
                diff = stats['quiet']['mean'] - stats['playing']['mean']
                print(f"   차이: {diff:+.3f} (조용할 때가 더 큰 값 = 더 floating)")
        
        plt.tight_layout()
        
        # 이미지 저장
        filename = self.output_dir / f"03_basic_distribution_{analysis_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 기본 분포 분석 저장: {filename}")
        plt.close()
        
        # 결과 저장
        results['summary'] = {
            'analysis_name': analysis_name,
            'total_data': len(df),
            'midi_data': len(midi_data),
            'no_midi_data': len(no_midi_data),
            'playing_ratio': len(midi_data[midi_data['is_playing']]) / len(midi_data) if len(midi_data) > 0 else 0,
            'quiet_ratio': len(midi_data[~midi_data['is_playing']]) / len(midi_data) if len(midi_data) > 0 else 0,
            'depth_range': (global_min, global_max)
        }
        
        with open(f'midi_depth_analysis_{analysis_name}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        return results
    



def main():
    """메인 실행 함수"""
    print("🎵 MIDI-Floating 분포 분석기 (depth_data 통합)")
    print("목표: MIDI 연주 여부에 따른 손 깊이값 분포 비교")
    print("소스: depth_data/*.json 파일들")
    print("출력: analysis_results/ 폴더에 이미지 및 보고서 저장")
    print("=" * 60)
    
    analyzer = MIDIFloatingDistributionAnalyzer()
    results = analyzer.analyze_floating_distribution()
    
    if results:
        print("\n🎯 Playing vs Quiet 통합 분석 완료!")
        print("✅ 모든 depth_data JSON 파일을 통합하여 분석")
        print("✅ 이미 계산된 정확한 깊이값과 정확한 시간 정렬 사용")
        print("✅ 상세한 기술통계량 분석 (평균, 표준편차, 왜도, 첨도, 변동계수 등)")
        print("✅ 정규성 검정 (Shapiro-Wilk, Anderson-Darling)")
        print("✅ 다각도 분포 형태 분석 (히스토그램, 박스플롯, KDE, Q-Q plot, CDF)")
        print("✅ Playing vs Quiet 상황별 통계적 차이 시각화")
        print("✅ 손별 세부 분석 (Left/Right 각각)")
        print("✅ 분포가 다르면 MIDI 기반 floating 판정이 유효함을 증명")
        
        print(f"\n📁 분석 결과 파일들:")
        print(f"   📊 01_statistical_comparison.png - 통계량 비교표 및 바차트")
        print(f"   🫱 02l_left_hand_analysis.png - 왼손 전용 상세 분포 분석 (9개 서브플롯)")
        print(f"   🫲 02r_right_hand_analysis.png - 오른손 전용 상세 분포 분석 (9개 서브플롯)")
        print(f"   📈 02_overall_distribution_analysis.png - 전체 종합 분포 분석 (12개 서브플롯)")
        print(f"   📉 03_basic_distribution_all_combined.png - 기본 분포 분석")
        print(f"   📋 04_statistics_report.txt - 상세 통계 분석 보고서")
        print(f"   📖 05_plot_analysis_report.txt - 각 플롯별 해석 가이드 및 분석 결과")
        print(f"\n👀 결과 확인: analysis_results/ 폴더를 확인하세요!")
        print(f"\n🎯 권장 확인 순서:")
        print(f"   1️⃣ 05_plot_analysis_report.txt - 플롯 해석 방법 및 결과 이해")
        print(f"   2️⃣ 01_statistical_comparison.png - 전체 통계량 한눈에 보기")
        print(f"   3️⃣ 02l_left_hand_analysis.png - 왼손 상세 분석")
        print(f"   4️⃣ 02r_right_hand_analysis.png - 오른손 상세 분석")
        print(f"   5️⃣ 04_statistics_report.txt - 상세 수치 확인")


if __name__ == "__main__":
    main() 
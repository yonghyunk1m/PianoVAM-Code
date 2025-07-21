#!/usr/bin/env python3
"""
MIDI-Video Synchronization Validator
MIDI와 비디오 동기화 정확성을 검증하는 도구
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mido
from scipy import stats
from scipy.signal import correlate
import seaborn as sns

class MIDISyncValidator:
    """MIDI-비디오 동기화 검증기"""
    
    def __init__(self, depth_data_dir: str = "depth_data", 
                 midi_dir: str = "/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert"):
        self.depth_data_dir = Path(depth_data_dir)
        self.midi_dir = Path(midi_dir)
        self.target_fps = 20
        
    def load_sample_data(self, video_name: str) -> Dict:
        """특정 비디오의 깊이 데이터와 MIDI 데이터 로드"""
        # 깊이 데이터 로드
        depth_file = self.depth_data_dir / f"{video_name}_depth_data.json"
        if not depth_file.exists():
            raise FileNotFoundError(f"깊이 데이터 없음: {depth_file}")
        
        with open(depth_file, 'r') as f:
            depth_data = json.load(f)
        
        # 대응되는 MIDI 파일 찾기
        midi_files = list(self.midi_dir.glob(f"*{video_name}*"))
        if not midi_files:
            # 부분 매칭 시도
            midi_files = [f for f in self.midi_dir.glob("*.mid") 
                         if video_name in f.stem or f.stem in video_name]
        
        if not midi_files:
            raise FileNotFoundError(f"MIDI 파일 없음: {video_name}")
        
        midi_file = midi_files[0]
        midi_data = self._parse_midi_file(midi_file)
        
        return {
            'depth_data': depth_data,
            'midi_data': midi_data,
            'video_name': video_name
        }
    
    def _parse_midi_file(self, midi_file: Path) -> Dict:
        """MIDI 파일 파싱"""
        mid = mido.MidiFile(midi_file)
        
        # 모든 음표 이벤트 수집
        note_events = []
        
        for track in mid.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
                
                if msg.type in ['note_on', 'note_off']:
                    time_seconds = mido.tick2second(track_time, mid.ticks_per_beat, 500000)
                    hand_type = 'Right' if msg.note >= 60 else 'Left'
                    
                    note_events.append({
                        'time': time_seconds,
                        'note': msg.note,
                        'velocity': msg.velocity if msg.type == 'note_on' else 0,
                        'type': msg.type,
                        'hand': hand_type
                    })
        
        return {
            'file': midi_file.name,
            'events': sorted(note_events, key=lambda x: x['time']),
            'duration': max(event['time'] for event in note_events) if note_events else 0
        }
    
    def analyze_timing_correlation(self, data: Dict, time_shift_range: Tuple[float, float] = (-2.0, 2.0), 
                                 step: float = 0.1) -> Dict:
        """시간 이동에 따른 상관관계 분석"""
        print(f"🔍 타이밍 상관관계 분석: {data['video_name']}")
        
        # 프레임별 깊이 시계열 생성
        frame_data = data['depth_data']['frame_data']
        video_duration = max(frame['frame'] for frame in frame_data) / self.target_fps
        
        # 손별로 분석
        results = {}
        
        for hand_type in ['Left', 'Right']:
            print(f"   분석 중: {hand_type}손")
            
            # 깊이 시계열 생성
            depth_timeline = self._create_depth_timeline(frame_data, hand_type, video_duration)
            
            # MIDI 활성도 시계열 생성
            midi_timeline = self._create_midi_timeline(
                data['midi_data']['events'], hand_type, video_duration
            )
            
            if len(depth_timeline) == 0 or len(midi_timeline) == 0:
                continue
            
            # 다양한 시간 이동으로 상관관계 계산
            time_shifts = np.arange(time_shift_range[0], time_shift_range[1] + step, step)
            correlations = []
            
            for shift in time_shifts:
                shifted_midi = self._shift_timeline(midi_timeline, shift, video_duration)
                
                # 같은 길이로 맞추기
                min_len = min(len(depth_timeline), len(shifted_midi))
                if min_len > 10:  # 최소 길이 확인
                    depth_seg = depth_timeline[:min_len]
                    midi_seg = shifted_midi[:min_len]
                    
                    # 깊이 변화율과 MIDI 활성도 상관관계
                    depth_changes = np.diff(depth_seg)
                    midi_changes = np.diff(midi_seg)
                    
                    if len(depth_changes) > 5:
                        corr = np.corrcoef(depth_changes, midi_changes)[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0)
                    else:
                        correlations.append(0)
                else:
                    correlations.append(0)
            
            # 최적 시간 이동 찾기
            correlations = np.array(correlations)
            best_idx = np.argmax(np.abs(correlations))
            best_shift = time_shifts[best_idx]
            best_corr = correlations[best_idx]
            
            results[hand_type] = {
                'time_shifts': time_shifts.tolist(),
                'correlations': correlations.tolist(),
                'best_shift': best_shift,
                'best_correlation': best_corr,
                'depth_timeline': depth_timeline[:100],  # 샘플만 저장
                'midi_timeline': midi_timeline[:100]
            }
            
            print(f"      최적 시간 이동: {best_shift:.1f}초")
            print(f"      최대 상관관계: {best_corr:.3f}")
        
        return results
    
    def _create_depth_timeline(self, frame_data: List[Dict], hand_type: str, duration: float) -> np.ndarray:
        """프레임 데이터에서 깊이 시계열 생성"""
        # 시간 해상도 (초 단위)
        time_resolution = 0.05  # 20fps
        time_points = int(duration / time_resolution) + 1
        
        depth_timeline = np.full(time_points, np.nan)
        
        for frame in frame_data:
            frame_time = frame['frame'] / self.target_fps
            time_idx = int(frame_time / time_resolution)
            
            if time_idx < time_points:
                # 해당 손의 깊이 찾기
                for hand in frame.get('hands', []):
                    if hand['type'] == hand_type:
                        depth_timeline[time_idx] = hand['depth']
                        break
        
        # NaN 값 보간
        mask = ~np.isnan(depth_timeline)
        if np.sum(mask) > 1:
            from scipy.interpolate import interp1d
            valid_indices = np.where(mask)[0]
            valid_values = depth_timeline[mask]
            
            if len(valid_indices) > 1:
                interp_func = interp1d(valid_indices, valid_values, 
                                     kind='linear', fill_value='extrapolate')
                depth_timeline = interp_func(np.arange(time_points))
        
        return depth_timeline
    
    def _create_midi_timeline(self, midi_events: List[Dict], hand_type: str, duration: float) -> np.ndarray:
        """MIDI 이벤트에서 활성도 시계열 생성"""
        time_resolution = 0.05
        time_points = int(duration / time_resolution) + 1
        
        midi_timeline = np.zeros(time_points)
        active_notes = set()
        
        # 시간순으로 정렬된 이벤트 처리
        hand_events = [e for e in midi_events if e['hand'] == hand_type]
        
        for event in hand_events:
            time_idx = int(event['time'] / time_resolution)
            
            if event['type'] == 'note_on' and event['velocity'] > 0:
                active_notes.add(event['note'])
            elif event['type'] == 'note_off' or (event['type'] == 'note_on' and event['velocity'] == 0):
                active_notes.discard(event['note'])
            
            # 활성 음표 개수를 시계열에 기록
            if time_idx < time_points:
                midi_timeline[time_idx:] = len(active_notes)
        
        return midi_timeline
    
    def _shift_timeline(self, timeline: np.ndarray, shift_seconds: float, duration: float) -> np.ndarray:
        """시계열을 시간 이동"""
        time_resolution = 0.05
        shift_samples = int(shift_seconds / time_resolution)
        
        if shift_samples == 0:
            return timeline.copy()
        elif shift_samples > 0:
            # 양수: 미래로 이동 (앞쪽을 0으로 채움)
            shifted = np.concatenate([np.zeros(shift_samples), timeline])
        else:
            # 음수: 과거로 이동 (뒤쪽을 0으로 채움)
            shifted = np.concatenate([timeline[-shift_samples:], np.zeros(-shift_samples)])
        
        # 원래 길이로 자르기
        return shifted[:len(timeline)]
    
    def visualize_sync_analysis(self, results: Dict, video_name: str, save_path: str = "sync_analysis.png"):
        """동기화 분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for idx, hand_type in enumerate(['Left', 'Right']):
            if hand_type not in results:
                continue
                
            hand_results = results[hand_type]
            
            # 상관관계 vs 시간 이동
            ax1 = axes[idx, 0]
            ax1.plot(hand_results['time_shifts'], hand_results['correlations'], 'b-', linewidth=2)
            ax1.axvline(hand_results['best_shift'], color='red', linestyle='--', 
                       label=f"Best: {hand_results['best_shift']:.1f}s")
            ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
            ax1.set_xlabel('Time Shift (seconds)')
            ax1.set_ylabel('Correlation')
            ax1.set_title(f'{hand_type} Hand: Correlation vs Time Shift')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 원본 시계열 비교
            ax2 = axes[idx, 1]
            time_axis = np.arange(len(hand_results['depth_timeline'])) * 0.05
            
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(time_axis, hand_results['depth_timeline'], 'b-', 
                           label='Depth', linewidth=2)
            line2 = ax2_twin.plot(time_axis, hand_results['midi_timeline'], 'r-', 
                                label='MIDI Activity', linewidth=2)
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Depth Value', color='blue')
            ax2_twin.set_ylabel('Active Notes', color='red')
            ax2.set_title(f'{hand_type} Hand: Timeline Comparison')
            
            # 범례 통합
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper right')
            
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'MIDI-Video Synchronization Analysis: {video_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 시각화 저장: {save_path}")
    
    def generate_sync_report(self, video_name: str) -> Dict:
        """특정 비디오의 동기화 보고서 생성"""
        try:
            data = self.load_sample_data(video_name)
            results = self.analyze_timing_correlation(data)
            
            # 시각화
            self.visualize_sync_analysis(results, video_name, f"sync_analysis_{video_name}.png")
            
            # 종합 평가
            sync_quality = self._evaluate_sync_quality(results)
            
            report = {
                'video_name': video_name,
                'sync_analysis': results,
                'quality_assessment': sync_quality,
                'recommendations': self._generate_sync_recommendations(sync_quality)
            }
            
            return report
            
        except Exception as e:
            print(f"❌ 동기화 분석 실패: {e}")
            return {}
    
    def _evaluate_sync_quality(self, results: Dict) -> Dict:
        """동기화 품질 평가"""
        quality = {}
        
        for hand_type, hand_results in results.items():
            best_corr = abs(hand_results['best_correlation'])
            best_shift = abs(hand_results['best_shift'])
            
            # 품질 등급 결정
            if best_corr > 0.7 and best_shift < 0.2:
                grade = "Excellent"
            elif best_corr > 0.5 and best_shift < 0.5:
                grade = "Good"
            elif best_corr > 0.3 and best_shift < 1.0:
                grade = "Fair"
            else:
                grade = "Poor"
            
            quality[hand_type] = {
                'grade': grade,
                'correlation': best_corr,
                'time_offset': hand_results['best_shift'],
                'needs_correction': best_shift > 0.1 or best_corr < 0.5
            }
        
        return quality
    
    def _generate_sync_recommendations(self, quality: Dict) -> List[str]:
        """동기화 개선 권장사항 생성"""
        recommendations = []
        
        for hand_type, qual in quality.items():
            if qual['needs_correction']:
                if abs(qual['time_offset']) > 0.5:
                    recommendations.append(
                        f"{hand_type}손: 큰 시간 오차 ({qual['time_offset']:.1f}초) - 녹화 시작점 재조정 필요"
                    )
                elif qual['correlation'] < 0.3:
                    recommendations.append(
                        f"{hand_type}손: 낮은 상관관계 ({qual['correlation']:.2f}) - 손 구분 알고리즘 재검토 필요"
                    )
                else:
                    recommendations.append(
                        f"{hand_type}손: 미세 조정 필요 (오차: {qual['time_offset']:.1f}초)"
                    )
        
        if not recommendations:
            recommendations.append("동기화 상태 양호 - 추가 조정 불필요")
        
        return recommendations

def main():
    """메인 실행 함수"""
    validator = MIDISyncValidator()
    
    # 첫 번째 비디오로 테스트
    try:
        # 가능한 비디오 이름들 시도
        test_videos = ["2024-02-15_20-38-23", "2024-03-04_04-14-36", "2024-09-05_21-13-49"]
        
        for video_name in test_videos:
            try:
                print(f"\n🔍 동기화 검증: {video_name}")
                report = validator.generate_sync_report(video_name)
                
                if report:
                    print(f"\n📊 {video_name} 동기화 품질:")
                    for hand_type, quality in report['quality_assessment'].items():
                        print(f"   {hand_type}손: {quality['grade']} (상관관계: {quality['correlation']:.3f}, 오차: {quality['time_offset']:.1f}초)")
                    
                    print(f"\n💡 권장사항:")
                    for rec in report['recommendations']:
                        print(f"   - {rec}")
                    
                    break  # 성공하면 중단
                    
            except FileNotFoundError:
                continue
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON-MIDI Duration Comparison Tool
JSON 깊이 데이터와 MIDI 파일의 길이를 비교하여 매칭 품질을 확인하는 도구
참고: 비디오 파일(.mp4)은 사용하지 않음, JSON과 MIDI만 사용
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

# MIDI 처리를 위한 라이브러리
try:
    import mido
    MIDO_AVAILABLE = True
    print("✅ MIDI processing module loaded successfully")
except ImportError:
    print("⚠️ MIDI module not available: pip install mido")
    MIDO_AVAILABLE = False

class DurationComparator:
    """비디오와 MIDI 길이 비교 클래스"""
    
    def __init__(self, 
                 depth_data_dir: str = "depth_data",
                 midi_dir: str = "/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert",
                 target_fps: int = 20):
        """
        Args:
            depth_data_dir: 깊이 데이터 디렉토리
            midi_dir: MIDI 파일 디렉토리  
            target_fps: 타겟 FPS
        """
        self.depth_data_dir = Path(depth_data_dir)
        self.midi_dir = Path(midi_dir)
        self.target_fps = target_fps
        
        print(f"📁 Depth data directory: {self.depth_data_dir}")
        print(f"🎵 MIDI directory: {self.midi_dir}")
        print(f"🎬 Target FPS: {self.target_fps}")
        
        # 디렉토리 존재 확인
        if not self.depth_data_dir.exists():
            raise FileNotFoundError(f"Depth data directory not found: {self.depth_data_dir}")
        
        if not self.midi_dir.exists():
            print(f"⚠️ MIDI directory not found: {self.midi_dir}")
    
    def calculate_video_duration(self, depth_data: Dict) -> float:
        """
        깊이 데이터에서 비디오 길이 계산
        방식: (max_frame - min_frame) / target_fps
        """
        frame_data = depth_data.get('frame_data', [])
        if not frame_data:
            return 0.0
        
        frames = [frame['frame'] for frame in frame_data]
        max_frame = max(frames)
        min_frame = min(frames)
        
        # 실제 프레임 수로 길이 계산 (오프셋 보정)
        actual_frames = max_frame - min_frame
        duration = actual_frames / self.target_fps
        
        return duration
    
    def calculate_midi_duration(self, midi_file: Path) -> float:
        """
        MIDI 파일에서 길이 계산 - 실제 MIDI 템포 사용
        방식: 모든 이벤트의 최대 시간값
        """
        if not MIDO_AVAILABLE:
            return 0.0
        
        try:
            mid = mido.MidiFile(midi_file)
            max_time = 0.0
            
            # MIDI 파일에서 실제 템포 추출
            current_tempo = 500000  # 기본값 (120 BPM)
            
            # 첫 번째 패스: 템포 정보 수집
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        current_tempo = msg.tempo
                        break  # 첫 번째 템포만 사용
                if current_tempo != 500000:  # 템포를 찾았으면 중단
                    break
            
            # 트랙별 처리
            for track in mid.tracks:
                track_time = 0
                
                for msg in track:
                    track_time += msg.time
                    
                    # 실제 MIDI 템포를 사용한 시간 변환
                    time_seconds = mido.tick2second(track_time, mid.ticks_per_beat, current_tempo)
                    max_time = max(max_time, time_seconds)
            
            return max_time
            
        except Exception as e:
            print(f"❌ MIDI duration calculation failed for {midi_file.name}: {e}")
            return 0.0
    
    def load_depth_files(self) -> List[Dict]:
        """깊이 데이터 파일들 로드"""
        depth_files = list(self.depth_data_dir.glob("*_depth_data.json"))
        print(f"📊 Found {len(depth_files)} depth data files")
        
        datasets = []
        for file_path in depth_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 세션 식별자 추출 (비디오 파일은 사용하지 않음)
                video_name = data.get('video_name', file_path.stem.replace('_depth_data', ''))
                data['video_name'] = video_name
                
                datasets.append(data)
                print(f"   ✅ Loaded: {video_name}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {file_path.name}: {e}")
        
        return datasets
    
    def find_midi_files(self) -> List[Path]:
        """MIDI 파일들 찾기"""
        if not self.midi_dir.exists():
            return []
        
        midi_files = list(self.midi_dir.glob("*.mid")) + list(self.midi_dir.glob("*.midi"))
        print(f"🎵 Found {len(midi_files)} MIDI files")
        
        for midi_file in midi_files[:5]:  # 처음 5개만 출력
            print(f"   📄 {midi_file.name}")
        
        if len(midi_files) > 5:
            print(f"   ... and {len(midi_files) - 5} more")
        
        return midi_files
    
    # 불필요한 복잡한 매칭 함수 제거됨 (단순 파일명 매칭으로 대체)
    
    def compare_durations(self) -> pd.DataFrame:
        """비디오와 MIDI 길이 비교 수행"""
        print("\n🔍 Starting duration comparison analysis...")
        
        # 데이터 로드
        depth_datasets = self.load_depth_files()
        midi_files = self.find_midi_files()
        
        if not depth_datasets:
            print("❌ No depth data files found")
            return pd.DataFrame()
        
        if not midi_files:
            print("❌ No MIDI files found")
            return pd.DataFrame()
        
        # 비교 결과 저장
        comparisons = []
        
        print(f"\n📊 Analyzing {len(depth_datasets)} JSON files...")
        
        for depth_data in depth_datasets:
            video_name = depth_data['video_name']
            
            # 비디오 길이 계산
            video_duration = self.calculate_video_duration(depth_data)
            
            print(f"\n📊 Session: {video_name}")
            print(f"   Duration: {video_duration:.2f} seconds")
            
            # 최적 MIDI 파일 찾기
            best_match = None
            best_score = 0
            best_midi_duration = 0
            
            for midi_file in midi_files:
                midi_name = midi_file.stem
                
                # 간단한 파일명 패턴 매칭
                video_pattern = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', video_name)
                if video_pattern and video_pattern.group(1) in midi_name:
                    print(f"   🎯 정확한 매칭 발견: {midi_file.name}")
                    
                    midi_duration = self.calculate_midi_duration(midi_file)
                    duration_ratio = min(video_duration, midi_duration) / max(video_duration, midi_duration) if video_duration > 0 and midi_duration > 0 else 0
                    
                    best_match = {
                        'midi_file': midi_file.name,
                        'midi_name': midi_name,
                        'midi_duration': midi_duration,
                        'name_score': 100.0,  # 정확한 매칭
                        'duration_score': duration_ratio * 100,
                        'total_score': 100.0,  # 정확한 매칭
                        'duration_ratio': duration_ratio
                    }
                    break  # 정확한 매칭을 찾았으므로 중단
            
            # 결과 저장
            if best_match:
                print(f"   🎵 Best MIDI match: {best_match['midi_file']}")
                print(f"   📏 MIDI duration: {best_match['midi_duration']:.2f} seconds")
                print(f"   📊 Duration ratio: {best_match['duration_ratio']:.3f}")
                print(f"   🎯 Total score: {best_match['total_score']:.1f}")
                
                # 길이 차이 분석
                duration_diff = abs(video_duration - best_match['midi_duration'])
                duration_diff_percent = (duration_diff / max(video_duration, best_match['midi_duration'])) * 100
                
                comparisons.append({
                    'video_name': video_name,
                    'video_duration': video_duration,
                    'midi_file': best_match['midi_file'],
                    'midi_duration': best_match['midi_duration'],
                    'duration_diff': duration_diff,
                    'duration_diff_percent': duration_diff_percent,
                    'duration_ratio': best_match['duration_ratio'],
                    'name_score': best_match['name_score'],
                    'duration_score': best_match['duration_score'],
                    'total_score': best_match['total_score'],
                    'match_quality': self._determine_match_quality(best_match['total_score'])
                })
            else:
                print(f"   ❌ No suitable MIDI match found")
                comparisons.append({
                    'video_name': video_name,
                    'video_duration': video_duration,
                    'midi_file': None,
                    'midi_duration': 0,
                    'duration_diff': 0,
                    'duration_diff_percent': 0,
                    'duration_ratio': 0,
                    'name_score': 0,
                    'duration_score': 0,
                    'total_score': 0,
                    'match_quality': 'no_match'
                })
        
        # DataFrame 생성
        df = pd.DataFrame(comparisons)
        return df
    
    def _determine_match_quality(self, total_score: float) -> str:
        """매칭 품질 결정"""
        if total_score >= 85:
            return 'perfect'
        elif total_score >= 70:
            return 'good'
        elif total_score >= 50:
            return 'partial'
        elif total_score >= 30:
            return 'poor'
        else:
            return 'very_poor'
    
    def print_summary(self, df: pd.DataFrame):
        """결과 요약 출력"""
        if len(df) == 0:
            print("❌ No comparison data available")
            return
        
        print("\n" + "="*70)
        print("📊 DURATION COMPARISON SUMMARY")
        print("="*70)
        
        # 전체 통계
        total_videos = len(df)
        matched_videos = len(df[df['midi_file'].notna()])
        
        print(f"📊 Total JSON files: {total_videos}")
        print(f"🎵 Matched with MIDI: {matched_videos} ({matched_videos/total_videos*100:.1f}%)")
        print(f"❌ No MIDI match: {total_videos - matched_videos}")
        
        if matched_videos > 0:
            matched_df = df[df['midi_file'].notna()]
            
            # 길이 차이 통계
            print(f"\n📏 Duration Analysis (for matched pairs):")
            print(f"   Average duration difference: {matched_df['duration_diff'].mean():.2f} ± {matched_df['duration_diff'].std():.2f} seconds")
            print(f"   Average duration difference: {matched_df['duration_diff_percent'].mean():.1f} ± {matched_df['duration_diff_percent'].std():.1f}%")
            print(f"   Average duration ratio: {matched_df['duration_ratio'].mean():.3f}")
            
            # 매칭 품질별 분포
            print(f"\n🎯 Match Quality Distribution:")
            quality_counts = matched_df['match_quality'].value_counts()
            for quality, count in quality_counts.items():
                percentage = count / matched_videos * 100
                print(f"   {quality}: {count} ({percentage:.1f}%)")
            
            # 우수한 매칭 (길이 차이 10% 이내)
            good_matches = matched_df[matched_df['duration_diff_percent'] <= 10]
            print(f"\n✅ Excellent duration matches (≤10% difference): {len(good_matches)} ({len(good_matches)/matched_videos*100:.1f}%)")
            
            # 문제가 있는 매칭 (길이 차이 50% 이상)
            poor_matches = matched_df[matched_df['duration_diff_percent'] >= 50]
            if len(poor_matches) > 0:
                print(f"\n⚠️ Poor duration matches (≥50% difference): {len(poor_matches)}")
                for _, row in poor_matches.iterrows():
                    print(f"   {row['video_name']}: {row['video_duration']:.1f}s vs {row['midi_duration']:.1f}s ({row['duration_diff_percent']:.1f}% diff)")
        
        print("\n" + "="*70)
    
    def save_results(self, df: pd.DataFrame, output_file: str = "duration_comparison_results.csv"):
        """결과를 CSV 파일로 저장"""
        if len(df) > 0:
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"💾 Results saved to: {output_file}")
        else:
            print("❌ No data to save")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JSON-MIDI Duration Comparison Tool')
    parser.add_argument('--depth-dir', default="depth_data", 
                       help='Depth data directory (default: depth_data)')
    parser.add_argument('--midi-dir', default="/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert",
                       help='MIDI files directory')
    parser.add_argument('--fps', type=int, default=20,
                       help='Target FPS (default: 20)')
    parser.add_argument('--output', default="duration_comparison_results.csv",
                       help='Output CSV file (default: duration_comparison_results.csv)')
    
    args = parser.parse_args()
    
    try:
        # 비교기 초기화
        comparator = DurationComparator(
            depth_data_dir=args.depth_dir,
            midi_dir=args.midi_dir,
            target_fps=args.fps
        )
        
        # 길이 비교 수행
        results_df = comparator.compare_durations()
        
        # 결과 요약 출력
        comparator.print_summary(results_df)
        
        # 결과 저장
        comparator.save_results(results_df, args.output)
        
        print(f"\n✅ Duration comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during duration comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
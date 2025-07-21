#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI 기반 Floating Hand Detection
MIDI가 전혀 연주되지 않은 순간에만 floating hands로 판정
"""

import mido
import numpy as np
import json
import pickle
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from stqdm import stqdm
import time


class MIDIBasedFloatingDetector:
    """MIDI 정보를 활용한 스마트 Floating Hand Detection"""
    
    def __init__(self, 
                 depth_data_dir: str = "depth_data",
                 midi_dir: str = "/home/jhbae/PianoVAM-Code/FingeringDetection/midiconvert",
                 target_fps: int = 20):
        self.depth_data_dir = Path(depth_data_dir)
        self.midi_dir = Path(midi_dir)
        self.target_fps = target_fps
        self.hand_split_note = 60  # Middle C (C4)
        
        # 캐시
        self._midi_cache = {}
        
    def detect_floating_hands_with_midi(self, handlist: List, video_name: str, ratio: float) -> List:
        """
        MIDI 정보를 활용한 스마트 floating hands detection
        
        Args:
            handlist: 손 데이터 리스트
            video_name: 비디오 이름 (MIDI 매칭용)
            ratio: 비디오 비율
            
        Returns:
            floating_results: [frame, handtype, depth_value, floating_status] 리스트
        """
        print(f"🎵 MIDI 기반 Floating Hands Detection 시작: {video_name}")
        
        # 1. MIDI 데이터 로드
        midi_data = self._load_and_match_midi(video_name)
        if not midi_data:
            print(f"   ⚠️ MIDI 데이터 없음 - 기존 방식으로 폴백")
            return self._fallback_depth_only_detection(handlist, ratio)
        
        # 2. MIDI 타임라인 생성
        midi_timeline = self._create_midi_timeline(midi_data)
        print(f"   🎵 MIDI 타임라인 생성: {len(midi_timeline)} 시간점")
        
        # 3. 개선된 floating detection
        floating_results = []
        
        print(f"   🔍 프레임별 분석 시작...")
        for hands in stqdm(handlist, desc="MIDI 기반 floating 분석"):
            if not hands:
                continue
                
            for hand in hands:
                frame_num = hand.handframe
                hand_type = hand.handtype
                frame_time = frame_num / self.target_fps
                
                # 기본 깊이 계산 (간단한 버전)
                depth_value = self._calculate_simple_depth(hand)
                
                # MIDI 기반 연주 상태 확인
                is_playing_midi = self._is_hand_playing_at_time(midi_timeline, hand_type, frame_time)
                
                # 스마트 floating 판정
                floating_status = self._smart_floating_decision(
                    depth_value, is_playing_midi, hand_type, frame_time
                )
                
                floating_results.append([
                    frame_num,
                    hand_type, 
                    depth_value,
                    floating_status
                ])
        
        # 통계 출력
        self._print_detection_stats(floating_results, midi_data)
        
        return floating_results
    
    def _load_and_match_midi(self, video_name: str) -> Optional[Dict]:
        """비디오명과 매칭되는 MIDI 파일 찾기 및 로드"""
        # 캐시 확인
        if video_name in self._midi_cache:
            return self._midi_cache[video_name]
        
        # 날짜-시간 패턴 추출 (YYYY-MM-DD_HH-MM-SS)
        video_pattern = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', video_name)
        if not video_pattern:
            print(f"   ⚠️ 날짜-시간 패턴을 찾을 수 없음: {video_name}")
            return None
        
        target_pattern = video_pattern.group(1)
        
        # 매칭되는 MIDI 파일 찾기
        midi_files = list(self.midi_dir.glob("*.mid")) + list(self.midi_dir.glob("*.midi"))
        
        for midi_file in midi_files:
            if target_pattern in midi_file.stem:
                print(f"   ✅ MIDI 매칭 성공: {midi_file.name}")
                midi_data = self._parse_midi_file(midi_file)
                
                # 캐싱
                self._midi_cache[video_name] = midi_data
                return midi_data
        
        print(f"   ❌ 매칭되는 MIDI 파일 없음: {target_pattern}")
        return None
    
    def _parse_midi_file(self, midi_file: Path) -> Dict:
        """MIDI 파일 파싱 - 실제 템포 사용"""
        try:
            mid = mido.MidiFile(midi_file)
            
            # 실제 템포 추출
            current_tempo = 500000  # 기본값 (120 BPM)
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        current_tempo = msg.tempo
                        break
                if current_tempo != 500000:
                    break
            
            bpm = 60000000 / current_tempo
            print(f"   🎵 MIDI 템포: {bpm:.1f} BPM")
            
            # 음표 세그먼트 추출
            note_segments = {'Left': [], 'Right': []}
            active_notes = {'Left': {}, 'Right': {}}
            
            for track in mid.tracks:
                track_time = 0
                
                for msg in track:
                    track_time += msg.time
                    
                    if msg.type in ['note_on', 'note_off']:
                        time_seconds = mido.tick2second(track_time, mid.ticks_per_beat, current_tempo)
                        hand_type = 'Right' if msg.note >= self.hand_split_note else 'Left'
                        
                        if msg.type == 'note_on' and msg.velocity > 0:
                            # 음표 시작
                            active_notes[hand_type][msg.note] = {
                                'start_time': time_seconds,
                                'velocity': msg.velocity
                            }
                        else:
                            # 음표 종료
                            if msg.note in active_notes[hand_type]:
                                start_info = active_notes[hand_type].pop(msg.note)
                                duration = time_seconds - start_info['start_time']
                                
                                if duration > 0.01:  # 10ms 이상
                                    note_segments[hand_type].append({
                                        'start_time': start_info['start_time'],
                                        'end_time': time_seconds,
                                        'note': msg.note,
                                        'velocity': start_info['velocity'],
                                        'duration': duration
                                    })
            
            total_segments = len(note_segments['Left']) + len(note_segments['Right'])
            max_time = max([seg['end_time'] for segs in note_segments.values() for seg in segs]) if total_segments > 0 else 0
            
            print(f"   🎵 파싱 완료: {total_segments}개 음표, {max_time:.1f}초")
            
            return {
                'file': midi_file.name,
                'note_segments': note_segments,
                'total_segments': total_segments,
                'duration': max_time,
                'tempo_bpm': bpm
            }
            
        except Exception as e:
            print(f"   ❌ MIDI 파싱 실패: {e}")
            return None
    
    def _create_midi_timeline(self, midi_data: Dict) -> Dict:
        """MIDI 타임라인 생성 - 시간별 연주 상태"""
        duration = midi_data['duration']
        time_resolution = 0.05  # 50ms 해상도
        time_points = int(duration / time_resolution) + 1
        
        timeline = {
            'Left': np.zeros(time_points, dtype=bool),
            'Right': np.zeros(time_points, dtype=bool),
            'time_resolution': time_resolution,
            'duration': duration
        }
        
        # 각 손별로 타임라인 구축
        for hand_type in ['Left', 'Right']:
            segments = midi_data['note_segments'].get(hand_type, [])
            
            for segment in segments:
                start_idx = int(segment['start_time'] / time_resolution)
                end_idx = int(segment['end_time'] / time_resolution)
                
                # 해당 시간 구간을 연주 중으로 마킹
                if start_idx < time_points and end_idx >= 0:
                    start_idx = max(0, start_idx)
                    end_idx = min(time_points - 1, end_idx)
                    timeline[hand_type][start_idx:end_idx + 1] = True
        
        # 통계
        left_playing_time = np.sum(timeline['Left']) * time_resolution
        right_playing_time = np.sum(timeline['Right']) * time_resolution
        
        print(f"   📊 연주 시간 분석:")
        print(f"     좌손: {left_playing_time:.1f}초 ({left_playing_time/duration*100:.1f}%)")
        print(f"     우손: {right_playing_time:.1f}초 ({right_playing_time/duration*100:.1f}%)")
        
        return timeline
    
    def _is_hand_playing_at_time(self, midi_timeline: Dict, hand_type: str, time_seconds: float) -> bool:
        """특정 시간에 해당 손이 연주 중인지 확인"""
        time_resolution = midi_timeline['time_resolution']
        time_idx = int(time_seconds / time_resolution)
        
        if 0 <= time_idx < len(midi_timeline[hand_type]):
            return bool(midi_timeline[hand_type][time_idx])
        
        return False
    
    def _calculate_simple_depth(self, hand) -> float:
        """간단한 깊이 계산 (복잡한 3D 계산 대신)"""
        # 손목과 손가락 끝점의 Y 좌표 차이로 간단한 깊이 추정
        try:
            wrist_y = hand.handlandmark[0].y
            fingertip_y = hand.handlandmark[8].y  # 검지 끝
            
            # Y 차이가 클수록 손이 더 수직에 가까움 (floating 가능성 높음)
            depth_approximation = abs(wrist_y - fingertip_y) + 0.5
            
            return depth_approximation
        except:
            return 1.0  # 기본값
    
    def _smart_floating_decision(self, depth_value: float, is_playing_midi: bool, 
                               hand_type: str, frame_time: float) -> str:
        """스마트 floating 판정 로직"""
        
        # 핵심 규칙: MIDI가 연주되고 있으면 절대 floating이 아님
        if is_playing_midi:
            return 'notfloating'
        
        # MIDI가 연주되지 않는 순간에만 깊이로 추가 판정
        # 더 엄격한 임계값 사용 (기존 0.9 → 0.7)
        if depth_value < 0.7:
            return 'floating'
        else:
            return 'notfloating'
    
    def _fallback_depth_only_detection(self, handlist: List, ratio: float) -> List:
        """MIDI 데이터가 없을 때 기존 깊이 기반 검출 사용"""
        print(f"   🔄 깊이 기반 검출로 폴백")
        
        floating_results = []
        
        for hands in handlist:
            if not hands:
                continue
                
            for hand in hands:
                depth_value = self._calculate_simple_depth(hand)
                floating_status = 'floating' if depth_value < 0.9 else 'notfloating'
                
                floating_results.append([
                    hand.handframe,
                    hand.handtype,
                    depth_value,
                    floating_status
                ])
        
        return floating_results
    
    def _print_detection_stats(self, floating_results: List, midi_data: Optional[Dict]):
        """검출 결과 통계 출력"""
        if not floating_results:
            return
        
        total_hands = len(floating_results)
        floating_count = len([r for r in floating_results if r[3] == 'floating'])
        playing_count = total_hands - floating_count
        
        print(f"\n📊 MIDI 기반 Floating Detection 결과:")
        print(f"   총 손 데이터: {total_hands:,}개")
        print(f"   Floating 판정: {floating_count:,}개 ({floating_count/total_hands*100:.1f}%)")
        print(f"   Playing 판정: {playing_count:,}개 ({playing_count/total_hands*100:.1f}%)")
        
        if midi_data:
            print(f"   MIDI 음표 수: {midi_data['total_segments']}개")
            print(f"   MIDI 길이: {midi_data['duration']:.1f}초")
    
    def save_results(self, floating_results: List, output_path: str):
        """결과 저장"""
        with open(output_path, 'wb') as f:
            pickle.dump(floating_results, f)
        print(f"💾 결과 저장: {output_path}")


def detect_floating_with_midi_integration(handlist: List, video_name: str, ratio: float) -> List:
    """
    메인 함수: MIDI 통합 floating hands detection
    
    Args:
        handlist: 손 데이터 리스트
        video_name: 비디오 이름
        ratio: 비디오 비율
    
    Returns:
        floating_results: [frame, handtype, depth_value, floating_status] 리스트
    """
    detector = MIDIBasedFloatingDetector()
    return detector.detect_floating_hands_with_midi(handlist, video_name, ratio)


if __name__ == "__main__":
    print("🎵 MIDI 기반 Floating Hand Detection")
    print("=" * 50)
    print("✅ 핵심 개선 사항:")
    print("   • MIDI가 연주되는 순간 → 절대 floating 아님")
    print("   • MIDI가 조용한 순간 → 깊이로 추가 판정") 
    print("   • 더 정확하고 논리적인 floating 감지")
    print("   • 실제 연주 의도를 반영한 스마트 판정") 
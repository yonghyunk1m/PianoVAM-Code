#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI 기반 Floating Detection 테스트 스크립트
기존 방식과 새로운 MIDI 기반 방식을 비교
JSON 데이터 형식 지원
"""

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# 프로젝트 모듈 임포트
sys.path.append('.')
from midi_based_floating_detector import MIDIBasedFloatingDetector, detect_floating_with_midi_integration


@dataclass
class HandLandmark:
    """손 랜드마크 포인트"""
    x: float
    y: float


@dataclass
class HandData:
    """손 데이터 클래스 (기존 handclass와 호환)"""
    handtype: str
    handlandmark: List[HandLandmark]
    handframe: int
    handdepth: float = 1.0


def load_test_data(target_video: str = None):
    """Cache 디렉토리의 handlist pickle 파일 로드"""
    print("🔍 Cache handlist 데이터 검색 중...")
    
    cache_dir = Path("cache")
    if not cache_dir.exists():
        print("❌ cache 디렉토리를 찾을 수 없습니다.")
        return None, None, None
    
    # 사용 가능한 handlist 파일들 찾기
    handlist_files = []
    for item in cache_dir.iterdir():
        if item.is_dir():
            potential_handlist = list(item.glob("handlist_*.pkl"))
            if potential_handlist:
                handlist_files.extend(potential_handlist)
    
    if not handlist_files:
        print("❌ handlist pickle 파일을 찾을 수 없습니다.")
        return None, None, None
    
    # 특정 비디오 지정 또는 첫 번째 데이터 사용
    if target_video:
        target_file = next((f for f in handlist_files if target_video in f.stem), None)
        if not target_file:
            print(f"❌ 지정된 비디오를 찾을 수 없습니다: {target_video}")
            print(f"사용 가능한 파일들: {[f.stem for f in handlist_files]}")
            return None, None, None
    else:
        # 크기가 적당한 파일 선택 (테스트 용이성을 위해) - limit600이 좋을 것 같음
        suitable_files = [f for f in handlist_files if "limit600" in f.stem or "limit1200" in f.stem]
        if suitable_files:
            target_file = suitable_files[0]
        else:
            target_file = handlist_files[0]
    
    print(f"✅ 테스트 데이터: {target_file.stem}")
    print(f"   파일 크기: {target_file.stat().st_size / 1024 / 1024:.1f}MB")
    
    # handlist 로드
    with open(target_file, 'rb') as f:
        handlist = pickle.load(f)
    
    # 비디오 이름 추출 
    video_name = target_file.stem.replace('handlist_', '').replace('_limit600', '').replace('_limit1200', '').replace('_limit6000', '')
    
    # 비디오 정보 (간단한 비율 계산)
    ratio = 9/16  # 기본 비율
    
    print(f"   handlist: {len(handlist)}프레임")
    total_hands = sum(len(hands) for hands in handlist if hands)
    print(f"   총 손 데이터: {total_hands:,}개")
    
    return handlist, video_name, ratio



def run_comparison_test(handlist, video_name, ratio):
    """기존 방식과 MIDI 기반 방식 비교 테스트"""
    print("\n🔄 두 가지 방식으로 Floating Detection 실행")
    print("=" * 60)
    
    # 1. 새로운 MIDI 기반 방식
    print("\n1️⃣ MIDI 기반 Floating Detection")
    print("-" * 40)
    
    midi_detector = MIDIBasedFloatingDetector()
    midi_results = midi_detector.detect_floating_hands_with_midi(handlist, video_name, ratio)
    
    # 2. 기존 깊이 기반 방식 (간단 버전)
    print("\n2️⃣ 기존 깊이 기반 Detection")
    print("-" * 40)
    
    original_results = []
    for hands in handlist:
        if not hands:
            continue
        for hand in hands:
            # 간단한 깊이 계산
            depth_value = midi_detector._calculate_simple_depth(hand)
            floating_status = 'floating' if depth_value < 0.9 else 'notfloating'
            
            original_results.append([
                hand.handframe,
                hand.handtype,
                depth_value,
                floating_status
            ])
    
    print(f"   🔍 기존 방식 완료: {len(original_results)}개 분석")
    
    return midi_results, original_results


def analyze_differences(midi_results, original_results, video_name):
    """두 방식의 차이점 분석"""
    print("\n📊 결과 비교 분석")
    print("=" * 60)
    
    # DataFrame 변환
    midi_df = pd.DataFrame(midi_results, columns=['frame', 'hand_type', 'depth', 'floating_status'])
    original_df = pd.DataFrame(original_results, columns=['frame', 'hand_type', 'depth', 'floating_status'])
    
    # 기본 통계
    print(f"\n📈 기본 통계:")
    print(f"   총 손 데이터: {len(midi_df)}개")
    
    # Floating 비율 비교
    midi_floating_rate = (midi_df['floating_status'] == 'floating').mean() * 100
    original_floating_rate = (original_df['floating_status'] == 'floating').mean() * 100
    
    print(f"\n🎯 Floating 비율 비교:")
    print(f"   MIDI 기반:    {midi_floating_rate:.1f}%")
    print(f"   기존 방식:    {original_floating_rate:.1f}%")
    print(f"   차이:        {midi_floating_rate - original_floating_rate:+.1f}%")
    
    # 손별 분석
    print(f"\n👐 손별 분석:")
    for hand_type in ['Left', 'Right']:
        midi_hand = midi_df[midi_df['hand_type'] == hand_type]
        original_hand = original_df[original_df['hand_type'] == hand_type]
        
        if len(midi_hand) > 0:
            midi_rate = (midi_hand['floating_status'] == 'floating').mean() * 100
            original_rate = (original_hand['floating_status'] == 'floating').mean() * 100
            
            print(f"   {hand_type}손:")
            print(f"     MIDI 기반: {midi_rate:.1f}%")
            print(f"     기존 방식: {original_rate:.1f}%")
            print(f"     차이: {midi_rate - original_rate:+.1f}%")
    
    # 프레임별 일치도 분석
    print(f"\n🔍 판정 변화 분석:")
    
    # 같은 프레임-손 조합으로 병합
    merged = midi_df.merge(original_df, on=['frame', 'hand_type'], suffixes=('_midi', '_original'))
    
    # 판정이 다른 경우들
    different_judgments = merged[merged['floating_status_midi'] != merged['floating_status_original']]
    
    if len(different_judgments) > 0:
        print(f"   판정 차이: {len(different_judgments)}개 ({len(different_judgments)/len(merged)*100:.1f}%)")
        
        # 변화 패턴 분석
        midi_to_playing = different_judgments[
            (different_judgments['floating_status_midi'] == 'notfloating') & 
            (different_judgments['floating_status_original'] == 'floating')
        ]
        
        playing_to_midi = different_judgments[
            (different_judgments['floating_status_midi'] == 'floating') & 
            (different_judgments['floating_status_original'] == 'notfloating')
        ]
        
        print(f"   Floating → Playing: {len(midi_to_playing)}개 (MIDI 정보로 연주 중 판정)")
        print(f"   Playing → Floating: {len(playing_to_midi)}개 (MIDI 정보로 조용한 순간 판정)")
    else:
        print(f"   모든 판정이 일치합니다!")
    
    return merged


def visualize_timeline_comparison(merged_df, video_name, max_samples=2000):
    """타임라인 시각화"""
    print(f"\n📊 타임라인 시각화 생성...")
    
    # 샘플링 (너무 많으면 그래프가 복잡해짐)
    if len(merged_df) > max_samples:
        sample_df = merged_df.sample(n=max_samples).sort_values('frame')
    else:
        sample_df = merged_df.sort_values('frame')
    
    # 시간 축 생성 (프레임 → 초)
    sample_df['time'] = sample_df['frame'] / 20  # 20fps 가정
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 좌손과 우손별로 시각화
    for idx, hand_type in enumerate(['Left', 'Right']):
        ax = axes[idx]
        
        hand_data = sample_df[sample_df['hand_type'] == hand_type]
        if len(hand_data) == 0:
            continue
        
        # MIDI 기반 결과
        midi_floating = hand_data[hand_data['floating_status_midi'] == 'floating']['time']
        midi_playing = hand_data[hand_data['floating_status_midi'] == 'notfloating']['time']
        
        # 기존 방식 결과  
        original_floating = hand_data[hand_data['floating_status_original'] == 'floating']['time']
        original_playing = hand_data[hand_data['floating_status_original'] == 'notfloating']['time']
        
        # 시각화
        ax.scatter(midi_floating, [1] * len(midi_floating), 
                  alpha=0.6, s=2, color='red', label='MIDI: Floating')
        ax.scatter(midi_playing, [1] * len(midi_playing), 
                  alpha=0.6, s=2, color='blue', label='MIDI: Playing')
        
        ax.scatter(original_floating, [0] * len(original_floating), 
                  alpha=0.6, s=2, color='orange', label='기존: Floating')
        ax.scatter(original_playing, [0] * len(original_playing), 
                  alpha=0.6, s=2, color='green', label='기존: Playing')
        
        ax.set_title(f'{hand_type}손 Floating Detection 비교')
        ax.set_xlabel('시간 (초)')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['기존 방식', 'MIDI 기반'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'floating_comparison_{video_name}.png', dpi=150, bbox_inches='tight')
    
    print(f"   💾 그래프 저장: floating_comparison_{video_name}.png")
    
    # 그래프를 보여주려고 하지만 환경에서 지원되지 않을 수 있음
    try:
        plt.show()
    except:
        print("   ℹ️  GUI 환경이 아니므로 그래프 표시를 건너뜁니다.")
    
    plt.close()


def main():
    """메인 테스트 실행"""
    print("🎵 MIDI 기반 Floating Detection 테스트 (Cache Handlist)")
    print("=" * 60)
    
    # 테스트 데이터 로드
    handlist, video_name, ratio = load_test_data()
    
    if handlist is None:
        return
    
    # 비교 테스트 실행
    midi_results, original_results = run_comparison_test(handlist, video_name, ratio)
    
    # 결과 분석
    merged_df = analyze_differences(midi_results, original_results, video_name)
    
    # 시각화
    visualize_timeline_comparison(merged_df, video_name)
    
    # 결과 저장
    print(f"\n💾 결과 저장...")
    with open(f'midi_floating_results_{video_name}.pkl', 'wb') as f:
        pickle.dump(midi_results, f)
    
    with open(f'original_floating_results_{video_name}.pkl', 'wb') as f:
        pickle.dump(original_results, f)
    
    print(f"   ✅ 저장 완료!")
    
    print(f"\n🎯 테스트 완료!")
    print(f"핵심 개선사항:")
    print(f"✅ MIDI가 연주되는 순간 → 절대 floating 아님")
    print(f"✅ MIDI가 조용한 순간 → 깊이로 정확한 판정")
    print(f"✅ 논리적이고 일관성 있는 floating 감지")


if __name__ == "__main__":
    main() 
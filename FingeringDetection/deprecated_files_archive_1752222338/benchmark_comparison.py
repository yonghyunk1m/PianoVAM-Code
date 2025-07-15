#!/usr/bin/env python3
"""
성능 비교 벤치마크 스크립트
scipy vs PyTorch vs CuPy 깊이 계산 성능 비교
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Mock handlandmark class for testing
class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def generate_test_data(num_samples: int = 1000) -> List[Tuple]:
    """테스트 데이터 생성"""
    np.random.seed(42)
    
    test_cases = []
    for _ in range(num_samples):
        # 랜덤 손 좌표 생성
        w = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        i = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        r = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        
        # Mock 모델 생성
        lhmodel = [MockLandmark(0, 0) for _ in range(21)]
        lhmodel[0] = MockLandmark(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        lhmodel[5] = MockLandmark(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        lhmodel[13] = MockLandmark(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        
        rhmodel = lhmodel  # 동일하게 사용
        ratio = 1.0
        
        test_cases.append((w, i, r, lhmodel, rhmodel, ratio))
    
    return test_cases

def benchmark_scipy(test_cases: List[Tuple], verbose: bool = True) -> Tuple[float, List[float]]:
    """SciPy 기반 계산 벤치마크"""
    from scipy.optimize import fsolve
    
    def landmarkdistance(landmarka, landmarkb, ratio):
        return ((landmarka.x - landmarkb.x) ** 2 + (landmarka.y*ratio - landmarkb.y*ratio) ** 2) ** 0.5 * 2
    
    def system(vars, w, i, r, lhmodel, rhmodel, ratio):
        t, u, v = vars
        eq1 = (t*w[0]-u*i[0])**2+(t*w[1]-u*i[1])**2+(t-u)**2 - landmarkdistance(lhmodel[0],lhmodel[5],ratio)**2
        eq2 = (u*i[0]-v*r[0])**2+(u*i[1]-v*r[1])**2+(u-v)**2 - landmarkdistance(lhmodel[5],lhmodel[13],ratio)**2
        eq3 = (v*r[0]-t*w[0])**2+(v*r[1]-t*w[1])**2+(v-t)**2 - landmarkdistance(lhmodel[13],lhmodel[0],ratio)**2
        return [eq1, eq2, eq3]
    
    def calcdepth_scipy(w, i, r, lhmodel, rhmodel, ratio):
        initial_guess = [1, 1, 1]
        solution = fsolve(system, initial_guess, args=(w, i, r, lhmodel, rhmodel, ratio))
        t, u, v = solution
        return (t + u + v) / 3
    
    if verbose:
        print("? SciPy 벤치마크 시작...")
    
    start_time = time.time()
    results = []
    
    for case in test_cases:
        w, i, r, lhmodel, rhmodel, ratio = case
        result = calcdepth_scipy(w, i, r, lhmodel, rhmodel, ratio)
        results.append(result)
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"? SciPy 완료: {total_time:.3f}초, 평균 {total_time/len(test_cases)*1000:.2f}ms/sample")
    
    return total_time, results

def benchmark_torch(test_cases: List[Tuple], verbose: bool = True) -> Tuple[float, List[float]]:
    """PyTorch 기반 계산 벤치마크"""
    try:
        from torch_depth_calculator import calcdepth_torch, batch_calcdepth_torch
        
        if verbose:
            print("? PyTorch 벤치마크 시작...")
        
        # 단일 계산 벤치마크
        start_time = time.time()
        results = []
        
        for case in test_cases:
            w, i, r, lhmodel, rhmodel, ratio = case
            result = calcdepth_torch(w, i, r, lhmodel, rhmodel, ratio)
            results.append(result)
        
        single_time = time.time() - start_time
        
        # 배치 계산 벤치마크
        start_time = time.time()
        coordinates_list = [(case[0], case[1], case[2]) for case in test_cases]
        lhmodel = test_cases[0][3]
        rhmodel = test_cases[0][4]
        ratio = test_cases[0][5]
        
        batch_results = batch_calcdepth_torch(coordinates_list, lhmodel, rhmodel, ratio)
        batch_time = time.time() - start_time
        
        if verbose:
            print(f"? PyTorch 단일: {single_time:.3f}초, 평균 {single_time/len(test_cases)*1000:.2f}ms/sample")
            print(f"? PyTorch 배치: {batch_time:.3f}초, 평균 {batch_time/len(test_cases)*1000:.2f}ms/sample")
            print(f"? PyTorch 배치 속도 향상: {single_time/batch_time:.1f}x")
        
        return min(single_time, batch_time), batch_results
        
    except ImportError:
        if verbose:
            print("?? PyTorch 모듈을 찾을 수 없습니다.")
        return float('inf'), []

def benchmark_cupy(test_cases: List[Tuple], verbose: bool = True) -> Tuple[float, List[float]]:
    """CuPy 기반 계산 벤치마크"""
    try:
        from cupy_depth_calculator import calcdepth_cupy
        
        if verbose:
            print("? CuPy 벤치마크 시작...")
        
        start_time = time.time()
        results = []
        
        for case in test_cases:
            w, i, r, lhmodel, rhmodel, ratio = case
            result = calcdepth_cupy(w, i, r, lhmodel, rhmodel, ratio)
            results.append(result)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"? CuPy 완료: {total_time:.3f}초, 평균 {total_time/len(test_cases)*1000:.2f}ms/sample")
        
        return total_time, results
        
    except ImportError:
        if verbose:
            print("?? CuPy 모듈을 찾을 수 없습니다.")
        return float('inf'), []

def run_comprehensive_benchmark():
    """종합 벤치마크 실행"""
    print("=" * 60)
    print("? 깊이 계산 성능 벤치마크")
    print("=" * 60)
    
    # 다양한 크기로 테스트
    test_sizes = [100, 500, 1000, 2000]
    results = {
        'sizes': test_sizes,
        'scipy': [],
        'torch': [],
        'cupy': []
    }
    
    for size in test_sizes:
        print(f"\n? 테스트 크기: {size} samples")
        print("-" * 40)
        
        test_cases = generate_test_data(size)
        
        # SciPy 벤치마크
        scipy_time, scipy_results = benchmark_scipy(test_cases)
        results['scipy'].append(scipy_time)
        
        # PyTorch 벤치마크
        torch_time, torch_results = benchmark_torch(test_cases)
        results['torch'].append(torch_time)
        
        # CuPy 벤치마크
        cupy_time, cupy_results = benchmark_cupy(test_cases)
        results['cupy'].append(cupy_time)
        
        # 결과 검증 (첫 번째 케이스)
        if len(scipy_results) > 0 and len(torch_results) > 0:
            diff = abs(scipy_results[0] - torch_results[0])
            print(f"? 결과 검증: SciPy vs PyTorch 차이 = {diff:.6f}")
    
    # 결과 시각화
    plot_benchmark_results(results)
    
    # 최종 결과 요약
    print("\n" + "=" * 60)
    print("? 최종 성능 요약 (1000 samples 기준)")
    print("=" * 60)
    
    idx = test_sizes.index(1000) if 1000 in test_sizes else -1
    if idx >= 0:
        scipy_time = results['scipy'][idx]
        torch_time = results['torch'][idx]
        cupy_time = results['cupy'][idx]
        
        print(f"SciPy:   {scipy_time:.3f}초")
        if torch_time != float('inf'):
            speedup = scipy_time / torch_time
            print(f"PyTorch: {torch_time:.3f}초 ({speedup:.1f}x 가속)")
        if cupy_time != float('inf'):
            speedup = scipy_time / cupy_time
            print(f"CuPy:    {cupy_time:.3f}초 ({speedup:.1f}x 가속)")
    
    return results

def plot_benchmark_results(results):
    """벤치마크 결과를 그래프로 표시"""
    plt.figure(figsize=(12, 8))
    
    sizes = results['sizes']
    
    # 유효한 결과만 플롯
    if any(t != float('inf') for t in results['scipy']):
        plt.plot(sizes, results['scipy'], 'o-', label='SciPy (baseline)', linewidth=2, markersize=8)
    
    if any(t != float('inf') for t in results['torch']):
        plt.plot(sizes, results['torch'], 's-', label='PyTorch (GPU)', linewidth=2, markersize=8)
    
    if any(t != float('inf') for t in results['cupy']):
        plt.plot(sizes, results['cupy'], '^-', label='CuPy (GPU)', linewidth=2, markersize=8)
    
    plt.xlabel('테스트 케이스 수', fontsize=12)
    plt.ylabel('실행 시간 (초)', fontsize=12)
    plt.title('깊이 계산 알고리즘 성능 비교', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 로그 스케일로 표시
    
    # 그래프 저장
    plt.tight_layout()
    plt.savefig('depth_calculation_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("? 벤치마크 그래프가 'depth_calculation_benchmark.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_comprehensive_benchmark() 
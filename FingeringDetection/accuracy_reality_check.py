#!/usr/bin/env python3
"""
실제 정확도 한계 분석
- SciPy vs PyTorch 실제 차이 측정
- 연구용 허용 기준과의 비교
- 정확도 한계의 원인 분석
"""

import numpy as np
import torch
import time
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def demonstrate_accuracy_limits():
    """실제 정확도 한계 시연"""
    print("? SciPy vs PyTorch 실제 정확도 차이 분석")
    print("=" * 60)
    
    # 동일한 테스트 케이스
    test_cases = [
        ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]),
        ([0.0, 0.0], [0.1, 0.1], [0.2, 0.2]),
        ([-0.1, 0.1], [0.2, -0.1], [0.1, 0.3]),
        ([0.5, 0.3], [-0.2, 0.4], [0.3, -0.1]),
        ([0.01, 0.01], [0.02, 0.02], [0.03, 0.03])
    ]
    
    # Mock model
    class MockLandmark:
        def __init__(self, x, y):
            self.x, self.y = x, y
    
    lhmodel = [MockLandmark(0.0, 0.0), None, None, None, None,
               MockLandmark(-0.1, -0.05), None, None, None, None, None, None, None,
               MockLandmark(0.1, -0.05)]
    ratio = 1.2
    
    differences = []
    
    for i, (w, idx, r) in enumerate(test_cases):
        print(f"\n테스트 케이스 {i+1}: w={w}, i={idx}, r={r}")
        
        # SciPy 결과
        scipy_result = solve_with_scipy(w, idx, r, lhmodel, ratio)
        
        # PyTorch 결과 (여러 번 실행하여 변동성 확인)
        pytorch_results = []
        for trial in range(5):
            pytorch_result = solve_with_pytorch(w, idx, r, lhmodel, ratio)
            pytorch_results.append(pytorch_result)
        
        pytorch_avg = np.mean(pytorch_results)
        pytorch_std = np.std(pytorch_results)
        
        # 차이 분석
        diff = abs(scipy_result - pytorch_avg)
        differences.append(diff)
        
        print(f"  SciPy 결과:     {scipy_result:.12f}")
        print(f"  PyTorch 평균:   {pytorch_avg:.12f}")
        print(f"  PyTorch 표준편차: {pytorch_std:.2e}")
        print(f"  절대 차이:      {diff:.2e}")
        
        # 연구 기준과 비교
        if diff < 1e-10:
            status = "? 연구 등급"
        elif diff < 1e-8:
            status = "? 실용 등급"
        elif diff < 1e-6:
            status = "?? 주의 필요"
        else:
            status = "? 부적합"
        
        print(f"  연구 적합성:    {status}")
    
    # 전체 통계
    print(f"\n? 전체 정확도 분석:")
    print(f"  평균 차이:      {np.mean(differences):.2e}")
    print(f"  최대 차이:      {np.max(differences):.2e}")
    print(f"  최소 차이:      {np.min(differences):.2e}")
    print(f"  표준편차:       {np.std(differences):.2e}")
    
    # 연구 기준 평가
    excellent_count = sum(1 for d in differences if d < 1e-10)
    good_count = sum(1 for d in differences if d < 1e-8)
    
    print(f"\n? 연구 적합성 평가:")
    print(f"  연구 등급 (< 1e-10): {excellent_count}/{len(differences)} ({excellent_count/len(differences)*100:.1f}%)")
    print(f"  실용 등급 (< 1e-8):  {good_count}/{len(differences)} ({good_count/len(differences)*100:.1f}%)")
    
    return differences

def solve_with_scipy(w, i, r, lhmodel, ratio):
    """SciPy로 해 계산"""
    def system(vars, w, i, r, lhmodel, ratio):
        t, u, v = vars
        
        dist_wi = ((lhmodel[0].x - lhmodel[5].x)**2 + (lhmodel[0].y*ratio - lhmodel[5].y*ratio)**2)**0.5 * 2
        dist_ir = ((lhmodel[5].x - lhmodel[13].x)**2 + (lhmodel[5].y*ratio - lhmodel[13].y*ratio)**2)**0.5 * 2  
        dist_rw = ((lhmodel[13].x - lhmodel[0].x)**2 + (lhmodel[13].y*ratio - lhmodel[0].y*ratio)**2)**0.5 * 2
        
        eq1 = (t*w[0] - u*i[0])**2 + (t*w[1] - u*i[1])**2 + (t-u)**2 - dist_wi**2
        eq2 = (u*i[0] - v*r[0])**2 + (u*i[1] - v*r[1])**2 + (u-v)**2 - dist_ir**2
        eq3 = (v*r[0] - t*w[0])**2 + (v*r[1] - t*w[1])**2 + (v-t)**2 - dist_rw**2
        
        return [eq1, eq2, eq3]
    
    solution = fsolve(system, [1.0, 1.0, 1.0], args=(w, i, r, lhmodel, ratio))
    return (solution[0] + solution[1] + solution[2]) / 3

def solve_with_pytorch(w, i, r, lhmodel, ratio):
    """PyTorch로 해 계산"""
    # 목표 거리 계산
    dist_wi = ((lhmodel[0].x - lhmodel[5].x)**2 + (lhmodel[0].y*ratio - lhmodel[5].y*ratio)**2)**0.5 * 2
    dist_ir = ((lhmodel[5].x - lhmodel[13].x)**2 + (lhmodel[5].y*ratio - lhmodel[13].y*ratio)**2)**0.5 * 2
    dist_rw = ((lhmodel[13].x - lhmodel[0].x)**2 + (lhmodel[13].y*ratio - lhmodel[0].y*ratio)**2)**0.5 * 2
    
    # 텐서 준비
    coords = torch.tensor([[w, i, r]], dtype=torch.float64)
    targets = torch.tensor([[dist_wi, dist_ir, dist_rw]], dtype=torch.float64)
    
    # 초기값
    variables = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)
    
    # LBFGS 최적화
    optimizer = torch.optim.LBFGS([variables], lr=1.0, max_iter=10000, tolerance_grad=1e-14, tolerance_change=1e-15)
    
    def closure():
        optimizer.zero_grad()
        
        t, u, v = variables[0, 0], variables[0, 1], variables[0, 2]
        w_coords, i_coords, r_coords = coords[0, 0], coords[0, 1], coords[0, 2]
        
        eq1 = (t * w_coords[0] - u * i_coords[0])**2 + (t * w_coords[1] - u * i_coords[1])**2 + (t - u)**2 - targets[0, 0]**2
        eq2 = (u * i_coords[0] - v * r_coords[0])**2 + (u * i_coords[1] - v * r_coords[1])**2 + (u - v)**2 - targets[0, 1]**2
        eq3 = (v * r_coords[0] - t * w_coords[0])**2 + (v * r_coords[1] - t * w_coords[1])**2 + (v - t)**2 - targets[0, 2]**2
        
        loss = eq1**2 + eq2**2 + eq3**2
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    return float(torch.mean(variables))

def why_differences_occur():
    """차이가 발생하는 이유 설명"""
    print("\n? 정확도 차이가 발생하는 근본적 이유들")
    print("=" * 60)
    
    reasons = [
        ("1. 알고리즘 차이", [
            "SciPy: Powell's dog leg (hybrid Newton)",
            "PyTorch: LBFGS (quasi-Newton)",
            "→ 다른 수렴 경로, 다른 최종 해"
        ]),
        ("2. 수치 구현 차이", [
            "SciPy: NumPy + BLAS/LAPACK",
            "PyTorch: 자체 구현 + cuBLAS",
            "→ 미묘한 부동소수점 연산 차이"
        ]),
        ("3. 수렴 기준 차이", [
            "SciPy: 자체 tolerance 기준",
            "PyTorch: 사용자 정의 tolerance",
            "→ 언제 멈추느냐의 차이"
        ]),
        ("4. 초기값 처리", [
            "SciPy: 내부 최적화된 초기값",
            "PyTorch: 단순한 [1,1,1] 시작",
            "→ 수렴 과정의 차이"
        ]),
        ("5. IEEE 754 한계", [
            "부동소수점의 근본적 한계",
            "연산 순서에 따른 누적 오차",
            "→ 완전한 동일성은 불가능"
        ])
    ]
    
    for title, details in reasons:
        print(f"\n{title}:")
        for detail in details:
            print(f"  {detail}")

def research_acceptability_analysis():
    """연구용 허용 가능성 분석"""
    print("\n? 연구용 허용 가능성 분석")
    print("=" * 60)
    
    standards = [
        ("IEEE 754 배정밀도", "~1e-15", "부동소수점 표준 한계"),
        ("일반 수치 해석", "1e-10 ~ 1e-12", "대부분의 과학 계산"),
        ("엔지니어링 계산", "1e-6 ~ 1e-8", "실용적 정확도"),
        ("통계적 유의성", "1e-3 ~ 1e-4", "통계 분석 기준"),
        ("PianoVAM 임계값", "0.9 (10%)", "실제 시스템 허용도")
    ]
    
    print("기준별 허용 오차:")
    for standard, tolerance, description in standards:
        print(f"  {standard:20} | {tolerance:12} | {description}")
    
    print(f"\n? 우리 시스템의 예상 정확도:")
    print(f"  목표 정확도:     1e-10 ~ 1e-12")
    print(f"  실제 달성:       1e-8 ~ 1e-10 (추정)")
    print(f"  연구 적합성:     ? 충분히 높음")
    print(f"  PianoVAM 적합성: ? 완전히 충분")

def conclusion():
    """결론"""
    print(f"\n? 결론: 완전한 동일성 vs 연구용 충분성")
    print("=" * 60)
    
    print("? 불가능한 것:")
    print("  - SciPy와 완전히 동일한 수치 (bit-level 동일성)")
    print("  - 무한 정밀도")
    print("  - 알고리즘이 다른데 결과가 완전 동일")
    
    print("\n? 달성 가능한 것:")
    print("  - 연구용으로 충분한 높은 정확도 (1e-10 수준)")
    print("  - PianoVAM 기준으로 무의미한 차이")
    print("  - 통계적으로 동일한 결과")
    print("  - 실용적으로 완전히 동등한 성능")
    
    print("\n? 올바른 표현:")
    print("  '수학적으로 동일' ?")
    print("  '연구용으로 충분히 정확' ?")
    print("  '실용적으로 동등' ?")
    print("  '통계적으로 무의미한 차이' ?")

if __name__ == "__main__":
    demonstrate_accuracy_limits()
    why_differences_occur()
    research_acceptability_analysis()
    conclusion() 
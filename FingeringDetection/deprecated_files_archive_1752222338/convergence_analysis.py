#!/usr/bin/env python3
"""
수렴 차이 원인 분석 및 해결 방안
"""

import torch
import torch.optim as optim
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def analyze_convergence_issue():
    """수렴 차이 원인 분석"""
    print("🚨 수렴 차이 원인 분석")
    print("=" * 50)
    
    # 동일한 테스트 케이스
    w = [0.3, 0.2]
    i = [0.1, 0.4] 
    r = [-0.2, 0.1]
    
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    lhmodel = [MockLandmark(0, 0) for _ in range(21)]
    lhmodel[0] = MockLandmark(0.0, 0.0)
    lhmodel[5] = MockLandmark(0.2, 0.3)
    lhmodel[13] = MockLandmark(-0.1, 0.2)
    ratio = 1.0
    
    def landmarkdistance(landmark_a, landmark_b, ratio):
        return ((landmark_a.x - landmark_b.x) ** 2 + 
                (landmark_a.y * ratio - landmark_b.y * ratio) ** 2) ** 0.5 * 2
    
    # 목표 거리 계산
    target_WI = landmarkdistance(lhmodel[0], lhmodel[5], ratio)
    target_IR = landmarkdistance(lhmodel[5], lhmodel[13], ratio)
    target_RW = landmarkdistance(lhmodel[13], lhmodel[0], ratio)
    
    print(f"목표 거리:")
    print(f"  WI: {target_WI:.6f}")
    print(f"  IR: {target_IR:.6f}")
    print(f"  RW: {target_RW:.6f}")
    
    # SciPy 해
    def system(vars):
        t, u, v = vars
        eq1 = (t*w[0] - u*i[0])**2 + (t*w[1] - u*i[1])**2 + (t - u)**2 - target_WI**2
        eq2 = (u*i[0] - v*r[0])**2 + (u*i[1] - v*r[1])**2 + (u - v)**2 - target_IR**2  
        eq3 = (v*r[0] - t*w[0])**2 + (v*r[1] - t*w[1])**2 + (v - t)**2 - target_RW**2
        return [eq1, eq2, eq3]
    
    scipy_solution = fsolve(system, [1.0, 1.0, 1.0])
    scipy_depth = scipy_solution.mean()
    
    print(f"\nSciPy 해:")
    print(f"  t, u, v: {scipy_solution}")
    print(f"  깊이: {scipy_depth:.6f}")
    
    # 방정식 잔차 확인
    residual = system(scipy_solution)
    print(f"  방정식 잔차: {residual}")
    print(f"  잔차 크기: {np.linalg.norm(residual):.2e}")
    
    return scipy_solution, target_WI, target_IR, target_RW, w, i, r

def improved_pytorch_approach(scipy_solution, target_WI, target_IR, target_RW, w, i, r):
    """개선된 PyTorch 접근법"""
    print(f"\n🔧 개선된 PyTorch 접근법")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    target_distances = torch.tensor([target_WI, target_IR, target_RW], device=device)
    w_tensor = torch.tensor(w, device=device)
    i_tensor = torch.tensor(i, device=device)  
    r_tensor = torch.tensor(r, device=device)
    
    # 개선 사항들을 테스트
    configs = [
        {"name": "기본 설정", "lr": 0.01, "max_iter": 100, "tol": 1e-6},
        {"name": "낮은 학습률", "lr": 0.001, "max_iter": 200, "tol": 1e-8},
        {"name": "매우 낮은 학습률", "lr": 0.0001, "max_iter": 500, "tol": 1e-10},
        {"name": "LBFGS 옵티마이저", "optimizer": "LBFGS", "lr": 1.0, "max_iter": 100, "tol": 1e-10}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 테스트: {config['name']}")
        
        # SciPy 해에 가까운 초기값 사용
        tuv = torch.tensor(scipy_solution, device=device, requires_grad=True, dtype=torch.float32)
        
        if config.get("optimizer") == "LBFGS":
            optimizer = optim.LBFGS([tuv], lr=config["lr"])
        else:
            optimizer = optim.Adam([tuv], lr=config["lr"])
        
        loss_history = []
        
        for iteration in range(config["max_iter"]):
            if config.get("optimizer") == "LBFGS":
                def closure():
                    optimizer.zero_grad()
                    loss = compute_loss(tuv, w_tensor, i_tensor, r_tensor, target_distances)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss = compute_loss(tuv, w_tensor, i_tensor, r_tensor, target_distances)
            else:
                optimizer.zero_grad()
                loss = compute_loss(tuv, w_tensor, i_tensor, r_tensor, target_distances)
                loss.backward()
                optimizer.step()
            
            loss_history.append(loss.item())
            
            if loss.item() < config["tol"]:
                print(f"  수렴 완료 (반복 {iteration})")
                break
        
        final_tuv = tuv.detach().cpu().numpy()
        final_depth = final_tuv.mean()
        final_loss = loss_history[-1] if loss_history else float('inf')
        
        depth_diff = abs(final_depth - scipy_solution.mean())
        solution_diff = np.linalg.norm(final_tuv - scipy_solution)
        
        print(f"  최종 해: {final_tuv}")
        print(f"  최종 깊이: {final_depth:.6f}")
        print(f"  최종 Loss: {final_loss:.2e}")
        print(f"  SciPy와 깊이 차이: {depth_diff:.2e}")
        print(f"  SciPy와 해 차이: {solution_diff:.2e}")
        
        results.append({
            "config": config["name"],
            "final_tuv": final_tuv,
            "final_depth": final_depth,
            "final_loss": final_loss,
            "depth_diff": depth_diff,
            "solution_diff": solution_diff,
            "loss_history": loss_history
        })
    
    return results

def compute_loss(tuv, w_tensor, i_tensor, r_tensor, target_distances):
    """Loss 함수 계산"""
    t, u, v = tuv[0], tuv[1], tuv[2]
    
    # 3D 좌표 계산
    w_3d = torch.stack([t * w_tensor[0], t * w_tensor[1], t])
    i_3d = torch.stack([u * i_tensor[0], u * i_tensor[1], u])
    r_3d = torch.stack([v * r_tensor[0], v * r_tensor[1], v])
    
    # 거리 계산
    dist_wi = torch.norm(w_3d - i_3d)
    dist_ir = torch.norm(i_3d - r_3d)
    dist_rw = torch.norm(r_3d - w_3d)
    
    # Loss function
    loss = ((dist_wi - target_distances[0])**2 + 
           (dist_ir - target_distances[1])**2 + 
           (dist_rw - target_distances[2])**2)
    
    return loss

def plot_comparison_results(results):
    """결과 비교 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss 히스토리
    for result in results:
        if len(result["loss_history"]) > 0:
            ax1.plot(result["loss_history"], label=result["config"])
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Convergence Comparison')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 정확도 비교
    configs = [r["config"] for r in results]
    depth_diffs = [r["depth_diff"] for r in results]
    solution_diffs = [r["solution_diff"] for r in results]
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax2.bar(x - width/2, depth_diffs, width, label='Depth Difference', alpha=0.7)
    ax2.bar(x + width/2, solution_diffs, width, label='Solution Difference', alpha=0.7)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Difference from SciPy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📈 분석 결과가 'convergence_analysis.png'로 저장되었습니다.")

def recommend_solution():
    """추천 해결책"""
    print(f"\n💡 추천 해결책")
    print("=" * 50)
    
    recommendations = [
        "1. 🎯 **더 엄격한 수렴 기준 사용**",
        "   - tolerance를 1e-10 이하로 설정",
        "   - max_iterations를 500 이상으로 증가",
        "",
        "2. 🔧 **LBFGS 옵티마이저 사용**",
        "   - 2차 최적화 방법으로 더 정확한 수렴",
        "   - 메모리 사용량은 늘지만 정확도 향상",
        "",
        "3. 🎲 **초기값 개선**",
        "   - SciPy 결과에 가까운 초기값 사용",
        "   - 또는 여러 초기값으로 테스트 후 최적 선택",
        "",
        "4. ✅ **결과 검증 추가**",
        "   - 최종 Loss가 허용 기준 이하인지 확인",
        "   - SciPy 결과와 비교하여 차이가 크면 경고",
        "",
        "5. 🔄 **하이브리드 접근**",
        "   - 배치 처리는 PyTorch로 빠르게",
        "   - 중요한 계산은 SciPy로 검증"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    # 문제 분석
    scipy_solution, target_WI, target_IR, target_RW, w, i, r = analyze_convergence_issue()
    
    # 개선된 PyTorch 테스트
    results = improved_pytorch_approach(scipy_solution, target_WI, target_IR, target_RW, w, i, r)
    
    # 결과 시각화
    plot_comparison_results(results)
    
    # 해결책 제시
    recommend_solution()
    
    # 최고 성능 결과 출력
    best_result = min(results, key=lambda x: x["depth_diff"])
    print(f"\n🏆 최고 성능:")
    print(f"  설정: {best_result['config']}")
    print(f"  깊이 차이: {best_result['depth_diff']:.2e}")
    print(f"  해 차이: {best_result['solution_diff']:.2e}")
    
    if best_result['depth_diff'] < 1e-6:
        print("  ✅ 허용 가능한 정확도!")
    else:
        print("  ⚠️ 여전히 개선 필요!") 
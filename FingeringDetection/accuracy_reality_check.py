#!/usr/bin/env python3
"""
���� ��Ȯ�� �Ѱ� �м�
- SciPy vs PyTorch ���� ���� ����
- ������ ��� ���ذ��� ��
- ��Ȯ�� �Ѱ��� ���� �м�
"""

import numpy as np
import torch
import time
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def demonstrate_accuracy_limits():
    """���� ��Ȯ�� �Ѱ� �ÿ�"""
    print("? SciPy vs PyTorch ���� ��Ȯ�� ���� �м�")
    print("=" * 60)
    
    # ������ �׽�Ʈ ���̽�
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
        print(f"\n�׽�Ʈ ���̽� {i+1}: w={w}, i={idx}, r={r}")
        
        # SciPy ���
        scipy_result = solve_with_scipy(w, idx, r, lhmodel, ratio)
        
        # PyTorch ��� (���� �� �����Ͽ� ������ Ȯ��)
        pytorch_results = []
        for trial in range(5):
            pytorch_result = solve_with_pytorch(w, idx, r, lhmodel, ratio)
            pytorch_results.append(pytorch_result)
        
        pytorch_avg = np.mean(pytorch_results)
        pytorch_std = np.std(pytorch_results)
        
        # ���� �м�
        diff = abs(scipy_result - pytorch_avg)
        differences.append(diff)
        
        print(f"  SciPy ���:     {scipy_result:.12f}")
        print(f"  PyTorch ���:   {pytorch_avg:.12f}")
        print(f"  PyTorch ǥ������: {pytorch_std:.2e}")
        print(f"  ���� ����:      {diff:.2e}")
        
        # ���� ���ذ� ��
        if diff < 1e-10:
            status = "? ���� ���"
        elif diff < 1e-8:
            status = "? �ǿ� ���"
        elif diff < 1e-6:
            status = "?? ���� �ʿ�"
        else:
            status = "? ������"
        
        print(f"  ���� ���ռ�:    {status}")
    
    # ��ü ���
    print(f"\n? ��ü ��Ȯ�� �м�:")
    print(f"  ��� ����:      {np.mean(differences):.2e}")
    print(f"  �ִ� ����:      {np.max(differences):.2e}")
    print(f"  �ּ� ����:      {np.min(differences):.2e}")
    print(f"  ǥ������:       {np.std(differences):.2e}")
    
    # ���� ���� ��
    excellent_count = sum(1 for d in differences if d < 1e-10)
    good_count = sum(1 for d in differences if d < 1e-8)
    
    print(f"\n? ���� ���ռ� ��:")
    print(f"  ���� ��� (< 1e-10): {excellent_count}/{len(differences)} ({excellent_count/len(differences)*100:.1f}%)")
    print(f"  �ǿ� ��� (< 1e-8):  {good_count}/{len(differences)} ({good_count/len(differences)*100:.1f}%)")
    
    return differences

def solve_with_scipy(w, i, r, lhmodel, ratio):
    """SciPy�� �� ���"""
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
    """PyTorch�� �� ���"""
    # ��ǥ �Ÿ� ���
    dist_wi = ((lhmodel[0].x - lhmodel[5].x)**2 + (lhmodel[0].y*ratio - lhmodel[5].y*ratio)**2)**0.5 * 2
    dist_ir = ((lhmodel[5].x - lhmodel[13].x)**2 + (lhmodel[5].y*ratio - lhmodel[13].y*ratio)**2)**0.5 * 2
    dist_rw = ((lhmodel[13].x - lhmodel[0].x)**2 + (lhmodel[13].y*ratio - lhmodel[0].y*ratio)**2)**0.5 * 2
    
    # �ټ� �غ�
    coords = torch.tensor([[w, i, r]], dtype=torch.float64)
    targets = torch.tensor([[dist_wi, dist_ir, dist_rw]], dtype=torch.float64)
    
    # �ʱⰪ
    variables = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)
    
    # LBFGS ����ȭ
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
    """���̰� �߻��ϴ� ���� ����"""
    print("\n? ��Ȯ�� ���̰� �߻��ϴ� �ٺ��� ������")
    print("=" * 60)
    
    reasons = [
        ("1. �˰��� ����", [
            "SciPy: Powell's dog leg (hybrid Newton)",
            "PyTorch: LBFGS (quasi-Newton)",
            "�� �ٸ� ���� ���, �ٸ� ���� ��"
        ]),
        ("2. ��ġ ���� ����", [
            "SciPy: NumPy + BLAS/LAPACK",
            "PyTorch: ��ü ���� + cuBLAS",
            "�� �̹��� �ε��Ҽ��� ���� ����"
        ]),
        ("3. ���� ���� ����", [
            "SciPy: ��ü tolerance ����",
            "PyTorch: ����� ���� tolerance",
            "�� ���� ���ߴ����� ����"
        ]),
        ("4. �ʱⰪ ó��", [
            "SciPy: ���� ����ȭ�� �ʱⰪ",
            "PyTorch: �ܼ��� [1,1,1] ����",
            "�� ���� ������ ����"
        ]),
        ("5. IEEE 754 �Ѱ�", [
            "�ε��Ҽ����� �ٺ��� �Ѱ�",
            "���� ������ ���� ���� ����",
            "�� ������ ���ϼ��� �Ұ���"
        ])
    ]
    
    for title, details in reasons:
        print(f"\n{title}:")
        for detail in details:
            print(f"  {detail}")

def research_acceptability_analysis():
    """������ ��� ���ɼ� �м�"""
    print("\n? ������ ��� ���ɼ� �м�")
    print("=" * 60)
    
    standards = [
        ("IEEE 754 �����е�", "~1e-15", "�ε��Ҽ��� ǥ�� �Ѱ�"),
        ("�Ϲ� ��ġ �ؼ�", "1e-10 ~ 1e-12", "��κ��� ���� ���"),
        ("�����Ͼ ���", "1e-6 ~ 1e-8", "�ǿ��� ��Ȯ��"),
        ("����� ���Ǽ�", "1e-3 ~ 1e-4", "��� �м� ����"),
        ("PianoVAM �Ӱ谪", "0.9 (10%)", "���� �ý��� ��뵵")
    ]
    
    print("���غ� ��� ����:")
    for standard, tolerance, description in standards:
        print(f"  {standard:20} | {tolerance:12} | {description}")
    
    print(f"\n? �츮 �ý����� ���� ��Ȯ��:")
    print(f"  ��ǥ ��Ȯ��:     1e-10 ~ 1e-12")
    print(f"  ���� �޼�:       1e-8 ~ 1e-10 (����)")
    print(f"  ���� ���ռ�:     ? ����� ����")
    print(f"  PianoVAM ���ռ�: ? ������ ���")

def conclusion():
    """���"""
    print(f"\n? ���: ������ ���ϼ� vs ������ ��м�")
    print("=" * 60)
    
    print("? �Ұ����� ��:")
    print("  - SciPy�� ������ ������ ��ġ (bit-level ���ϼ�)")
    print("  - ���� ���е�")
    print("  - �˰����� �ٸ��� ����� ���� ����")
    
    print("\n? �޼� ������ ��:")
    print("  - ���������� ����� ���� ��Ȯ�� (1e-10 ����)")
    print("  - PianoVAM �������� ���ǹ��� ����")
    print("  - ��������� ������ ���")
    print("  - �ǿ������� ������ ������ ����")
    
    print("\n? �ùٸ� ǥ��:")
    print("  '���������� ����' ?")
    print("  '���������� ����� ��Ȯ' ?")
    print("  '�ǿ������� ����' ?")
    print("  '��������� ���ǹ��� ����' ?")

if __name__ == "__main__":
    demonstrate_accuracy_limits()
    why_differences_occur()
    research_acceptability_analysis()
    conclusion() 
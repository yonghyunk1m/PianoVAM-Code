#!/usr/bin/env python3
"""
���� �� ��ġ��ũ ��ũ��Ʈ
scipy vs PyTorch vs CuPy ���� ��� ���� ��
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
    """�׽�Ʈ ������ ����"""
    np.random.seed(42)
    
    test_cases = []
    for _ in range(num_samples):
        # ���� �� ��ǥ ����
        w = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        i = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        r = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        
        # Mock �� ����
        lhmodel = [MockLandmark(0, 0) for _ in range(21)]
        lhmodel[0] = MockLandmark(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        lhmodel[5] = MockLandmark(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        lhmodel[13] = MockLandmark(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        
        rhmodel = lhmodel  # �����ϰ� ���
        ratio = 1.0
        
        test_cases.append((w, i, r, lhmodel, rhmodel, ratio))
    
    return test_cases

def benchmark_scipy(test_cases: List[Tuple], verbose: bool = True) -> Tuple[float, List[float]]:
    """SciPy ��� ��� ��ġ��ũ"""
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
        print("? SciPy ��ġ��ũ ����...")
    
    start_time = time.time()
    results = []
    
    for case in test_cases:
        w, i, r, lhmodel, rhmodel, ratio = case
        result = calcdepth_scipy(w, i, r, lhmodel, rhmodel, ratio)
        results.append(result)
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"? SciPy �Ϸ�: {total_time:.3f}��, ��� {total_time/len(test_cases)*1000:.2f}ms/sample")
    
    return total_time, results

def benchmark_torch(test_cases: List[Tuple], verbose: bool = True) -> Tuple[float, List[float]]:
    """PyTorch ��� ��� ��ġ��ũ"""
    try:
        from torch_depth_calculator import calcdepth_torch, batch_calcdepth_torch
        
        if verbose:
            print("? PyTorch ��ġ��ũ ����...")
        
        # ���� ��� ��ġ��ũ
        start_time = time.time()
        results = []
        
        for case in test_cases:
            w, i, r, lhmodel, rhmodel, ratio = case
            result = calcdepth_torch(w, i, r, lhmodel, rhmodel, ratio)
            results.append(result)
        
        single_time = time.time() - start_time
        
        # ��ġ ��� ��ġ��ũ
        start_time = time.time()
        coordinates_list = [(case[0], case[1], case[2]) for case in test_cases]
        lhmodel = test_cases[0][3]
        rhmodel = test_cases[0][4]
        ratio = test_cases[0][5]
        
        batch_results = batch_calcdepth_torch(coordinates_list, lhmodel, rhmodel, ratio)
        batch_time = time.time() - start_time
        
        if verbose:
            print(f"? PyTorch ����: {single_time:.3f}��, ��� {single_time/len(test_cases)*1000:.2f}ms/sample")
            print(f"? PyTorch ��ġ: {batch_time:.3f}��, ��� {batch_time/len(test_cases)*1000:.2f}ms/sample")
            print(f"? PyTorch ��ġ �ӵ� ���: {single_time/batch_time:.1f}x")
        
        return min(single_time, batch_time), batch_results
        
    except ImportError:
        if verbose:
            print("?? PyTorch ����� ã�� �� �����ϴ�.")
        return float('inf'), []

def benchmark_cupy(test_cases: List[Tuple], verbose: bool = True) -> Tuple[float, List[float]]:
    """CuPy ��� ��� ��ġ��ũ"""
    try:
        from cupy_depth_calculator import calcdepth_cupy
        
        if verbose:
            print("? CuPy ��ġ��ũ ����...")
        
        start_time = time.time()
        results = []
        
        for case in test_cases:
            w, i, r, lhmodel, rhmodel, ratio = case
            result = calcdepth_cupy(w, i, r, lhmodel, rhmodel, ratio)
            results.append(result)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"? CuPy �Ϸ�: {total_time:.3f}��, ��� {total_time/len(test_cases)*1000:.2f}ms/sample")
        
        return total_time, results
        
    except ImportError:
        if verbose:
            print("?? CuPy ����� ã�� �� �����ϴ�.")
        return float('inf'), []

def run_comprehensive_benchmark():
    """���� ��ġ��ũ ����"""
    print("=" * 60)
    print("? ���� ��� ���� ��ġ��ũ")
    print("=" * 60)
    
    # �پ��� ũ��� �׽�Ʈ
    test_sizes = [100, 500, 1000, 2000]
    results = {
        'sizes': test_sizes,
        'scipy': [],
        'torch': [],
        'cupy': []
    }
    
    for size in test_sizes:
        print(f"\n? �׽�Ʈ ũ��: {size} samples")
        print("-" * 40)
        
        test_cases = generate_test_data(size)
        
        # SciPy ��ġ��ũ
        scipy_time, scipy_results = benchmark_scipy(test_cases)
        results['scipy'].append(scipy_time)
        
        # PyTorch ��ġ��ũ
        torch_time, torch_results = benchmark_torch(test_cases)
        results['torch'].append(torch_time)
        
        # CuPy ��ġ��ũ
        cupy_time, cupy_results = benchmark_cupy(test_cases)
        results['cupy'].append(cupy_time)
        
        # ��� ���� (ù ��° ���̽�)
        if len(scipy_results) > 0 and len(torch_results) > 0:
            diff = abs(scipy_results[0] - torch_results[0])
            print(f"? ��� ����: SciPy vs PyTorch ���� = {diff:.6f}")
    
    # ��� �ð�ȭ
    plot_benchmark_results(results)
    
    # ���� ��� ���
    print("\n" + "=" * 60)
    print("? ���� ���� ��� (1000 samples ����)")
    print("=" * 60)
    
    idx = test_sizes.index(1000) if 1000 in test_sizes else -1
    if idx >= 0:
        scipy_time = results['scipy'][idx]
        torch_time = results['torch'][idx]
        cupy_time = results['cupy'][idx]
        
        print(f"SciPy:   {scipy_time:.3f}��")
        if torch_time != float('inf'):
            speedup = scipy_time / torch_time
            print(f"PyTorch: {torch_time:.3f}�� ({speedup:.1f}x ����)")
        if cupy_time != float('inf'):
            speedup = scipy_time / cupy_time
            print(f"CuPy:    {cupy_time:.3f}�� ({speedup:.1f}x ����)")
    
    return results

def plot_benchmark_results(results):
    """��ġ��ũ ����� �׷����� ǥ��"""
    plt.figure(figsize=(12, 8))
    
    sizes = results['sizes']
    
    # ��ȿ�� ����� �÷�
    if any(t != float('inf') for t in results['scipy']):
        plt.plot(sizes, results['scipy'], 'o-', label='SciPy (baseline)', linewidth=2, markersize=8)
    
    if any(t != float('inf') for t in results['torch']):
        plt.plot(sizes, results['torch'], 's-', label='PyTorch (GPU)', linewidth=2, markersize=8)
    
    if any(t != float('inf') for t in results['cupy']):
        plt.plot(sizes, results['cupy'], '^-', label='CuPy (GPU)', linewidth=2, markersize=8)
    
    plt.xlabel('�׽�Ʈ ���̽� ��', fontsize=12)
    plt.ylabel('���� �ð� (��)', fontsize=12)
    plt.title('���� ��� �˰��� ���� ��', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # �α� �����Ϸ� ǥ��
    
    # �׷��� ����
    plt.tight_layout()
    plt.savefig('depth_calculation_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("? ��ġ��ũ �׷����� 'depth_calculation_benchmark.png'�� ����Ǿ����ϴ�.")

if __name__ == "__main__":
    run_comprehensive_benchmark() 
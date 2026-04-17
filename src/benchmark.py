import time
import numpy as np


def measure_latency(predict_fn, X, n_trials=100):
    """
    추론 지연시간 측정
    ARM baseline vs FPGA 비교에 사용
    """
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        predict_fn(X)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms 단위

    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
    }


def print_benchmark(name, result):
    print(f"\n[{name}]")
    print(f"  평균 지연시간: {result['mean_ms']:.3f} ms")
    print(f"  표준편차:     {result['std_ms']:.3f} ms")
    print(f"  최소:         {result['min_ms']:.3f} ms")
    print(f"  최대:         {result['max_ms']:.3f} ms")

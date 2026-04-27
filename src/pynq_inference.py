import pickle
import numpy as np
import time

def measure_latency(model, X, n_trials=100):
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        model.predict(X)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
    }

print("모델 로드 중...")
with open('/home/xilinx/model.pkl', 'rb') as f:
    model = pickle.load(f)

# 96개 특징 (MAV, RMS, ZC, WL, MNF, MDF x 16채널)
X_dummy = np.random.randn(1, 96).astype(np.float32)

print("워밍업 중...")
for _ in range(10):
    model.predict(X_dummy)

print("지연시간 측정 중 (100회)...")
result = measure_latency(model, X_dummy)

print(f"\n[PYNQ ARM baseline]")
print(f"  평균 지연시간: {result['mean_ms']:.3f} ms")
print(f"  표준편차:     {result['std_ms']:.3f} ms")
print(f"  최소:         {result['min_ms']:.3f} ms")
print(f"  최대:         {result['max_ms']:.3f} ms")

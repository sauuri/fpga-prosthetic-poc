# 프로젝트 전체 공부 정리

이 프로젝트에서 다룬 모든 개념을 처음부터 공부할 수 있도록 정리했습니다.

---

## 1. EMG 신호란?

**EMG (Electromyography, 근전도)**
근육이 수축할 때 발생하는 전기 신호. 피부 표면에 전극을 붙여 측정한다.

```
뇌 → 신경 → 근육 수축 → 전기 신호 발생 → 전극이 감지
```

- 신호 크기: 0.1 ~ 5 mV (매우 약함)
- 주파수 대역: 20 ~ 450Hz
- 노이즈가 많아 전처리 필수

### 왜 의수에 쓰나?
팔을 잃어도 근육은 남아있다. 남은 근육의 EMG 신호로 "어떤 동작을 하려 했는지" 분류하면 의수를 제어할 수 있다.

---

## 2. 전처리 (Preprocessing)

### 2-1. 밴드패스 필터 (Bandpass Filter)
원하는 주파수 범위만 통과시키고 나머지 노이즈를 제거한다.

```
20Hz 이하 제거: 움직임 아티팩트 (몸 흔들림)
90Hz 이상 제거: 고주파 노이즈
```

**Butterworth 필터**
- 통과 대역이 최대한 평탄한 필터
- `order=4`: 차수가 높을수록 경계가 날카로움

```python
from scipy.signal import butter, filtfilt
b, a = butter(4, [20/100, 90/100], btype='band')
filtered = filtfilt(b, a, signal)
```

### 2-2. 슬라이딩 윈도우 (Sliding Window)
연속 신호를 일정 크기의 창으로 잘라서 특징을 추출한다.

```
|----window(200)---->|
        |----window(200)---->|
step=100
```

- `window_size=200`: 1초치 데이터 (200Hz × 1s)
- `step_size=100`: 0.5초씩 이동 (50% 겹침)

### 2-3. 특징 추출 (Feature Extraction)
윈도우마다 숫자 몇 개로 요약한다. 16채널 × 6특징 = **96차원**.

| 특징 | 공식 | 의미 |
|------|------|------|
| MAV | `mean(|x|)` | 신호 평균 크기 |
| RMS | `sqrt(mean(x²))` | 신호 에너지 |
| ZC | 부호가 바뀐 횟수 | 신호 진동 속도 |
| WL | `sum(|x[i+1]-x[i]|)` | 신호 복잡도 |
| MNF | `sum(f×PSD) / sum(PSD)` | 평균 주파수 |
| MDF | 누적 전력의 50% 지점 | 중앙 주파수 |

---

## 3. 머신러닝 — SVM

### 3-1. SVM (Support Vector Machine) 이란?
데이터를 두 그룹으로 나누는 **최적의 경계선(hyperplane)** 을 찾는 알고리즘.
경계선과 가장 가까운 데이터 포인트(support vector)와의 거리(margin)를 최대화한다.

```
  O  O  |  X  X
  O     |     X
  O  O  |  X
      margin
```

### 3-2. RBF 커널
데이터가 선형으로 분리 안 될 때 고차원으로 변환해서 분리한다.

```
K(x, x') = exp(-γ × ||x - x'||²)
```

- γ (gamma)가 클수록 경계가 복잡해짐 (과적합 위험)
- γ가 작을수록 경계가 단순해짐 (과소적합 위험)

### 3-3. GridSearchCV
최적 하이퍼파라미터를 자동으로 찾는 방법.

```python
param_grid = {
    'svm__C':     [1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.001]
}
# 9가지 조합 × 5-fold CV = 45번 학습해서 가장 좋은 조합 선택
```

- `C`: 오분류 허용 정도. 클수록 경계가 복잡
- `gamma`: RBF 커널 폭
- 최적값: `C=100, gamma=0.001`

### 3-4. OvO (One-vs-One) 다중 분류
12개 클래스 → 12×11/2 = **66개 이진 분류기**로 쪼갠다.
각 분류기가 투표, 가장 많은 표를 받은 클래스가 최종 예측.

```
(1 vs 2), (1 vs 3), ..., (11 vs 12) → 66번 투표 → 다수결
```

### 3-5. StandardScaler
특징값의 스케일을 맞춘다.

```
x_scaled = (x - mean) / std
```

- SVM은 스케일에 민감 → 반드시 정규화 필요

---

## 4. FPGA 기초

### 4-1. FPGA란?
**Field Programmable Gate Array**
하드웨어를 소프트웨어처럼 프로그래밍할 수 있는 칩.

| | CPU | GPU | FPGA |
|--|-----|-----|------|
| 특징 | 범용 | 병렬 계산 | 회로 직접 구현 |
| 장점 | 유연함 | 행렬 연산 빠름 | 저지연, 저전력 |
| 단점 | 느림 | 전력 많음 | 개발 어려움 |

### 4-2. PYNQ-Z2
Xilinx의 교육용 FPGA 보드.
- **PS (Processing System)**: ARM Cortex-A9 × 2 (667MHz)
- **PL (Programmable Logic)**: FPGA 로직 (LUT, DSP, BRAM)

```
PYNQ-Z2
├── PS (ARM) : Python 실행, OS 동작
└── PL (FPGA): 가속기 회로 동작
```

### 4-3. FPGA 리소스
| 리소스 | 역할 |
|--------|------|
| LUT (Look-Up Table) | 논리 연산 구현 |
| DSP (Digital Signal Processor) | 곱셈, 덧셈 고속 처리 |
| BRAM (Block RAM) | 온칩 메모리 |
| FF (Flip-Flop) | 1비트 레지스터 |

---

## 5. HLS (High-Level Synthesis)

### 5-1. HLS란?
C/C++ 코드를 자동으로 RTL(Verilog/VHDL) 하드웨어 코드로 변환하는 도구.
직접 Verilog 짜는 것보다 훨씬 빠르게 FPGA 가속기를 개발할 수 있다.

```
C++ 코드
   ↓ Vitis HLS
Verilog/VHDL (RTL)
   ↓ Vivado
비트스트림 (.bit)
   ↓ PYNQ
FPGA 동작
```

### 5-2. HLS Pragma
C++ 주석으로 하드웨어 최적화 지시

```cpp
// 루프 파이프라이닝: 매 사이클 새 입력 처리
#pragma HLS PIPELINE II=1

// 루프 언롤: N개 병렬로 처리
#pragma HLS UNROLL factor=8

// 배열 분할: 병렬 접근 가능하게
#pragma HLS ARRAY_PARTITION variable=x complete

// AXI Lite 인터페이스: CPU가 레지스터로 제어
#pragma HLS INTERFACE s_axilite port=x bundle=CTRL
```

### 5-3. Pipeline vs Unroll

**PIPELINE**: 공장 컨베이어 벨트처럼, 이전 계산이 끝나기 전에 다음 계산 시작
```
원래:  [A][B][C][A][B][C]
파이프라인: [A][B][C]
              [A][B][C]
```

**UNROLL**: 루프를 펼쳐서 병렬로 계산
```
원래:  for i in 0,1,2,3: x[i]*2
언롤:  x[0]*2, x[1]*2, x[2]*2, x[3]*2  (동시에)
```

→ UNROLL은 빠르지만 하드웨어(LUT, DSP)를 많이 쓴다.

### 5-4. AXI Lite 인터페이스
ARM CPU와 FPGA IP 사이의 통신 프로토콜.
레지스터 읽기/쓰기로 IP를 제어한다.

```python
# Python에서 FPGA IP 제어
svm_ip.write(0x00, 1)      # 시작 신호
while not (svm_ip.read(0x00) & 0x2):  # 완료 대기
    pass
result = svm_ip.read(0x10)  # 결과 읽기
```

---

## 6. SVM 추론 HLS 구현

### 전체 흐름
```
입력 x (96차원)
  ↓ Step 1: 정규화
x_scaled = (x - mean) / std
  ↓ Step 2: RBF 커널
K[m] = exp(-γ × ||x_scaled - sv[m]||²)   for m in 291 SVs
  ↓ Step 3: OvO 투표
for each pair (i,j):
    d = K[sv_i] @ dual_coef[j-1][sv_i] + K[sv_j] @ dual_coef[i][sv_j] + b
    if d > 0: votes[i]++  else: votes[j]++
  ↓ Step 4: argmax
pred = class[argmax(votes)]
```

### 리소스 vs 속도 트레이드오프
첫 시도 (UNROLL factor=8):
- LUT 207%, DSP 310% → **칩에 안 들어감**
- 지연시간: 1.121ms

두 번째 시도 (PIPELINE만):
- LUT 8%, DSP 7% → **칩에 들어감**
- 지연시간: 1.955ms

---

## 7. Vivado 블록 디자인

### 흐름
1. **IP Repository 추가**: HLS에서 만든 IP 등록
2. **Block Design 생성**: GUI에서 IP 블록 연결
3. **Zynq PS 추가**: ARM 프로세서 블록
4. **svm_inference IP 추가**: 가속기 블록
5. **Connection Automation**: AXI 버스 자동 연결
6. **HDL Wrapper 생성**: 최상위 Verilog 파일
7. **비트스트림 생성**: FPGA 프로그래밍 파일 (.bit)

### 파일 구조
```
비트스트림 배포에 필요한 파일:
├── svm_overlay.bit    # FPGA 회로 정보
└── svm_overlay.hwh    # 하드웨어 handshake (IP 주소 맵)
```

---

## 8. PYNQ 프레임워크

### PYNQ란?
FPGA를 Python으로 쉽게 제어할 수 있게 해주는 프레임워크.

```python
from pynq import Overlay

# 비트스트림 로드 (FPGA 프로그래밍)
overlay = Overlay('svm_overlay.bit')

# IP 접근
svm_ip = overlay.svm_inference_0

# 레지스터 제어
svm_ip.write(주소, 값)
값 = svm_ip.read(주소)
```

---

## 9. 성능 측정

### 지연시간 측정
```python
import time

latencies = []
for _ in range(100):
    start = time.perf_counter()
    model.predict(X)
    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # ms

print(f"평균: {np.mean(latencies):.3f} ms")
```

### 결과 해석
| 환경 | 지연시간 | 이유 |
|------|---------|------|
| PC | 0.210ms | 고성능 CPU |
| PYNQ ARM | 24.580ms | 저사양 Cortex-A9, Python 오버헤드 |
| FPGA | 6.83ms | AXI Lite 전송 오버헤드가 병목 |

→ FPGA가 ARM 대비 **3.6배 빠름**

FPGA 연산 자체(HLS 추정)는 1.955ms지만 실제 측정이 6.83ms인 이유:
AXI Lite로 96개 값을 Python 루프로 하나씩 전송하는 오버헤드 때문.
**DMA** 사용 시 이 병목 제거 가능.

---

## 10. 추가로 공부할 것

| 주제 | 내용 |
|------|------|
| DMA | AXI DMA로 대용량 데이터 고속 전송 |
| ap_fixed | 부동소수점 대신 고정소수점으로 리소스 절감 |
| Co-simulation | HLS RTL 검증 |
| Vivado IP Integrator | 복잡한 블록 디자인 |
| PYNQ DMA | `pynq.lib.dma` 사용법 |
| 실시간 EMG | Myo Armband, ADS1298 등 실제 센서 연결 |

---

## 전체 흐름 한눈에 보기

```
[NinaPro DB5 데이터]
        ↓
[전처리: 밴드패스 필터 + 특징 추출 (96차원)]
        ↓
[SVM 학습: GridSearch, C=100, gamma=0.001]
        ↓ 정확도 92.93%
[파라미터 추출: SV, dual_coef, intercept → .npz]
        ↓
        ├──[ARM 추론]──────────────── 24.580ms
        │   numpy로 직접 계산
        │
        └──[FPGA 가속]─────────────── 6.83ms (3.6배)
            HLS C++ → Vitis HLS → Vivado → 비트스트림
            PYNQ AXI Lite로 제어
```

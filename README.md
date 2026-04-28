# fpga-prosthetic-poc

FPGA 기반 EMG 실시간 추론 가속 — 의수 제어용 PoC

## 목표

EMG 신호 기반 손동작 분류 모델을 PYNQ-Z2 FPGA에 올려서,
ARM 소프트웨어 추론 대비 **지연시간(latency)** 을 비교한다.

```
EMG 신호 입력
    → 전처리 (필터링, 특징 추출)
    → 제스처 분류 모델 추론
    → [ARM baseline] vs [FPGA 가속]
    → 지연시간 비교
```

---

## 결과 요약

| 항목 | ARM (PYNQ Cortex-A9) | FPGA (HLS 가속) |
|------|----------------------|-----------------|
| 추론 지연시간 | 24.580 ms | 6.83 ms |
| 속도 향상 | 1x (baseline) | **3.6x 빠름** |
| 분류 정확도 | 92.93% (SVM, 12 클래스) | 동일 |

> **실측 결과**: 모든 지연시간 수치는 실제 PYNQ-Z2 보드에서 측정한 값입니다 (시뮬레이션 아님).

---

## 하드웨어

- **FPGA 보드**: Xilinx PYNQ-Z2 (Zynq-7020, Cortex-A9 667MHz)
- **EMG 데이터**: NinaPro DB5 (공개 데이터셋)

---

## 프로젝트 단계

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | 데이터셋 확보 및 탐색 | ✅ |
| 2 | 전처리 파이프라인 구현 | ✅ |
| 3 | PC에서 분류 모델 학습 | ✅ |
| 4 | PYNQ ARM에서 추론 (baseline) | ✅ |
| 5 | HLS로 추론 가속 구현 | ✅ |
| 6 | 성능 비교 및 결과 정리 | ✅ |

---

## 폴더 구조

```
fpga-prosthetic-poc/
├── data/
│   └── README.md           # NinaPro DB5 다운로드 가이드
├── hardware/
│   └── sensors.md          # 센서 구성 (IMU, Flex Sensor)
├── src/
│   ├── preprocess.py       # EMG 신호 전처리
│   ├── train.py            # SVM 모델 학습
│   ├── benchmark.py        # 지연시간 측정 유틸
│   ├── main.py             # 전체 파이프라인 실행 스크립트
│   ├── pynq_inference.ipynb    # PYNQ ARM baseline 측정 노트북
│   └── fpga_benchmark.ipynb    # FPGA 가속 측정 노트북
├── hls/
│   ├── svm_inference.h     # HLS 함수 선언
│   ├── svm_inference.cpp   # HLS SVM 추론 가속 코드
│   ├── tb_svm_inference.cpp # 테스트벤치
│   ├── run_hls.tcl         # Vitis HLS 자동화 스크립트
│   └── hls_config.cfg      # Vitis HLS 설정 파일
├── results/
│   └── svm_params.npz      # 학습된 SVM 파라미터 (numpy)
└── requirements.txt
```

---

## 전처리 파이프라인

### 필터링
- 밴드패스 필터: 20 ~ 90Hz (Butterworth 4차)
- 샘플링 레이트: 200Hz

### 특징 추출 (`src/preprocess.py`)
슬라이딩 윈도우 (200 samples, step 100) 기반 특징:

| 특징 | 설명 |
|------|------|
| MAV | Mean Absolute Value |
| RMS | Root Mean Square |
| ZC | Zero Crossing |
| WL | Waveform Length |
| MNF | Mean Frequency (주파수 도메인) |
| MDF | Median Frequency (주파수 도메인) |

→ 채널 16개 × 6가지 = **96차원 특징 벡터**

---

## 모델 학습

### 모델: SVM (RBF 커널)

- GridSearchCV로 하이퍼파라미터 튜닝
- 최적 파라미터: `C=100`, `gamma=0.001`
- StandardScaler 정규화 포함 (Pipeline)

### 데이터셋

- NinaPro DB5, Subject 1, Exercise 1
- 12가지 손동작 클래스
- 특징 벡터 495개 (train 80% / test 20%)

### 정확도

| 모델 | 정확도 |
|------|--------|
| SVM (기본, 64 features) | 84.85% |
| SVM (GridSearch + MNF/MDF 추가) | **92.93%** |

---

## 실행 방법

### 환경 설정

```bash
pip install -r requirements.txt
```

### 학습 실행

```bash
# NinaPro DB5 기본 실행
python src/main.py --dataset ninapro --data_dir data/s1 --subjects 1 --model svm

# GridSearch 튜닝 포함
python src/main.py --dataset ninapro --data_dir data/s1 --subjects 1 --model svm --tune

# 벤치마크까지
python src/main.py --dataset ninapro --data_dir data/s1 --subjects 1 --model svm --tune --benchmark

# UCI EMG (빠른 테스트)
python src/main.py --dataset uci --data_dir data/uci_emg
```

---

## FPGA 가속 구현

### HLS 코드 (`hls/svm_inference.cpp`)

**추론 파이프라인:**
1. 입력 정규화 (StandardScaler)
2. RBF 커널 계산: `K[m] = exp(-γ × ||x - sv_m||²)`
3. OvO (One-vs-One) 투표: 66쌍 이진 분류기
4. 최다 득표 클래스 반환

**HLS pragma:**
- `PIPELINE II=1`: 루프 파이프라이닝
- `s_axilite`: AXI Lite 인터페이스 (PYNQ 제어)

### 합성 결과 (Vitis HLS 2025.2)

| 항목 | 값 |
|------|-----|
| 타겟 디바이스 | xc7z020clg400-1 |
| 클럭 | 10ns (100MHz) |
| 최대 지연시간 | 1.955 ms (41,201 cycles) |
| LUT | 4,462 / 53,200 (8%) |
| DSP | 17 / 220 (7%) |
| BRAM | 79 / 280 (28%) |
| FF | 2,411 / 106,400 (2%) |

### Vitis HLS 합성

```bash
cd hls/
vitis_hls -f run_hls.tcl
```

---

## PYNQ 배포

### 파일 전송 (PC → PYNQ)

```bash
scp results/svm_params.npz xilinx@<PYNQ_IP>:~/jupyter_notebooks/fpga-prosthetic-poc/
scp src/pynq_inference.ipynb xilinx@<PYNQ_IP>:~/jupyter_notebooks/fpga-prosthetic-poc/
scp src/fpga_benchmark.ipynb xilinx@<PYNQ_IP>:~/jupyter_notebooks/fpga-prosthetic-poc/
# 비트스트림 (.bit, .hwh) 도 전송
```

### PYNQ ARM baseline 측정

Jupyter에서 `pynq_inference.ipynb` 실행
→ 평균 지연시간: **24.580 ms**

### FPGA 가속 측정

Jupyter에서 `fpga_benchmark.ipynb` 실행
→ 평균 지연시간: **6.83 ms (3.6배 향상)**

---

## 데이터셋

NinaPro DB5 — 손동작 53가지, 피험자 10명, EMG 16채널 (200Hz)
→ [다운로드 방법](data/README.md)

---

## 센서 구성 (향후 실시간 연동)

→ [센서 가이드](hardware/sensors.md)


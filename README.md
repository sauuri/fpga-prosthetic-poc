# fpga-prosthetic-poc

FPGA 기반 EMG 실시간 추론 가속 — 의수 제어용 PoC

## 목표

EMG 신호 기반 손동작 분류 모델을 PYNQ-Z2 FPGA에 올려서,  
ARM 소프트웨어 추론 대비 **지연시간(latency)** 과 **전력소비** 를 비교한다.

```
EMG 신호 입력
    → 전처리 (필터링, 특징 추출)
    → 제스처 분류 모델 추론
    → [ARM baseline] vs [FPGA 가속]
    → 지연시간 / 전력 비교
```

## 하드웨어

- **FPGA 보드**: Xilinx PYNQ-Z2
- **EMG 데이터**: NinaPro DB5 (공개 데이터셋)

## 프로젝트 단계

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | 데이터셋 확보 및 탐색 | 🔲 |
| 2 | 전처리 파이프라인 구현 | 🔲 |
| 3 | PC에서 분류 모델 학습 | 🔲 |
| 4 | PYNQ ARM에서 추론 (baseline) | 🔲 |
| 5 | HLS로 추론 가속 구현 | 🔲 |
| 6 | 성능 비교 및 결과 정리 | 🔲 |

## 폴더 구조

```
fpga-prosthetic-poc/
├── data/           # 데이터셋 다운로드 가이드
├── notebooks/      # 데이터 탐색, 모델 학습 노트북
├── src/            # 전처리, 학습, 평가 코드
├── hls/            # FPGA 추론 HLS C++ 코드
├── pynq/           # PYNQ 보드 배포 스크립트
└── results/        # 측정 결과 (지연시간, 전력)
```

## 데이터셋

NinaPro DB5 사용 — 손동작 53가지, 10명 피험자, EMG 16채널  
→ [다운로드 방법](data/README.md)

## 환경 설정

```bash
pip install -r requirements.txt
```

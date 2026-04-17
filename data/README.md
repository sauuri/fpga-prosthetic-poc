# 데이터셋 다운로드

## NinaPro DB5

손동작 EMG 데이터셋. 이 프로젝트에서 사용하는 공개 데이터.

- 피험자: 10명
- 손동작: 53가지
- EMG 채널: 16채널 (Myo Armband 2개)
- 샘플링 레이트: 200Hz

### 다운로드

1. http://ninapro.hevs.ch 접속
2. DB5 신청 (무료, 이메일 등록 필요)
3. `data/` 폴더에 압축 해제

### 파일 구조 (다운로드 후)

```
data/
└── ninapro_db5/
    ├── s1/
    │   ├── S1_E1_A1.mat
    │   ├── S1_E2_A1.mat
    │   └── S1_E3_A1.mat
    ├── s2/
    ...
```

### 빠른 시작용 (소규모 테스트)

DB5 신청이 번거로우면 먼저 UCI EMG 데이터셋으로 시작 가능:
- https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures
- 즉시 다운로드 가능, 4가지 손동작

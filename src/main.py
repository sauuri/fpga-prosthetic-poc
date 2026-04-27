import argparse
import os
import numpy as np
import scipy.io
from tqdm import tqdm

from preprocess import bandpass_filter, extract_features
from train import train_classifier, save_model
from benchmark import measure_latency, print_benchmark


def load_ninapro_db5(data_dir, subjects=None, exercises=None):
    """
    NinaPro DB5 .mat 파일 로드
    data_dir: data/ninapro_db5/ 경로
    subjects: 불러올 피험자 번호 리스트 (기본: 전체)
    exercises: 불러올 운동 번호 리스트 (기본: [1])
    """
    if subjects is None:
        subjects = list(range(1, 11))  # s1 ~ s10
    if exercises is None:
        exercises = [1]

    all_emg, all_labels = [], []

    for s in tqdm(subjects, desc="Loading subjects"):
        for e in exercises:
            path = os.path.join(data_dir, f"s{s}", f"S{s}_E{e}_A1.mat")
            if not os.path.exists(path):
                print(f"  [skip] {path} not found")
                continue

            mat = scipy.io.loadmat(path)
            emg = mat['emg'].astype(np.float32)        # (N, 16)
            labels = mat['restimulus'].flatten()        # (N,)

            # 휴식(0) 제외
            mask = labels > 0
            all_emg.append(emg[mask])
            all_labels.append(labels[mask])

    if not all_emg:
        raise FileNotFoundError(f"데이터를 찾을 수 없습니다: {data_dir}")

    return np.concatenate(all_emg), np.concatenate(all_labels)


def load_uci_emg(data_dir):
    """
    UCI EMG Gestures 데이터셋 로드 (CSV 포맷)
    data_dir: UCI EMG 데이터 폴더 경로
    """
    import pandas as pd

    all_emg, all_labels = [], []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")

    for fname in tqdm(sorted(csv_files), desc="Loading CSV files"):
        df = pd.read_csv(os.path.join(data_dir, fname), header=None)
        # 마지막 열이 레이블, 나머지가 EMG
        emg = df.iloc[:, :-1].values.astype(np.float32)
        labels = df.iloc[:, -1].values
        all_emg.append(emg)
        all_labels.append(labels)

    return np.concatenate(all_emg), np.concatenate(all_labels)


def build_dataset(emg_raw, labels_raw, fs=200, window_size=200, step_size=100):
    """전처리 + 특징 추출 → (X, y) 반환"""
    print("전처리 중...")
    emg_filtered = bandpass_filter(emg_raw, fs=fs)

    print("특징 추출 중...")
    X = extract_features(emg_filtered, window_size=window_size, step_size=step_size, fs=fs)

    # 레이블을 윈도우 단위로 맞춤 (윈도우 중앙값 기준)
    n_windows = len(X)
    label_indices = [
        int((start + window_size // 2))
        for start in range(0, len(emg_raw) - window_size, step_size)
    ][:n_windows]
    y = labels_raw[label_indices]

    print(f"데이터셋: X={X.shape}, y={y.shape}, 클래스 수={len(np.unique(y))}")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="EMG 제스처 분류 파이프라인")
    parser.add_argument('--dataset', choices=['ninapro', 'uci'], default='ninapro',
                        help="사용할 데이터셋 (default: ninapro)")
    parser.add_argument('--data_dir', type=str, default='data/ninapro_db5',
                        help="데이터 폴더 경로")
    parser.add_argument('--model', choices=['svm', 'rf'], default='svm',
                        help="분류 모델 (default: svm)")
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help="NinaPro 피험자 번호 (기본: 전체)")
    parser.add_argument('--exercises', type=int, nargs='+', default=None,
                        help="NinaPro 운동 번호 (기본: [1])")
    parser.add_argument('--save_path', type=str, default='results/model.pkl',
                        help="모델 저장 경로")
    parser.add_argument('--benchmark', action='store_true',
                        help="학습 후 지연시간 벤치마크 실행")
    parser.add_argument('--tune', action='store_true',
                        help="SVM GridSearch 하이퍼파라미터 튜닝")
    args = parser.parse_args()

    # 1. 데이터 로드
    print(f"\n[1] 데이터 로드 ({args.dataset})")
    if args.dataset == 'ninapro':
        emg_raw, labels_raw = load_ninapro_db5(args.data_dir, subjects=args.subjects, exercises=args.exercises)
    else:
        emg_raw, labels_raw = load_uci_emg(args.data_dir)

    # 2. 전처리 + 특징 추출
    print("\n[2] 전처리 & 특징 추출")
    X, y = build_dataset(emg_raw, labels_raw)

    # 3. 모델 학습
    print(f"\n[3] 모델 학습 ({args.model})")
    model, acc = train_classifier(X, y, model_type=args.model, tune=args.tune)

    # 4. 모델 저장
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_model(model, args.save_path)

    # 5. 벤치마크 (선택)
    if args.benchmark:
        print("\n[4] 지연시간 벤치마크")
        sample = X[:1]
        result = measure_latency(model.predict, sample)
        print_benchmark("ARM baseline (PC)", result)

    print("\n완료.")


if __name__ == '__main__':
    main()

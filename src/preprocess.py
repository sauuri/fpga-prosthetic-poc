import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=20, highcut=450, fs=200, order=4):
    """EMG 신호 밴드패스 필터 (20-450Hz)"""
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


def extract_features(signal, window_size=200, step_size=100):
    """
    슬라이딩 윈도우로 특징 추출
    - MAV: Mean Absolute Value
    - RMS: Root Mean Square
    - ZC: Zero Crossing
    - WL: Waveform Length
    """
    features = []
    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        mav = np.mean(np.abs(window), axis=0)
        rms = np.sqrt(np.mean(window ** 2, axis=0))
        zc = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)
        wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
        features.append(np.concatenate([mav, rms, zc, wl]))
    return np.array(features)

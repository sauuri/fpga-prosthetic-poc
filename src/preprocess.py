import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=20, highcut=90, fs=200, order=4):
    """EMG 신호 밴드패스 필터 (20-90Hz)"""
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


def extract_features(signal, window_size=200, step_size=100, fs=200):
    """
    슬라이딩 윈도우로 특징 추출
    시간 도메인: MAV, RMS, ZC, WL
    주파수 도메인: MNF (Mean Frequency), MDF (Median Frequency)
    """
    features = []
    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)

    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]

        # 시간 도메인
        mav = np.mean(np.abs(window), axis=0)
        rms = np.sqrt(np.mean(window ** 2, axis=0))
        zc = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)
        wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)

        # 주파수 도메인
        psd = np.abs(np.fft.rfft(window, axis=0)) ** 2  # (freq_bins, channels)
        total_power = np.sum(psd, axis=0) + 1e-10
        mnf = np.sum(freqs[:, None] * psd, axis=0) / total_power
        cumpower = np.cumsum(psd, axis=0)
        mdf = np.array([
            freqs[np.searchsorted(cumpower[:, ch], total_power[ch] / 2)]
            for ch in range(psd.shape[1])
        ])

        features.append(np.concatenate([mav, rms, zc, wl, mnf, mdf]))
    return np.array(features)

"""
fpga-prosthetic-poc 시각화 스크립트
results/svm_params.npz + 실측 벤치마크 수치 기반
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(OUT_DIR, 'svm_params.npz')

# ── 색상 팔레트 ─────────────────────────────────────────────
C_ARM  = '#E07B54'
C_FPGA = '#4A90D9'
C_GRAY = '#AAAAAA'
FEATURE_COLORS = ['#4A90D9','#E07B54','#2ECC71','#9B59B6','#F1C40F','#1ABC9C']

FEATURE_NAMES = ['MAV', 'RMS', 'ZC', 'WL', 'MNF', 'MDF']
N_CH = 16
N_FEAT = 6


def load():
    d = np.load(NPZ_PATH, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


# ── 1. 지연시간 비교 ─────────────────────────────────────────
def plot_latency(ax):
    labels = ['ARM\n(Cortex-A9)', 'FPGA\n(HLS Accel)']
    values = [24.580, 6.83]
    colors = [C_ARM, C_FPGA]

    bars = ax.bar(labels, values, color=colors, width=0.45,
                  edgecolor='white', linewidth=1.5, zorder=3)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Inference Latency Comparison\n(Measured on PYNQ-Z2)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 30)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f'{val:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.annotate('', xy=(1, 6.83), xytext=(1, 24.580),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    ax.text(1.28, 15.7, '3.6x\nFaster', ha='center', va='center',
            fontsize=10, color='#333333', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── 2. FPGA 리소스 사용률 ────────────────────────────────────
def plot_resources(ax):
    resources = ['LUT', 'DSP', 'BRAM', 'FF']
    used  = [4462,  17,   79,   2411]
    total = [53200, 220, 280, 106400]
    pct   = [u / t * 100 for u, t in zip(used, total)]
    colors = [C_FPGA, '#2ECC71', '#E07B54', '#9B59B6']

    bars = ax.barh(resources, pct, color=colors, height=0.5,
                   edgecolor='white', linewidth=1.2)
    ax.set_xlim(0, 40)
    ax.set_xlabel('Utilization (%)', fontsize=11)
    ax.set_title('FPGA Resource Utilization\n(xc7z020clg400-1)', fontsize=12, fontweight='bold')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    for bar, p, u, t in zip(bars, pct, used, total):
        ax.text(p + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{p:.1f}%  ({u:,} / {t:,})',
                va='center', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── 3. 클래스별 서포트 벡터 수 ──────────────────────────────
def plot_n_support(ax, data):
    n_support = data['n_support']
    classes   = data['classes'].astype(int)
    x = np.arange(len(classes))

    bars = ax.bar(x, n_support, color=C_FPGA, width=0.6,
                  edgecolor='white', linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{c}' for c in classes], fontsize=9)
    ax.set_ylabel('Support Vectors', fontsize=11)
    ax.set_title(f'Support Vectors per Class\n(Total: {n_support.sum()}, 12 classes)', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, n_support):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                str(v), ha='center', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── 4. 특징 벡터 평균값 히트맵 (6특징 × 16채널) ──────────────
def plot_feature_heatmap(ax, data):
    mean = data['scaler_mean'].reshape(N_FEAT, N_CH)

    # 각 특징 타입을 독립적으로 정규화 (단위가 달라서)
    mean_norm = np.zeros_like(mean)
    for i in range(N_FEAT):
        row = mean[i]
        rng = row.max() - row.min()
        mean_norm[i] = (row - row.min()) / (rng + 1e-10)

    im = ax.imshow(mean_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(N_FEAT))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=10)
    ax.set_xticks(range(0, N_CH, 2))
    ax.set_xticklabels([f'Ch{i+1}' for i in range(0, N_CH, 2)], fontsize=8, rotation=45)
    ax.set_xlabel('EMG Channel', fontsize=11)
    ax.set_title('Feature Mean Heatmap\n(Row-normalized, 6 features x 16 channels)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Relative Magnitude')


# ── 5. 특징 타입별 평균·분산 ─────────────────────────────────
def plot_feature_stats(ax, data):
    mean = data['scaler_mean'].reshape(N_FEAT, N_CH)
    std  = data['scaler_std'].reshape(N_FEAT, N_CH)

    feat_mean = mean.mean(axis=1)
    feat_std_of_mean = mean.std(axis=1)

    x = np.arange(N_FEAT)
    bars = ax.bar(x, feat_mean, yerr=feat_std_of_mean,
                  color=FEATURE_COLORS, width=0.55,
                  edgecolor='white', linewidth=1.2,
                  capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': '#555555'},
                  zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, fontsize=11)
    ax.set_ylabel('Mean Value (avg over 16 channels)', fontsize=11)
    ax.set_title('Mean Value per Feature Type\n(error bar: std across channels)', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── 메인 ─────────────────────────────────────────────────────
def main():
    data = load()

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('fpga-prosthetic-poc  |  EMG Gesture Classification FPGA Acceleration',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2])

    plot_latency(ax1)
    plot_resources(ax2)
    plot_n_support(ax3, data)
    plot_feature_heatmap(ax4, data)
    plot_feature_stats(ax5, data)

    out_path = os.path.join(OUT_DIR, 'analysis.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()

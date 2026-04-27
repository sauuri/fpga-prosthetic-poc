#include "svm_inference.h"

void svm_inference(data_t x[N_FEAT], label_t *pred) {
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pred    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=x       bundle=CTRL

    // ----------------------------------------------------------------
    // Step 1: 정규화 (StandardScaler)
    // ----------------------------------------------------------------
    data_t x_scaled[N_FEAT];

    SCALE_LOOP:
    for (int f = 0; f < N_FEAT; f++) {
#pragma HLS PIPELINE II=1
        x_scaled[f] = (x[f] - SCALER_MEAN[f]) / SCALER_STD[f];
    }

    // ----------------------------------------------------------------
    // Step 2: RBF 커널 계산
    // K[m] = exp(-gamma * sum_f (x_scaled[f] - sv[m][f])^2)
    // ----------------------------------------------------------------
    data_t K[N_SV];

    KERNEL_OUTER:
    for (int m = 0; m < N_SV; m++) {
        data_t dist2 = 0.0f;
        KERNEL_INNER:
        for (int f = 0; f < N_FEAT; f++) {
#pragma HLS PIPELINE II=1
            data_t diff = x_scaled[f] - SUPPORT_VECTORS[m][f];
            dist2 += diff * diff;
        }
        K[m] = expf(-GAMMA * dist2);
    }

    // ----------------------------------------------------------------
    // Step 3: OvO 투표
    // ----------------------------------------------------------------
    int votes[N_CLASSES];

    VOTE_INIT:
    for (int c = 0; c < N_CLASSES; c++) {
#pragma HLS UNROLL
        votes[c] = 0;
    }

    int pair_idx = 0;
    OVO_I:
    for (int i = 0; i < N_CLASSES; i++) {
        OVO_J:
        for (int j = i + 1; j < N_CLASSES; j++) {

            data_t d = INTERCEPT[pair_idx];

            SV_I_LOOP:
            for (int k = 0; k < N_SUPPORT[i]; k++) {
#pragma HLS PIPELINE II=1
                d += K[SV_START[i] + k] * DUAL_COEF[j-1][SV_START[i] + k];
            }

            SV_J_LOOP:
            for (int k = 0; k < N_SUPPORT[j]; k++) {
#pragma HLS PIPELINE II=1
                d += K[SV_START[j] + k] * DUAL_COEF[i][SV_START[j] + k];
            }

            if (d > 0) votes[i]++;
            else       votes[j]++;

            pair_idx++;
        }
    }

    // ----------------------------------------------------------------
    // Step 4: 최다 득표 클래스 선택
    // ----------------------------------------------------------------
    int best_idx = 0;
    ARGMAX:
    for (int c = 1; c < N_CLASSES; c++) {
#pragma HLS UNROLL
        if (votes[c] > votes[best_idx]) best_idx = c;
    }

    *pred = CLASSES[best_idx];
}

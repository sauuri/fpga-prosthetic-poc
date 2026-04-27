#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "svm_inference.h"

int main() {
    // 더미 입력 (모두 0.5)
    data_t x[N_FEAT];
    for (int f = 0; f < N_FEAT; f++) x[f] = 0.5f;

    label_t pred = -1;
    svm_inference(x, &pred);

    printf("예측 클래스: %d\n", pred);

    if (pred >= 1 && pred <= 12) {
        printf("PASS: 유효한 클래스 출력\n");
        return 0;
    } else {
        printf("FAIL: 비정상 출력\n");
        return 1;
    }
}

#ifndef SVM_INFERENCE_H
#define SVM_INFERENCE_H

#include "svm_params.h"
#include <math.h>

typedef float data_t;
typedef int   label_t;

// Top-level HLS function
// x   : 입력 특징 벡터 (N_FEAT 개)
// pred: 예측된 제스처 클래스 (1~12)
void svm_inference(data_t x[N_FEAT], label_t *pred);

#endif

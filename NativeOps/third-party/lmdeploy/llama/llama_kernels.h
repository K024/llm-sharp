// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <numeric>

namespace turbomind {

template<typename T>
void invokeRootMeanSquareNorm(T* out, const T* input, const T* scale, float eps, int m, int n, cudaStream_t stream);

}  // namespace turbomind

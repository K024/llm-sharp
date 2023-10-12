#include "utils.hpp"

#include <c10/cuda/CUDAGuard.h>


void vecquant4matmul_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

EXPORT_API(void)
gptq_vecquant4matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  CATCH(
    const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
    vecquant4matmul_cuda_old(vec, mat, mul, scales, zeros, groupsize);
  )
}


void vecquant4matmul_faster_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
); 

EXPORT_API(void)
gptq_vecquant4matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  CATCH(
    const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
    vecquant4matmul_faster_cuda_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
  )
}

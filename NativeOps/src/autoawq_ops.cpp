#include "utils.hpp"

// #include "../third-party/AutoAWQ/awq_cuda/layernorm/layernorm.h"
// #include "../third-party/AutoAWQ/awq_cuda/quantization/gemm_cuda.h"
// #include "../third-party/AutoAWQ/awq_cuda/quantization/gemv_cuda.h"
// #include "../third-party/AutoAWQ/awq_cuda/position_embedding/pos_encoding.h"
// #include "../third-party/AutoAWQ/awq_cuda/attention/ft_attention.h"


torch::Tensor gemm_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters);

torch::Tensor gemmv2_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int group_size, int split_k_iters);

torch::Tensor gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size);


EXPORT_API(Tensor)
awq_gemm_forward(const Tensor in_feats, const Tensor kernel, const Tensor scaling_factors, const Tensor zeros, int split_k_iters)
{
    CATCH_TENSOR(gemm_forward_cuda(*in_feats, *kernel, *scaling_factors, *zeros, split_k_iters));
}

EXPORT_API(Tensor)
awq_gemmv2_forward(const Tensor in_feats, const Tensor kernel, const Tensor scaling_factors, const Tensor zeros, int group_size, int split_k_iters)
{
    CATCH_TENSOR(gemmv2_forward_cuda(*in_feats, *kernel, *scaling_factors, *zeros, group_size, split_k_iters));
}

EXPORT_API(Tensor)
awq_gemv_forward(const Tensor in_feats, const Tensor kernel, const Tensor scaling_factors, const Tensor zeros, int group_size)
{
    CATCH_TENSOR(gemv_forward_cuda(*in_feats, *kernel, *scaling_factors, *zeros, group_size));
}

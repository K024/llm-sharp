#include "utils.hpp"

#include "../third-party/AutoAWQ/quantization/gemm_cuda.h"
#include "../third-party/AutoAWQ/quantization/gemv_cuda.h"
#include "../third-party/AutoAWQ/position_embedding/pos_encoding.h"


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

EXPORT_API(void)
awq_rotary_embedding_neox(const Tensor cos, const Tensor sin, const Tensor query, const Tensor key, Tensor out_query, Tensor out_key)
{
    CATCH(
        rotary_embedding_neox(*cos, *sin, *query, *key, *out_query, *out_key);
    );
}

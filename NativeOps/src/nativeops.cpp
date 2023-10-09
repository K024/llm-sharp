#include "utils.hpp"

#include "../third-party/AutoAWQ/awq_cuda/layernorm/layernorm.h"
#include "../third-party/AutoAWQ/awq_cuda/quantization/gemm_cuda.h"
#include "../third-party/AutoAWQ/awq_cuda/quantization/gemv_cuda.h"
#include "../third-party/AutoAWQ/awq_cuda/position_embedding/pos_encoding.h"
// #include "../third-party/AutoAWQ/awq_cuda/attention/ft_attention.h"

thread_local char *torch_last_err = nullptr;

EXPORT_API(const char *)
llm_sharp_check_last_err()
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

EXPORT_API(Tensor)
llm_sharp_hello(const Tensor tensor)
{
    CATCH_TENSOR(tensor->add(1));
}

EXPORT_API(Tensor)
awq_layernorm_forward(const Tensor input, const Tensor gamma, float eps)
{
    at::Tensor res = at::Tensor();
    CATCH(
        at::Tensor output = torch::empty_like(*input);
        layernorm_forward_cuda(*input, *gamma, output, eps);
        res = output;
    );
    return ResultTensor(res);
}

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
awq_rotary_embedding_neox(const Tensor positions, const Tensor query, const Tensor key, int head_size, const Tensor cos_sin_cache)
{
    CATCH({
        rotary_embedding_neox(*positions, *query, *key, head_size, *cos_sin_cache);
    });
}

// EXPORT_API(Tensor)
// awq_single_query_attention(const Tensor q, const Tensor k, const Tensor v, const Tensor k_cache, const Tensor v_cache,
//     const Tensor length_per_sample_, const Tensor alibi_slopes_, int timestep, int rotary_embedding_dim, float rotary_base, bool neox_rotary_style)
// {
//     CATCH_TENSOR(single_query_attention(*q, *k, *v, *k_cache, *v_cache,
//         *length_per_sample_, *alibi_slopes_, timestep, rotary_embedding_dim, rotary_base, neox_rotary_style));
// }

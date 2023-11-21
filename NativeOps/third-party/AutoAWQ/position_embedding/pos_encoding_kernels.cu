/*

Adapted from the VLLM project:
https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu

*/

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include "pos_encoding.h"

template<typename scalar_t>
__global__ void rotary_embedding_neox_kernel(
  const scalar_t* __restrict__ rot_cos,    // [b, num_tokens, 1, rot_dim]
  const scalar_t* __restrict__ rot_sin,    // [b, num_tokens, 1, rot_dim]
  const scalar_t* __restrict__ query,      // [b, num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ key,        // [b, num_tokens, num_heads, head_size]
  scalar_t* __restrict__ out_query,
  scalar_t* __restrict__ out_key,
  const int rot_dim,
  const int stride_in,
  const int stride_out,
  const int num_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  const scalar_t* cos_ptr = rot_cos + token_idx * rot_dim;
  const scalar_t* sin_ptr = rot_sin + token_idx * rot_dim;

  const int embed_dim = rot_dim / 2;
  const int n = num_heads * embed_dim;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * stride_in + head_idx * head_size;
    const int out_token_head = token_idx * stride_out + head_idx * head_size;

    const int rot_offset = i % embed_dim;
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;

    const scalar_t cos = __ldg(cos_ptr + x_index); // redundant at last dim
    const scalar_t sin = __ldg(sin_ptr + x_index); // redundant at last dim

    const scalar_t q_x = query[token_head + x_index];
    const scalar_t q_y = query[token_head + y_index];
    out_query[out_token_head + x_index] = q_x * cos - q_y * sin;
    out_query[out_token_head + y_index] = q_y * cos + q_x * sin;

    const scalar_t k_x = key[token_head + x_index];
    const scalar_t k_y = key[token_head + y_index];
    out_key[out_token_head + x_index] = k_x * cos - k_y * sin;
    out_key[out_token_head + y_index] = k_y * cos + k_x * sin;
  }
}

void rotary_embedding_neox(
  torch::Tensor& cos,               // [b, num_tokens, 1, rot_dim]
  torch::Tensor& sin,               // [b, num_tokens, 1, rot_dim]
  torch::Tensor& query,             // [b, num_tokens, num_heads, head_size]
  torch::Tensor& key,               // [b, num_tokens, num_heads, head_size]
  torch::Tensor& out_query,
  torch::Tensor& out_key)
{
  int rot_dim = cos.size(-1);
  int num_tokens = query.size(0) * query.size(1);
  int num_heads = query.size(2);
  int head_size = query.size(3);
  int stride_in = query.stride(1);
  int stride_out = out_query.stride(1);

  TORCH_CHECK(cos.size(0) == query.size(0));
  TORCH_CHECK(cos.size(1) == query.size(1));
  TORCH_CHECK(key.size(2) == query.size(2));
  TORCH_CHECK(key.size(3) == query.size(3));

  TORCH_CHECK(cos.is_contiguous());
  TORCH_CHECK(sin.is_contiguous());
  TORCH_CHECK(query.stride(3) == 1);
  TORCH_CHECK(key.stride(3) == 1);
  TORCH_CHECK(query.stride(2) == head_size);
  TORCH_CHECK(key.stride(2) == head_size);
  TORCH_CHECK(out_query.is_contiguous());
  TORCH_CHECK(out_key.is_contiguous());

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(cos.device().index()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    query.scalar_type(),
    "rotary_embedding_neox",
    [&] {
      rotary_embedding_neox_kernel<scalar_t><<<grid, block, 0, stream>>>(
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        out_query.data_ptr<scalar_t>(),
        out_key.data_ptr<scalar_t>(),
        rot_dim,
        stride_in,
        stride_out,
        num_heads,
        head_size);
    });
}


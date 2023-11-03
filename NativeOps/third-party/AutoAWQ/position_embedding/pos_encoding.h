#pragma once
#include <torch/torch.h>

void rotary_embedding_neox(
  torch::Tensor& cos,               // [b, num_tokens, 1, rot_dim * 2]
  torch::Tensor& sin,               // [b, num_tokens, 1, rot_dim * 2]
  torch::Tensor& query,             // [b, num_tokens, num_heads, head_size]
  torch::Tensor& key,               // [b, num_tokens, num_heads, head_size]
  torch::Tensor& out_query,
  torch::Tensor& out_key);

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include <limits>
#include <assert.h>
#include <cmath>
#include <cstring>

#define UNUSED(x) (void)(x)
#define DEBUG_ONLY(x) (void)(x)

#ifdef _WIN32
#include <intrin.h>

#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret
#else
// #include "UnixSal.h"

#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret

#define __forceinline __attribute__((always_inline)) inline
#endif

#include <string>

#undef TORCH_API_INCLUDE_EXTENSION_H
#include "torch/torch.h"

extern thread_local char *torch_last_err;

typedef torch::Tensor *Tensor;
typedef torch::Scalar *Scalar;
typedef torch::Generator* Generator;
typedef c10::Storage* Storage;
typedef torch::nn::utils::rnn::PackedSequence* PackedSequence;

#define THS_API TH_API

#define CATCH(x) \
  try { \
    torch_last_err = 0; \
    x \
  } catch (const c10::Error e) { \
      torch_last_err = strdup(e.what()); \
  } catch (const std::runtime_error e) { \
      torch_last_err = strdup(e.what()); \
  }

#define CATCH_RETURN_RES(ty, dflt, stmt) \
    ty res = dflt; \
    CATCH(  \
        stmt;  \
    );  \
    return res;

#define CATCH_RETURN(ty, dflt, expr) CATCH_RETURN_RES(ty, dflt, res = expr)
#define CATCH_RETURN_NNModule(stmt) CATCH_RETURN_RES(NNModule, nullptr, stmt)
#define CATCH_RETURN_Tensor(stmt) CATCH_RETURN_RES(Tensor, nullptr, stmt)

// Return undefined tensors as nullptr to C#
inline Tensor ResultTensor(const at::Tensor & res)
{
    if (res.defined())
        return new torch::Tensor(res);
    else
        return nullptr;
}

#define CATCH_TENSOR(expr) \
    at::Tensor res = at::Tensor(); \
    CATCH(  \
        res = expr;  \
    );  \
    return ResultTensor(res);

#define CATCH_TENSORS_2(expr) \
    at::Tensor fst = at::Tensor();  \
    at::Tensor snd = at::Tensor();  \
    CATCH(  \
        std::tie(fst,snd) = expr;  \
    );  \
    res1 = ResultTensor(fst); \
    res2 = ResultTensor(snd);     

#define CATCH_SCALAR(expr) \
    at::Scalar res = at::Scalar(); \
    CATCH(  \
        res = expr;  \
    );  \
    return ResultTensor(res);

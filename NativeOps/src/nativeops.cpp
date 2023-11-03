#include "c10/util/Optional.h"
#include "utils.hpp"
#include "ATen/Context.h"

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
torch_scaled_dot_product_attention(
    const Tensor query, const Tensor key, const Tensor value, const Tensor mask,
    const double dropout_p, const bool is_causal,
    bool use_flash, bool use_mem_effecient, bool use_math)
{
    auto& ctx = at::globalContext();
    ctx.setSDPUseFlash(use_flash);
    ctx.setSDPUseMemEfficient(use_mem_effecient);
    ctx.setSDPUseMath(use_math);
    c10::optional<torch::Tensor> _mask;
    if (mask) _mask = *mask;
    CATCH_TENSOR(
        torch::scaled_dot_product_attention(*query, *key, *value, _mask, dropout_p, is_causal)
    );
}

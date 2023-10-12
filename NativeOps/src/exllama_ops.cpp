#include "utils.hpp"

#include "../third-party/AutoGPTQ/autogptq_cuda/exllama/cuda_func/q4_matrix.cuh"
#include "../third-party/AutoGPTQ/autogptq_cuda/exllama/cuda_func/q4_matmul.cuh"


EXPORT_API(void)
exllama_q4_matmul_cuda(const Tensor input, const Tensor qweight, const Tensor scales, const Tensor qzeros, Tensor output)
{
    CATCH(
        TORCH_CHECK(input->dtype() == torch::kFloat16);
        TORCH_CHECK(input->dim() == 2);
        TORCH_CHECK(output->dtype() == torch::kFloat16);
        TORCH_CHECK(output->dim() == 2);

        TORCH_CHECK(qweight->dtype() == torch::kInt32);
        TORCH_CHECK(qzeros->dtype() == torch::kInt32);
        TORCH_CHECK(scales->dtype() == torch::kFloat16);

        TORCH_CHECK(qweight->size(1) == qzeros->size(1) * 8);
        TORCH_CHECK(qweight->size(1) == scales->size(1));
        TORCH_CHECK(qzeros->size(0) == scales->size(0));

        // gptq packs elements on dim 0
        int in_dim = qweight->size(0) * 8;
        int out_dim = qweight->size(1);
        int groups = qzeros->size(0);
        
        TORCH_CHECK(input->size(0) == output->size(0));
        TORCH_CHECK(input->size(1) == in_dim);
        TORCH_CHECK(output->size(1) == out_dim);

        Q4Matrix ex_mat = Q4Matrix(
            in_dim, out_dim, groups,
            (uint32_t *) qweight->data_ptr(),
            (uint32_t *) qzeros->data_ptr(),
            (half *) scales->data_ptr(),
            nullptr,
            qweight->get_device()
        );

        ExLlamaTuning tuningParams;
        q4_matmul_cuda(
            &tuningParams,
            (half *) input->data_ptr(),
            input->size(0),
            &ex_mat,
            (half *) output->data_ptr()
        );
    );
}

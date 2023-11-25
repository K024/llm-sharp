#include "utils.hpp"

#include "../third-party/lmdeploy/gemm_s_f16/format.h"
#include "../third-party/lmdeploy/gemm_s_f16/gemm_s4_f16.h"
#include "../third-party/lmdeploy/llama/llama_kernels.h"

#include <c10/cuda/CUDAStream.h>


// input: [n, k]
// qweight: [k, m // 8] but permuted in lmdeploy
// scales_and_zeros: [k / group_size, m, [zero + 64;scale]]
EXPORT_API(Tensor)
turbomind_gemm_s4_f16(const Tensor input, const Tensor qweight, const Tensor scales_and_zeros, int group_size)
{
    static turbomind::GemmS4F16 gemm_instance;
    torch::Tensor res;
    CATCH(
        TORCH_CHECK(input->dtype() == torch::kHalf);
        TORCH_CHECK(qweight->dtype() == torch::kInt32);
        TORCH_CHECK(scales_and_zeros->dtype() == torch::kHalf);

        TORCH_CHECK(input->is_contiguous());
        TORCH_CHECK(qweight->is_contiguous());
        TORCH_CHECK(scales_and_zeros->is_contiguous());

        TORCH_CHECK(input->dim() == 2);
        TORCH_CHECK(qweight->dim() == 2);
        TORCH_CHECK(scales_and_zeros->dim() == 3);

        int m = qweight->size(1) * 8;
        int n = input->size(0);
        int k = input->size(1);

        TORCH_CHECK(qweight->size(0) == k);
        TORCH_CHECK(scales_and_zeros->size(0) == (k + group_size - 1) / group_size);
        TORCH_CHECK(scales_and_zeros->size(1) == m);
        TORCH_CHECK(scales_and_zeros->size(2) == 2);

        torch::Tensor output = torch::empty(
            {n, m},
            torch::TensorOptions()
                .dtype(torch::kHalf).device(input->device()));

        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input->device().index());

        gemm_instance.Run(
            (half *)output.data_ptr(),
            (uint *)qweight->data_ptr(),
            (half *)input->data_ptr(),
            (half2 *)scales_and_zeros->data_ptr(),
            m, n, k, group_size,
            turbomind::GemmS4F16::kGemm,
            -1, // estimate best kernel
            stream);

        res = std::move(output);
    );
    return ResultTensor(res);
}

EXPORT_API(void)
turbomind_convert_s4_k_m8(Tensor qweight_dst, Tensor scale_and_zeros_dst, const Tensor qweight, const Tensor scales, const Tensor qzeros, int group_size)
{
    CATCH(
        TORCH_CHECK(qweight->dtype() == torch::kInt32);
        TORCH_CHECK(qweight_dst->dtype() == torch::kInt32);
        TORCH_CHECK(qzeros->dtype() == torch::kInt32);
        TORCH_CHECK(scale_and_zeros_dst->dtype() == torch::kHalf);
        TORCH_CHECK(qweight->is_contiguous());
        TORCH_CHECK(scales->is_contiguous());
        TORCH_CHECK(qzeros->is_contiguous());

        int m = qweight->size(1) * 8;
        int k = qweight->size(0);

        torch::Tensor workspace = torch::zeros_like(*scales);

        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(qweight->device().index());

        turbomind::convert_s4_k_m8(
            (uint32_t *)qweight_dst->data_ptr(),
            (half2 *)scale_and_zeros_dst->data_ptr(),
            (half *)workspace.data_ptr(),
            (uint32_t *)qweight->data_ptr(),
            (half *)scales->data_ptr(),
            (uint32_t *)qzeros->data_ptr(),
            m, k, group_size, stream);
    )
}

EXPORT_API(Tensor)
turbomind_rms_norm(const Tensor input, const Tensor scale, float eps)
{
    torch::Tensor res;
    CATCH(
        TORCH_CHECK(input->dtype() == torch::kHalf);
        TORCH_CHECK(scale->dtype() == torch::kHalf);

        TORCH_CHECK(input->is_contiguous());
        TORCH_CHECK(scale->is_contiguous());

        TORCH_CHECK(input->dim() == 2);
        TORCH_CHECK(scale->dim() == 1);
        
        int m = input->size(0);
        int n = input->size(1);
        TORCH_CHECK(scale->size(0) == n);

        torch::Tensor output = torch::empty(
            {n, m},
            torch::TensorOptions()
                .dtype(torch::kHalf).device(input->device()));

        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input->device().index());

        turbomind::invokeRootMeanSquareNorm(
            (half *)output.data_ptr(),
            (half *)input->data_ptr(),
            (half *)scale->data_ptr(),
            eps, m, n, stream);

        res = std::move(output);
    );
    return ResultTensor(res);
}

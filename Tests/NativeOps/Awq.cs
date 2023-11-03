using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;
using TorchSharp;

namespace llm_sharp.Tests;

using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;

[TestClass]
public partial class NativeOpsTests
{

    [TestInitialize]
    public void Init()
    {
        LibTorchLoader.EnsureLoaded();
    }

    [TestMethod]
    public void NativeOps_ShouldWork()
    {
        var result = Ops.hello(torch.ones(2, 3, 4, device: torch.CUDA));

        Assert.IsTrue((result == 2).all().item<bool>());
    }

    [TestMethod]
    public void ExtraGemmOps_ShouldHaveSimilarResults()
    {
        Tensor pack_u4(Tensor x, int[]? order = null)
        {
            var (dim1, dim2) = x.shape;
            var packed = torch.zeros(new[] { dim1, (dim2 + 7) / 8 }, dtype: torch.int32, device: x.device);
            for (int n = 0; n < 8; n++)
            {
                var i = order?[n] ?? n;
                packed[TensorIndex.Colon] |= (x[TensorIndex.Colon, TensorIndex.Slice(i, null, 8)] & 0xF).bitwise_left_shift(n * 4);
            }
            return packed;
        }

        Tensor unpack_u4(Tensor x, int[]? order = null)
        {
            var (dim1, dim2) = x.shape;
            var unpacked = torch.zeros(new[] { dim1, dim2 * 8 }, dtype: torch.int32, device: x.device);
            for (int n = 0; n < 8; n++)
            {
                var i = order?[n] ?? n;
                unpacked[TensorIndex.Colon, TensorIndex.Slice(i, null, 8)] = x.bitwise_right_shift(n * 4) & 0xF;
            }
            return unpacked;
        }

        (Tensor x, Tensor zeros, Tensor scales) quant_u4(Tensor x, int group_size = 128, bool symmetric = false)
        {
            var (out_dim, in_dim) = x.shape;
            x = x.reshape(out_dim, in_dim / group_size, group_size);
            var max = x.max(-1, keepdim: true).values;
            var min = x.min(-1, keepdim: true).values;

            if (symmetric)
            {
                max = torch.max(max, -min);
                min = -max;
            }

            var scales = torch.clamp(max - min, min: 1e-10) / 15;
            var zeros = torch.nn.functional.relu(-min);

            x = torch.clamp(torch.round((x + zeros) / scales + 0.5), 0, 15).to(torch.int32).reshape(out_dim, in_dim);
            zeros = torch.clamp(torch.round(zeros / scales + 0.5), 0, 15).to(torch.int32).reshape(out_dim, in_dim / group_size);

            return (x, zeros, scales.reshape(out_dim, in_dim / group_size));
        }

        Tensor dequant_u4(Tensor x, Tensor zeros, Tensor scales, int group_size = 128)
        {
            var (out_dim, in_dim) = x.shape;
            x = x.reshape(out_dim, in_dim / group_size, group_size);
            zeros = zeros.reshape(out_dim, in_dim / group_size, 1);
            scales = scales.reshape(out_dim, in_dim / group_size, 1);
            return ((x - zeros) * scales).reshape(out_dim, in_dim);
        }

        var x = torch.eye(10, 2048, dtype: torch.float16, device: torch.CUDA);
        var a = torch.randn(1024, 2048, dtype: torch.float16, device: torch.CUDA) / Math.Sqrt(2048);

        a[TensorIndex.Slice(1, null, null)] = 0;

        var (qa, qzeros, scales) = quant_u4(a, symmetric: true);
        var deq_a = dequant_u4(unpack_u4(pack_u4(qa)), qzeros, scales);

        Assert.IsTrue((a - deq_a).abs().max().to(torch.float32).item<float>() < 0.05);
        Assert.IsTrue((a - deq_a).abs().mean().to(torch.float32).item<float>() < 0.005);

        var reference = x.matmul(deq_a.t());

        var output = Ops.awq_gemmv2_forward(
            x,
            pack_u4(qa),
            scales,
            pack_u4(qzeros),
            128
        );

        Assert.IsTrue((reference - output).abs().max().to(torch.float32).item<float>() < 0.05);
        Assert.IsTrue((reference - output).abs().mean().to(torch.float32).item<float>() < 0.005);

        // // exllama kernel dropped
        // var output2 = Ops.exllama_q4_matmul_cuda(
        //     x,
        //     pack_u4(qa).T.contiguous(),
        //     scales.T.contiguous(),
        //     // GPTQ kernel adds 1 to zero points
        //     pack_u4(torch.maximum(qzeros.T - 1, 0)).contiguous()
        // );

        // Assert.IsTrue((reference - output2).abs().max().to(torch.float32).item<float>() < 0.05);
        // Assert.IsTrue((reference - output2).abs().mean().to(torch.float32).item<float>() < 0.005);

        var output3 = Ops.awq_gemm_forward(
            x,
            pack_u4(qa.T, order: new[] { 0, 2, 4, 6, 1, 3, 5, 7 }),
            scales.T.contiguous(),
            pack_u4(qzeros.T, order: new[] { 0, 2, 4, 6, 1, 3, 5, 7 })
        );

        Assert.IsTrue((reference - output3).abs().max().to(torch.float32).item<float>() < 0.05);
        Assert.IsTrue((reference - output3).abs().mean().to(torch.float32).item<float>() < 0.005);

        var (converted_q, converted_sz) = Ops.turbomind_convert_s4_k_m8(
            pack_u4(qa.T, order: new[] { 0, 2, 4, 6, 1, 3, 5, 7 }),
            scales.T.contiguous(),
            pack_u4(qzeros.T, order: new[] { 0, 2, 4, 6, 1, 3, 5, 7 }),
            128
        );

        var output4 = Ops.turbomind_gemm_s4_f16(
            x,
            converted_q,
            converted_sz,
            128
        );

        Assert.IsTrue((reference - output4).abs().max().to(torch.float32).item<float>() < 0.05);
        Assert.IsTrue((reference - output4).abs().mean().to(torch.float32).item<float>() < 0.005);
    }
}

using llm_sharp.LLM.Utils;
using TorchSharp;

namespace llm_sharp.Tests;

using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using ops = NativeOps.NativeOps;

[TestClass]
public class NativeOpsTests
{

    [TestInitialize]
    public void Init()
    {
        LibTorchLoader.EnsureLoaded();
    }

    [TestMethod]
    public void NativeOps_ShouldWork()
    {
        var result = ops.hello(torch.ones(2, 3, 4, device: torch.CUDA));

        Assert.IsTrue((result == 2).all().item<bool>());
    }

    [TestMethod]
    public void ExtraOps_ShouldHaveSimilarResults()
    {
        Tensor pack_u4(Tensor x)
        {
            var (dim1, dim2) = x.shape;
            var packed = torch.zeros(new[] { dim1, (dim2 + 7) / 8 }, dtype: torch.int32, device: x.device);
            for (int n = 0; n < 8; n++)
                packed[TensorIndex.Colon] |= (x[TensorIndex.Colon, TensorIndex.Slice(n, null, 8)] & 0xF).bitwise_left_shift(n * 4);
            return packed;
        }

        Tensor unpack_u4(Tensor x)
        {
            var (dim1, dim2) = x.shape;
            var unpacked = torch.zeros(new[] { dim1, dim2 * 8 }, dtype: torch.int32, device: x.device);
            for (int n = 0; n < 8; n++)
                unpacked[TensorIndex.Colon, TensorIndex.Slice(n, null, 8)] = x.bitwise_right_shift(n * 4) & 0xF;
            return unpacked;
        }

        (Tensor x, Tensor zeros, Tensor scales) quant_u4(Tensor x, int group_size = 128)
        {
            var (out_dim, in_dim) = x.shape;
            x = x.reshape(out_dim, in_dim / group_size, group_size);
            var max = x.max(-1, keepdim: true).values;
            var min = x.min(-1, keepdim: true).values;

            var scales = torch.clamp(max - min, min: 1e-10) / 15;
            var zeros = torch.nn.functional.relu(-min);

            x = torch.clamp(torch.round((x + zeros) / scales), 0, 15).to(torch.int32).reshape(out_dim, in_dim);
            zeros = torch.clamp(torch.round(zeros / scales), 0, 15).to(torch.int32).reshape(out_dim, in_dim / group_size);

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

        var x = torch.randn(10, 2048, dtype: torch.float16, device: torch.CUDA);
        var a = torch.randn(1024, 2048, dtype: torch.float16, device: torch.CUDA) / Math.Sqrt(2048);

        var (qa, qzeros, scales) = quant_u4(a);
        var deq_a = dequant_u4(unpack_u4(pack_u4(qa)), qzeros, scales);

        Assert.IsTrue((a - deq_a).abs().max().to(torch.float32).item<float>() < 0.05);
        Assert.IsTrue((a - deq_a).abs().mean().to(torch.float32).item<float>() < 0.005);

        var reference = x.matmul(deq_a.t());

        var output = ops.awq_gemmv2_forward(
            x,
            pack_u4(qa),
            scales,
            pack_u4(qzeros),
            128
        );

        Assert.IsTrue((reference - output).abs().max().to(torch.float32).item<float>() < 0.05);
        Assert.IsTrue((reference - output).abs().mean().to(torch.float32).item<float>() < 0.005);

        var output2 = ops.exllama_q4_matmul_cuda(
            x,
            pack_u4(qa).T.contiguous(),
            scales.T.contiguous(),
            // GPTQ kernel adds 1 to zero points
            pack_u4(torch.maximum(qzeros.T - 1, 0)).contiguous()
        );

        Assert.IsTrue((reference - output2).abs().max().to(torch.float32).item<float>() < 0.05);
        Assert.IsTrue((reference - output2).abs().mean().to(torch.float32).item<float>() < 0.005);
    }
}

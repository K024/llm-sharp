using llm_sharp.LLM.Layers;
using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;
using TorchSharp;

namespace llm_sharp.Tests;


public partial class NativeOpsTests
{
    [TestMethod]
    public void ExtraTurbomindOps_RMSNormShouldWork()
    {
        var norm = new RMSNorm(new[] { 1024L }, 1e-6, dtype: torch.float16, device: torch.CUDA);

        var x = torch.randn(20, 128, 1024, dtype: torch.float16, device: torch.CUDA) * 10;

        var expected = norm.call(x);

        var actual = Ops.turbomind_rms_norm(
            x,
            norm.weight,
            (float)norm.eps
        );

        Assert.IsTrue((expected - actual).abs().mean().to(torch.float32).item<float>() < 0.000005);
    }

    [TestMethod]
    public void ExtraAwqOps_RotaryShouldWork()
    {
        var rotary = new RotaryEmbedding(1024, 128, dtype: torch.float16, device: torch.CUDA);
        var positions = torch.arange(100, 120, device: torch.CUDA).unsqueeze(0);

        var (cos, sin) = rotary.forward(positions).weights.ToArray();

        // make strided tensors
        var qkv = torch.randn(1, 20, 40 * 128 * 3, dtype: torch.float16, device: torch.CUDA);
        var (query, key, _) = torch.split(qkv, new long[] { 40 * 128, 40 * 128, 40 * 128 }, dim: -1);

        query = query.view(1, 20, 40, 128);
        key = key.view(1, 20, 40, 128);

        var expected_q = RotaryEmbedding.apply_rotary_emb(query, (cos, sin));
        var expected_k = RotaryEmbedding.apply_rotary_emb(key, (cos, sin));

        var actual = Ops.awq_rotary_embedding_neox(query, key, cos, sin);

        Assert.IsTrue((expected_q - actual.query).abs().max().to(torch.float32).item<float>() < 0.000005);
        Assert.IsTrue((expected_k - actual.key).abs().max().to(torch.float32).item<float>() < 0.000005);
    }

    [TestMethod]
    public void TorchOps_ScaledDotProductAttentionShouldWork()
    {
        var query = torch.randn(8, 40, 1, 128, dtype: torch.float16, device: torch.CUDA);
        var key = torch.randn(8, 40, 10, 128, dtype: torch.float16, device: torch.CUDA);
        var value = torch.randn(8, 40, 10, 128, dtype: torch.float16, device: torch.CUDA);
        var mask = torch.zeros(8, 1, 1, 10, dtype: torch.float16, device: torch.CUDA);
        mask[4.., 0, 0, ..4] = float.NegativeInfinity;

        var scale = Math.Sqrt(query.shape[^1]);
        var qk = torch.matmul(query, key.transpose(-1, -2)) / scale + mask;
        var scores = torch.softmax(qk, dim: -1, dtype: torch.float32);
        var expected = torch.matmul(scores.type_as(qk), value);

        var actual = Ops.torch_scaled_dot_product_attention(query, key, value, mask, use_flash: false, use_mem_effecient: false);

        Assert.IsTrue((expected - actual).abs().mean().to(torch.float32).item<float>() < 0.0005);
        Assert.IsTrue((expected - actual).abs().max().to(torch.float32).item<float>() < 0.005);
    }

    [TestMethod]
    public void TorchOps_ScaledDotProductAttentionShouldWorkWithCausal()
    {
        var query = torch.randn(8, 40, 10, 128, dtype: torch.float16, device: torch.CUDA);
        var key = torch.randn(8, 40, 10, 128, dtype: torch.float16, device: torch.CUDA);
        var value = torch.randn(8, 40, 10, 128, dtype: torch.float16, device: torch.CUDA);
        var mask = torch.zeros(8, 1, 10, 10, dtype: torch.float16, device: torch.CUDA);

        var temp_mask = torch.ones(10, 10, dtype: torch.ScalarType.Bool, device: torch.CUDA).tril(diagonal: 0);
        mask.masked_fill_(temp_mask.logical_not().unsqueeze(0).unsqueeze(0), float.NegativeInfinity);

        var scale = Math.Sqrt(query.shape[^1]);
        var qk = torch.matmul(query, key.transpose(-1, -2)) / scale + mask;
        var scores = torch.softmax(qk, dim: -1, dtype: torch.float32);
        var expected = torch.matmul(scores.type_as(qk), value);

        var actual = Ops.torch_scaled_dot_product_attention(query, key, value, is_causal: true, use_math: false, use_mem_effecient: false);

        Assert.IsTrue((expected - actual).abs().mean().to(torch.float32).item<float>() < 0.0005);
        Assert.IsTrue((expected - actual).abs().max().to(torch.float32).item<float>() < 0.005);

        var actual2 = Ops.torch_scaled_dot_product_attention(query, key, value, is_causal: true, use_math: false, use_flash: false);

        Assert.IsTrue((expected - actual2).abs().mean().to(torch.float32).item<float>() < 0.0005);
        Assert.IsTrue((expected - actual2).abs().max().to(torch.float32).item<float>() < 0.005);
    }
}

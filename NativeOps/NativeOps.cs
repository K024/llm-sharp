using System.Runtime.InteropServices;
using TorchSharp;

namespace llm_sharp.NativeOps;

public static class Ops
{
    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr llm_sharp_check_last_err();

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr llm_sharp_hello(IntPtr handle);

    //////////////////////////////////

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_layernorm_forward(IntPtr input, IntPtr gamma, float eps);

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_gemm_forward(IntPtr in_feats, IntPtr kernel, IntPtr scaling_factors, IntPtr zeros, int split_k_iters);

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_gemmv2_forward(IntPtr in_feats, IntPtr kernel, IntPtr scaling_factors, IntPtr zeros, int group_size, int split_k_iters);

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_gemv_forward(IntPtr in_feats, IntPtr kernel, IntPtr scaling_factors, IntPtr zeros, int group_size);

    [DllImport("llm_sharp_ops")]
    internal static extern void awq_rotary_embedding_neox(IntPtr positions, IntPtr query, IntPtr key, int head_size, IntPtr cos_sin_cache);

    //////////////////////////////////

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr turbomind_gemm_s4_f16(IntPtr input, IntPtr qweight, IntPtr scales_and_zeros, int group_size);

    [DllImport("llm_sharp_ops")]
    internal static extern void turbomind_convert_s4_k_m8(IntPtr qweight_dst, IntPtr scale_and_zeros_dst, IntPtr qweight, IntPtr scales, IntPtr qzeros, int group_size);

    //////////////////////////////////

    public static void CheckForErrors()
    {
        var error = llm_sharp_check_last_err();
        if (error != IntPtr.Zero)
            throw new ExternalException(Marshal.PtrToStringAnsi(error));
    }

    public static torch.Tensor hello(this torch.Tensor tensor)
    {
        var result = llm_sharp_hello(tensor.Handle);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result);
    }

    public static torch.Tensor awq_layernorm_forward(torch.Tensor input, torch.Tensor gamma, float eps)
    {
        var result = awq_layernorm_forward(input.Handle, gamma.Handle, eps);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result);
    }

    /// <param name="in_feats">[batch, in_dim]</param>
    /// <param name="kernel">[in_dim, out_dim // 8]</param>
    /// <param name="scaling_factors">[in_dim // group_size, out_dim]</param>
    /// <param name="zeros">[in_dim // group_size, out_dim // 8]</param>
    /// <summary>
    /// pack_order: [0, 2, 4, 6, 1, 3, 5, 7] for each packed element
    /// </summary>
    public static torch.Tensor awq_gemm_forward(torch.Tensor in_feats, torch.Tensor kernel, torch.Tensor scaling_factors, torch.Tensor zeros)
    {
        var outputShape = in_feats.shape.SkipLast(1).Concat(new[] { kernel.shape[1] * 8 }).ToArray();
        in_feats = in_feats.reshape(-1, in_feats.shape[^1]);
        var split_k_iters = 8;
        var result = awq_gemm_forward(in_feats.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, split_k_iters);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    /// <param name="in_feats">[batch, in_dim]</param>
    /// <param name="kernel">[out_dim, in_dim // 8]</param>
    /// <param name="scaling_factors">[out_dim, in_dim // group_size]</param>
    /// <param name="zeros">[out_dim, in_dim // group_size // 8]</param>
    public static torch.Tensor awq_gemmv2_forward(torch.Tensor in_feats, torch.Tensor kernel, torch.Tensor scaling_factors, torch.Tensor zeros, int group_size)
    {
        var outputShape = in_feats.shape.SkipLast(1).Concat(new[] { kernel.shape[0] }).ToArray();
        in_feats = in_feats.reshape(-1, in_feats.shape[^1]);
        var split_k_iters = 8;
        var result = awq_gemmv2_forward(in_feats.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, group_size, split_k_iters);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    public static torch.Tensor awq_gemv_forward(torch.Tensor in_feats, torch.Tensor kernel, torch.Tensor scaling_factors, torch.Tensor zeros, int group_size)
    {
        var outputShape = in_feats.shape.SkipLast(1).Concat(new[] { kernel.shape[0] }).ToArray();
        in_feats = in_feats.reshape(-1, in_feats.shape[^1]);
        var result = awq_gemv_forward(in_feats.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, group_size);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    public static void awq_rotary_embedding_neox(torch.Tensor positions, torch.Tensor query, torch.Tensor key, int head_size, torch.Tensor cos_sin_cache)
    {
        if (query.requires_grad || key.requires_grad)
            throw new ArgumentException("query/key must not require grad");
        awq_rotary_embedding_neox(positions.Handle, query.Handle, key.Handle, head_size, cos_sin_cache.Handle);
        CheckForErrors();
    }

    /// <param name="input">[batch, in_dim]</param>
    /// <param name="qweight">[in_dim // 8, out_dim]</param>
    /// <param name="scales">[in_dim // group_size, out_dim]</param>
    /// <param name="qzeros">[in_dim // group_size, out_dim // 8]</param>
    public static torch.Tensor exllama_q4_matmul_cuda(torch.Tensor input, torch.Tensor qweight, torch.Tensor scales, torch.Tensor qzeros)
    {
        throw new NotImplementedException();
    }

    /// <param name="input">[n, k]</param>
    /// <param name="qweight">[k, m // 8] should be permuted with </param>
    /// <param name="scales_and_zeros">[k / group_size, m] with packed [zero: f16, scale: f16]</param>
    /// <summary>
    /// pack_order: [0, 2, 4, 6, 1, 3, 5, 7] for each packed element
    /// </summary>
    public static torch.Tensor turbomind_gemm_s4_f16(torch.Tensor input, torch.Tensor qweight, torch.Tensor scales_and_zeros, int group_size)
    {
        var outputShape = input.shape.SkipLast(1).Concat(new[] { qweight.shape[1] * 8 }).ToArray();
        input = input.reshape(-1, input.shape[^1]);
        var result = turbomind_gemm_s4_f16(input.Handle, qweight.Handle, scales_and_zeros.Handle, group_size);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    public static (torch.Tensor qweight, torch.Tensor scale_and_zeros) turbomind_convert_s4_k_m8(torch.Tensor qweight, torch.Tensor scales, torch.Tensor qzeros, int group_size)
    {
        if (qweight.device.type != DeviceType.CUDA)
            throw new ArgumentException("qweight must be on CUDA device");

        var qweight_dst = torch.empty_like(qweight);
        var scale_and_zeros_dst = torch.empty(scales.shape[0], scales.shape[1], 2, dtype: torch.float16, device: scales.device);

        turbomind_convert_s4_k_m8(qweight_dst.Handle, scale_and_zeros_dst.Handle, qweight.Handle, scales.Handle, qzeros.Handle, group_size);
        CheckForErrors();
        return (qweight_dst, scale_and_zeros_dst);
    }
}

using System.Runtime.InteropServices;
using TorchSharp;

namespace llm_sharp.NativeOps;

using Tensor = torch.Tensor;

public static class Ops
{
    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr llm_sharp_check_last_err();

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr llm_sharp_hello(IntPtr handle);

    [DllImport("llm_sharp_ops")]
    internal static extern void torch_empty_cache();

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr torch_scaled_dot_product_attention(
        IntPtr query, IntPtr key, IntPtr value, IntPtr mask,
        double dropout_p, bool is_causal,
        bool use_flash, bool use_mem_effecient, bool use_math);

    //////////////////////////////////

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_gemm_forward(IntPtr in_feats, IntPtr kernel, IntPtr scaling_factors, IntPtr zeros, int split_k_iters);

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_gemmv2_forward(IntPtr in_feats, IntPtr kernel, IntPtr scaling_factors, IntPtr zeros, int group_size, int split_k_iters);

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_gemv_forward(IntPtr in_feats, IntPtr kernel, IntPtr scaling_factors, IntPtr zeros, int group_size);

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr awq_rotary_embedding_neox(IntPtr cos, IntPtr sin, IntPtr query, IntPtr key, IntPtr out_query, IntPtr out_key);

    //////////////////////////////////

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr turbomind_gemm_s4_f16(IntPtr input, IntPtr qweight, IntPtr scales_and_zeros, int group_size);

    [DllImport("llm_sharp_ops")]
    internal static extern void turbomind_convert_s4_k_m8(IntPtr qweight_dst, IntPtr scale_and_zeros_dst, IntPtr qweight, IntPtr scales, IntPtr qzeros, int group_size);

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr turbomind_rms_norm(IntPtr input, IntPtr scale, float eps);

    //////////////////////////////////

    public static void CheckForErrors()
    {
        var error = llm_sharp_check_last_err();
        if (error != IntPtr.Zero)
            throw new ExternalException(Marshal.PtrToStringAnsi(error));
    }

    public static Tensor hello(this Tensor tensor)
    {
        var result = llm_sharp_hello(tensor.Handle);
        CheckForErrors();
        return Tensor.UnsafeCreateTensor(result);
    }

    public static void cuda_empty_cache()
    {
        torch_empty_cache();
        CheckForErrors();
    }

    public static Tensor torch_scaled_dot_product_attention(
        Tensor query, Tensor key, Tensor value, Tensor? mask = null,
        double dropout_p = 0.0, bool is_causal = false,
        bool use_flash = true, bool use_mem_effecient = true, bool use_math = true)
    {
        if (!use_flash && !use_mem_effecient && !use_math)
            throw new ArgumentException("At least one of use_flash, use_mem_effecient, use_math must be true");

        var result = torch_scaled_dot_product_attention(
            query.Handle, key.Handle, value.Handle, mask?.Handle ?? IntPtr.Zero,
            dropout_p, is_causal,
            use_flash, use_mem_effecient, use_math);

        CheckForErrors();
        return Tensor.UnsafeCreateTensor(result);
    }

    private static (Tensor reshaped, long[] outputShape, long[] outputGemmShape) reshape_for_gemm(Tensor input, long out_dim)
    {
        var batch = input.shape.SkipLast(1).Aggregate(1L, (a, b) => a * b);
        var outputShape = input.shape.SkipLast(1).Concat(new[] { out_dim }).ToArray();
        var outputGemmShape = new[] { batch, out_dim };
        input = input.reshape(batch, -1);
        return (input, outputShape, outputGemmShape);
    }

    /// <param name="in_feats">[batch, in_dim]</param>
    /// <param name="kernel">[in_dim, out_dim // 8]</param>
    /// <param name="scaling_factors">[in_dim // group_size, out_dim]</param>
    /// <param name="zeros">[in_dim // group_size, out_dim // 8]</param>
    /// <summary>
    /// pack_order: [0, 2, 4, 6, 1, 3, 5, 7] for each packed element
    /// </summary>
    public static Tensor awq_gemm_forward(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros)
    {
        var split_k_iters = 8;
        var (reshaped, outputShape, _) = reshape_for_gemm(in_feats, kernel.shape[1] * 8);
        var result = awq_gemm_forward(reshaped.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, split_k_iters);
        CheckForErrors();
        return Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    /// <param name="in_feats">[batch, in_dim]</param>
    /// <param name="kernel">[out_dim, in_dim // 8]</param>
    /// <param name="scaling_factors">[out_dim, in_dim // group_size]</param>
    /// <param name="zeros">[out_dim, in_dim // group_size // 8]</param>
    public static Tensor awq_gemmv2_forward(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, int group_size)
    {
        var split_k_iters = 8;
        var (reshaped, outputShape, _) = reshape_for_gemm(in_feats, kernel.shape[0]);
        var result = awq_gemmv2_forward(reshaped.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, group_size, split_k_iters);
        CheckForErrors();
        return Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    public static Tensor awq_gemv_forward(Tensor in_feats, Tensor kernel, Tensor scaling_factors, Tensor zeros, int group_size)
    {
        var (reshaped, outputShape, _) = reshape_for_gemm(in_feats, kernel.shape[0]);
        var result = awq_gemv_forward(reshaped.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, group_size);
        CheckForErrors();
        return Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    /// <param name="cos">[batch, seq, heads, head_dim]</param>
    /// <param name="sin">[batch, seq, heads, head_dim]</param>
    /// <param name="query">[batch, seq, (1), rotary_dim * 2]</param>
    /// <param name="key">[batch, seq, (1), rotary_dim * 2]</param>
    public static (Tensor query, Tensor key) awq_rotary_embedding_neox(Tensor query, Tensor key, Tensor cos, Tensor sin)
    {
        var out_query = torch.empty_like(query);
        var out_key = torch.empty_like(key);
        awq_rotary_embedding_neox(cos.Handle, sin.Handle, query.Handle, key.Handle, out_query.Handle, out_key.Handle);
        CheckForErrors();
        return (out_query, out_key);
    }

    /// <param name="input">[batch, in_dim]</param>
    /// <param name="qweight">[in_dim // 8, out_dim]</param>
    /// <param name="scales">[in_dim // group_size, out_dim]</param>
    /// <param name="qzeros">[in_dim // group_size, out_dim // 8]</param>
    public static Tensor exllama_q4_matmul_cuda(Tensor input, Tensor qweight, Tensor scales, Tensor qzeros)
    {
        // left for shape information
        throw new NotImplementedException();
    }

    /// <param name="input">[n, k]</param>
    /// <param name="qweight">[k, m // 8] should be permuted with convert_s4_k_m8</param>
    /// <param name="scales_and_zeros">[k / group_size, m] with packed [zero: f16, scale: f16]</param>
    /// <summary>
    /// pack_order: [0, 2, 4, 6, 1, 3, 5, 7] for each packed element
    /// </summary>
    public static Tensor turbomind_gemm_s4_f16(Tensor input, Tensor qweight, Tensor scales_and_zeros, int group_size)
    {
        var (reshaped, outputShape, _) = reshape_for_gemm(input, qweight.shape[1] * 8);
        var result = turbomind_gemm_s4_f16(reshaped.Handle, qweight.Handle, scales_and_zeros.Handle, group_size);
        CheckForErrors();
        return Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }

    public static (Tensor qweight, Tensor scale_and_zeros) turbomind_convert_s4_k_m8(Tensor qweight, Tensor scales, Tensor qzeros, int group_size)
    {
        if (qweight.device.type != DeviceType.CUDA)
            throw new ArgumentException("qweight must be on CUDA device");

        var qweight_dst = torch.empty_like(qweight, device: qweight.device);
        var scale_and_zeros_dst = torch.empty(scales.shape[0], scales.shape[1], 2, dtype: torch.float16, device: scales.device);

        turbomind_convert_s4_k_m8(qweight_dst.Handle, scale_and_zeros_dst.Handle, qweight.Handle, scales.Handle, qzeros.Handle, group_size);
        CheckForErrors();
        return (qweight_dst, scale_and_zeros_dst);
    }

    public static Tensor turbomind_rms_norm(Tensor input, Tensor scale, float eps)
    {
        var (reshaped, outputShape, _) = reshape_for_gemm(input, scale.shape[0]);
        var result = turbomind_rms_norm(reshaped.Handle, scale.Handle, eps);
        CheckForErrors();
        return Tensor.UnsafeCreateTensor(result).reshape(outputShape);
    }
}

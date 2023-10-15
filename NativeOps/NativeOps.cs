using System.Runtime.InteropServices;
using TorchSharp;

namespace llm_sharp.NativeOps;

public static class Ops
{
    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr llm_sharp_check_last_err();

    [DllImport("llm_sharp_ops")]
    internal static extern IntPtr llm_sharp_hello(IntPtr handle);


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

    [DllImport("llm_sharp_ops")]
    internal static extern void exllama_q4_matmul_cuda(IntPtr input, IntPtr qweight, IntPtr scales, IntPtr qzeros, IntPtr output);


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
        var outputShape = input.shape.SkipLast(1).Concat(new[] { qweight.shape[1] }).ToArray();
        input = input.reshape(-1, input.shape[^1]);
        var output = torch.empty(new[] { input.shape[0], qweight.shape[1] }, dtype: input.dtype, device: input.device);
        exllama_q4_matmul_cuda(input.Handle, qweight.Handle, scales.Handle, qzeros.Handle, output.Handle);
        CheckForErrors();
        return output.reshape(outputShape);
    }
}

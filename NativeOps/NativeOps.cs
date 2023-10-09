using System.Runtime.InteropServices;
using TorchSharp;

namespace llm_sharp.NativeOps;

public static class NativeOps
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

    public static torch.Tensor awq_gemm_forward(torch.Tensor in_feats, torch.Tensor kernel, torch.Tensor scaling_factors, torch.Tensor zeros, int split_k_iters)
    {
        var result = awq_gemm_forward(in_feats.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, split_k_iters);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result);
    }

    public static torch.Tensor awq_gemmv2_forward(torch.Tensor in_feats, torch.Tensor kernel, torch.Tensor scaling_factors, torch.Tensor zeros, int group_size, int split_k_iters)
    {
        var result = awq_gemmv2_forward(in_feats.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, group_size, split_k_iters);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result);
    }

    public static torch.Tensor awq_gemv_forward(torch.Tensor in_feats, torch.Tensor kernel, torch.Tensor scaling_factors, torch.Tensor zeros, int group_size)
    {
        var result = awq_gemv_forward(in_feats.Handle, kernel.Handle, scaling_factors.Handle, zeros.Handle, group_size);
        CheckForErrors();
        return torch.Tensor.UnsafeCreateTensor(result);
    }

    public static void awq_rotary_embedding_neox(torch.Tensor positions, torch.Tensor query, torch.Tensor key, int head_size, torch.Tensor cos_sin_cache)
    {
        awq_rotary_embedding_neox(positions.Handle, query.Handle, key.Handle, head_size, cos_sin_cache.Handle);
        CheckForErrors();
    }
}

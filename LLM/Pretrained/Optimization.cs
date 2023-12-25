using TorchSharp;

namespace llm_sharp.LLM.Pretrained;

public record OptimizationConfig
{
    public static OptimizationConfig current { get; set; } = new OptimizationConfig();

    public bool auto_empty_cache { get; set; } = true;

    public bool use_faster_kv_cache { get; set; } = true;

    public bool fuse_layer_norm { get; set; } = true;

    public bool fuse_attention { get; set; } = true;

    public bool fuse_rotary_embedding { get; set; } = true;

    public bool enable_turbomind_gemm { get; set; } = true;

    public static void set_no_native_ops()
    {
        current.auto_empty_cache = false;
        // current.use_faster_kv_cache = false;
        current.fuse_layer_norm = false;
        current.fuse_attention = false;
        current.fuse_rotary_embedding = false;
        current.enable_turbomind_gemm = false;
    }
}

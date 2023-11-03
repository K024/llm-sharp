using TorchSharp;
using llm_sharp.LLM.Layers;
using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;

namespace llm_sharp.LLM.Models;

using Tensor = torch.Tensor;

public class FusedLlamaAttention : LlamaAttention
{
    protected double dropout_rate;
    public FusedLlamaAttention(LlamaBuilder builder) : base(builder)
    {
        dropout_rate = builder.config.dropout_rate;
    }

    public override (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache) forward(
        Tensor x,
        (Tensor cos, Tensor sin) freqs_cis,
        Tensor? attention_mask,
        (Tensor k_cache, Tensor v_cache)? kv_cache)
    {
        using var scope = torch.NewDisposeScope();

        var (n_batch, n_seq, _) = x.shape;

        var splitSizes = new[] { d_head * n_head, d_head * n_kv_head, d_head * n_kv_head };
        var (q, k, v) = torch.split(qkv_proj.call(x), splitSizes, dim: -1);

        q = q.view(n_batch, n_seq, n_head, d_head);
        k = k.view(n_batch, n_seq, n_kv_head, d_head);
        v = v.view(n_batch, n_seq, n_kv_head, d_head);

        (q, k) = RotaryEmbedding.apply_rotary_emb_fused(q, k, freqs_cis);

        if (kv_cache is not null)
        {
            var (k_cache, v_cache) = kv_cache.Value;
            k = torch.cat(new[] { k_cache, k }, dim: 1);
            v = torch.cat(new[] { v_cache, v }, dim: 1);
        }
        kv_cache = (k.detach(), v.detach());

        q = q.permute(0, 2, 1, 3);
        k = k.permute(0, 2, 1, 3);
        v = v.permute(0, 2, 1, 3);

        if (n_groups > 1)
        {
            k = repeat_kv(k, n_groups);
            v = repeat_kv(v, n_groups);
        }

        var output = Ops.torch_scaled_dot_product_attention(
            q, k, v, attention_mask,
            dropout_p: training ? dropout_rate : 0
        );

        output = output.permute(0, 2, 1, 3).reshape(n_batch, n_seq, -1);
        output = o_proj.call(output);

        return (
            scope.MoveToOuter(output),
            (
                scope.MoveToOuter(kv_cache.Value.k_cache),
                scope.MoveToOuter(kv_cache.Value.v_cache)
            )
        );
    }
}

class LlamaFastBuilder : LlamaBuilder
{
    public LlamaFastBuilder(LlamaConfig config) : base(config) { }

    public override torch.nn.Module<
        Tensor, (Tensor cos, Tensor sin), Tensor?, (Tensor k_cache, Tensor v_cache)?,
        (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache)>
        create_llama_attention()
        => new FusedLlamaAttention(this);

    public override torch.nn.Module<Tensor, Tensor> create_ln()
        => new FusedRMSNorm(new[] { config.hidden_size }, config.layernorm_epsilon, dtype: dtype, device: device);
}

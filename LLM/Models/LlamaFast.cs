using TorchSharp;
using llm_sharp.LLM.Layers;
using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;
using llm_sharp.LLM.Pretrained;
using llm_sharp.LLM.Tokenizers;

namespace llm_sharp.LLM.Models;

using Tensor = torch.Tensor;

public class FusedLlamaAttention : LlamaAttention
{
    protected double dropout_rate;
    public FusedLlamaAttention(LlamaBuilder builder) : base(builder)
    {
        dropout_rate = builder.config.dropout_rate;
    }

    public override Tensor forward(
        Tensor x,
        Tensor? attention_mask,
        IRotary? freqs_cis,
        IKvCache? kv_cache)
    {
        using var scope = torch.NewDisposeScope();

        var (n_batch, n_seq, _) = x.shape;

        var splitSizes = new[] { d_head * n_head, d_head * n_kv_head, d_head * n_kv_head };
        var (q, k, v) = torch.split(qkv_proj.call(x), splitSizes, dim: -1);

        q = q.view(n_batch, n_seq, n_head, d_head);
        k = k.view(n_batch, n_seq, n_kv_head, d_head);
        v = v.view(n_batch, n_seq, n_kv_head, d_head);

        if (freqs_cis is not null)
            (q, k) = freqs_cis.apply(q, k);

        if (kv_cache is not null)
            (k, v) = kv_cache.update(k, v);

        q = q.permute(0, 2, 1, 3);
        k = k.permute(0, 2, 1, 3);
        v = v.permute(0, 2, 1, 3);

        if (n_groups > 1)
        {
            k = repeat_kv(k, n_groups);
            v = repeat_kv(v, n_groups);
        }

        var is_causal = false;
        if (attention_mask is null ||
            attention_mask.shape[0] == 1 /* batch */ &&
            attention_mask.shape[1] == 1 /* head, alibi mask is not 1 */ )
        {
            // is default causal mask or generative mask
            attention_mask = null;
            // on equal seq len, enable internal causal
            if (q.shape[2] == k.shape[2])
                is_causal = true;
        }

        var output = Ops.torch_scaled_dot_product_attention(
            q, k, v,
            attention_mask,
            is_causal: is_causal,
            dropout_p: training ? dropout_rate : 0
        );

        output = output.permute(0, 2, 1, 3).reshape(n_batch, n_seq, -1);
        output = o_proj.call(output);

        return scope.MoveToOuter(output);
    }
}

class LlamaFastBuilder : LlamaBuilder
{
    public LlamaFastBuilder(LlamaConfig config) : base(config) { }

    public override torch.nn.Module<Tensor, Tensor?, IRotary?, IKvCache?, Tensor>
        create_llama_attention()
        => OptimizationConfig.current.fuse_attention
            ? new FusedLlamaAttention(this)
            : base.create_llama_attention();

    public override torch.nn.Module<Tensor, Tensor> create_ln()
        => OptimizationConfig.current.fuse_layer_norm
            ? new FusedRMSNorm(new[] { config.hidden_size }, config.layernorm_epsilon, dtype: dtype, device: device)
            : base.create_ln();

    public override torch.nn.Module<Tensor, IRotary> create_rotary_embedding()
        => OptimizationConfig.current.fuse_rotary_embedding
            ? new FastRotaryEmbedding(config.vocab_size, config.head_hidden_size, theta: config.rope_theta, dtype: dtype, device: device)
            : base.create_rotary_embedding();

    public override List<IKvCache> create_kv_cache(long batch_size)
        => OptimizationConfig.current.use_faster_kv_cache
            ? Enumerable.Range(0, config.num_layers)
                .Select(_ => (IKvCache)new FastKvCache(batch_size, config.num_key_value_heads, config.head_hidden_size, device, dtype))
                .ToList()
            : base.create_kv_cache(batch_size);
}

class LlamaAwqBuilder : LlamaFastBuilder
{
    public LlamaAwqBuilder(LlamaConfig config) : base(config) { }

    private bool creating_block = false;

    public override torch.nn.Module<Tensor, Tensor?, IRotary?, IKvCache?, Tensor> create_llama_block()
    {
        try
        {
            creating_block = true;
            return base.create_llama_block();
        }
        finally
        {
            creating_block = false;
        }
    }

    public override torch.nn.Module<Tensor, Tensor> create_linear(long input_size, long output_size, bool hasBias = true)
    {
        if (!creating_block)
            return base.create_linear(input_size, output_size, hasBias);
        return new AwqLinear(input_size, output_size, hasBias, dtype, device);
    }
}

public class LlamaAwq : Llama
{
    public static void convert_turbomind(torch.nn.Module module)
    {
        if (!OptimizationConfig.current.enable_turbomind_gemm)
            return;

        Console.WriteLine("Converting to TurboMind format...");
        foreach (var submodule in module.modules())
        {
            if (submodule is AwqLinear awq)
                awq.convert_turbomind();
        }
    }

    public static new Llama from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var (model, model_config) = model_from_pretrained<LlamaModel, LlamaConfig, LlamaAwqBuilder>(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained<SentencePieceBPE, SentencePieceBPEConfig>(path);

        convert_turbomind(model);

        return new LlamaAwq()
        {
            device = device,
            model = model,
            model_config = model_config,
            tokenizer = tokenizer,
            tokenizer_config = tokenizer_config,
        };
    }
}

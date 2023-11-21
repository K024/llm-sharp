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

    public override (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache) forward(
        Tensor x,
        (Tensor cos, Tensor sin)? freqs_cis,
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

        if (freqs_cis is not null)
            (q, k) = RotaryEmbedding.apply_rotary_emb_fused(q, k, freqs_cis.Value);

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

public class Llama : GenerativeLM<Llama.LlamaState>
{
    public class LlamaState : IDisposable
    {
        public List<(Tensor k_cache, Tensor v_cache)> past_key_values { get; set; } = new();

        public void Dispose()
        {
            past_key_values.SelectMany(x => new[] { x.k_cache, x.v_cache })
                .ToList().ForEach(x => x.Dispose());
        }
    }

    public torch.Device? device { get; protected set; }
    public LlamaModel model { get; init; }
    public LlamaConfig model_config { get; init; }
    public SentencePieceBPE tokenizer { get; init; }
    public SentencePieceBPEConfig tokenizer_config { get; init; }

#nullable disable
    protected Llama() { }
#nullable restore

    public virtual Llama to(torch.Device? device)
    {
        model.to(device);
        this.device = device;
        return this;
    }

    public static Llama from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var (model, model_config) = model_from_pretrained<LlamaModel, LlamaConfig, LlamaBuilder>(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained<SentencePieceBPE, SentencePieceBPEConfig>(path);
        return new Llama()
        {
            device = device,
            model = model,
            model_config = model_config,
            tokenizer = tokenizer,
            tokenizer_config = tokenizer_config,
        };
    }

    protected override List<int> prepare_input(List<(string query, string answer)> history, string input)
    {
        var prompt = "";

        foreach (var (query, answer) in history)
        {
            prompt += $"<s>[INST] {query} [/INST] {answer} </s>";
        }
        prompt += $"<s>[INST] {input} [/INST]";

        return tokenizer.encode_text(prompt);
    }

    protected override (int next_token, LlamaState? state) generate_step(List<int> tokens, LlamaState? state, GenerationConfig config)
    {
        using var scope = torch.NewDisposeScope();

        var past_key_values = state?.past_key_values;
        if (past_key_values is not null)
            tokens = tokens.TakeLast(1).ToList();

        var input_ids = torch.tensor(tokens, dtype: torch.int64, device: device).unsqueeze(0);

        model.eval();
        var output = model.call(new()
        {
            input_ids = input_ids,
            past_key_values = past_key_values,
        });

        var next = top_p_top_k_sampling(
            output.logits[0, ^1], config.top_k, config.top_p, config.temperature
        );
        var next_token = (int)next.item<long>();

        scope.MoveToOuter(output.current_key_values.SelectMany(x => new[] { x.k_cache, x.v_cache }));
        return (
            next_token,
            new() { past_key_values = output.current_key_values }
        );
    }

    protected override string decode_output(List<int> tokens)
    {
        return tokenizer.decode_text(tokens);
    }

    protected override List<int> get_eos_tokens()
    {
        return tokenizer.eos_ids;
    }
}

class LlamaFastBuilder : LlamaBuilder
{
    public LlamaFastBuilder(LlamaConfig config) : base(config) { }

    public override torch.nn.Module<
        Tensor, (Tensor cos, Tensor sin)?, Tensor?, (Tensor k_cache, Tensor v_cache)?,
        (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache)>
        create_llama_attention()
        => new FusedLlamaAttention(this);

    public override torch.nn.Module<Tensor, Tensor> create_ln()
        => new FusedRMSNorm(new[] { config.hidden_size }, config.layernorm_epsilon, dtype: dtype, device: device);
}

class LlamaAwqBuilder : LlamaFastBuilder
{
    public LlamaAwqBuilder(LlamaConfig config) : base(config) { }

    private bool creating_block = false;

    public override
        torch.nn.Module<Tensor, (Tensor cos, Tensor sin)?, Tensor?, (Tensor k_cache, Tensor v_cache)?,
            (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache)>
        create_llama_block()
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

public class LlamaAwq: Llama
{
    public static void convert_turbomind(torch.nn.Module module)
    {
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

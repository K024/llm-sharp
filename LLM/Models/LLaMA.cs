using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Layers;

namespace llm_sharp.LLM.Models;

using nn = torch.nn;
using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using F = torch.nn.functional;

/// input => output
using IModule = torch.nn.Module<torch.Tensor, torch.Tensor>;

/// input, rotary_embedding, attention_mask, (k_cache, v_cache) => output, (k_cache, v_cache)
using ILlamaAttention = torch.nn.Module<
    torch.Tensor, // x
    (torch.Tensor cos, torch.Tensor sin),
    torch.Tensor?, // attention mask
    (torch.Tensor k_cache, torch.Tensor v_cache)?,
    (torch.Tensor h, (torch.Tensor k_cache, torch.Tensor v_cache) kv_cache)
>;
using ILlamaBlock = torch.nn.Module<
    torch.Tensor, // x
    (torch.Tensor cos, torch.Tensor sin),
    torch.Tensor?, // attention mask
    (torch.Tensor k_cache, torch.Tensor v_cache)?,
    (torch.Tensor h, (torch.Tensor k_cache, torch.Tensor v_cache) kv_cache)
>;

/// position_ids => (cos, sin)
using IRotaryEmbedding = torch.nn.Module<torch.Tensor, (torch.Tensor cos, torch.Tensor sin)>;

public record LlamaConfig
{
    public virtual long hidden_size { get; set; } = 4096;
    public virtual long inner_hidden_size { get; set; } = 11008;
    public virtual long head_hidden_size { get; set; } = 128;
    public virtual string hidden_act { get; set; } = "silu";

    public virtual long num_attention_heads { get; set; } = 32;
    public virtual long num_key_value_heads { get; set; } = 32;
    public virtual int num_layers { get; set; } = 32;

    public virtual bool qkv_bias { get; set; } = false;
    public virtual bool o_bias { get; set; } = false;

    public virtual long vocab_size { get; set; } = 32000;
    public virtual double dropout_rate { get; set; } = 0.0;
    public virtual double layernorm_epsilon { get; set; } = 1e-06;
    public virtual long max_sequence_length { get; set; } = 2048;
    public virtual double rope_theta { get; set; } = 10000.0;
}

public class LlamaBuilder : AbstractBuilder
{
    public LlamaBuilder(LlamaConfig config)
    {
        this.config = config;
    }

    public virtual new LlamaConfig config { get => (LlamaConfig)base.config; set => base.config = value; }

    public virtual IModule create_ln()
        => new RMSNorm(new []{ config.hidden_size }, config.layernorm_epsilon, dtype: dtype, device: device);

    public virtual IModule create_dropout() => nn.Dropout(config.dropout_rate);

    public virtual ILlamaAttention create_llama_attention() => new LlamaAttention(this);

    public virtual IModule create_llama_ffn()
        => new GatedFeedForward(this, config.hidden_size, config.inner_hidden_size, config.dropout_rate, bias: false, act_fn_name: config.hidden_act);

    public virtual ILlamaBlock create_llama_block() => new LlamaBlock(this);

    public virtual IRotaryEmbedding create_rotary_embedding()
        => new RotaryEmbedding(config.max_sequence_length, config.head_hidden_size, config.rope_theta, dtype: dtype, device: device);
}

public class LlamaAttention : ILlamaAttention
{
    public long n_head;
    public long n_kv_head;
    public long d_head;
    public int n_groups;
    public IModule qkv_proj;
    public IModule o_proj;
    public Dropout dropout;

    public LlamaAttention(LlamaBuilder builder) : base("LlamaAttention")
    {
        var config = builder.config;
        n_head = config.num_attention_heads;
        n_kv_head = config.num_key_value_heads;
        d_head = config.head_hidden_size;

        if (n_head % n_kv_head != 0)
            throw new Exception("n_head should be multiple of n_kv_head");
        n_groups = (int)(n_head / n_kv_head);

        qkv_proj = builder.create_linear(config.hidden_size, d_head * (n_head + 2 * n_kv_head), hasBias: config.qkv_bias);
        o_proj = builder.create_linear(d_head * n_head, config.hidden_size, hasBias: config.o_bias);
        dropout = nn.Dropout(config.dropout_rate);

        RegisterComponents();
    }

    public static Tensor repeat_kv(Tensor x, int n)
    {
        using var scope = torch.NewDisposeScope();
        var (bs, n_heads, n_seq, d_head) = x.shape;
        x = x.unsqueeze(2).expand(bs, n_heads, n, n_seq, d_head);
        return scope.MoveToOuter(x.reshape(bs, n_heads * n, n_seq, d_head));
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

        q = RotaryEmbedding.apply_rotary_emb(q, freqs_cis);
        k = RotaryEmbedding.apply_rotary_emb(k, freqs_cis);

        if (kv_cache is not null)
        {
            var (k_cache, v_cache) = kv_cache.Value;
            k = torch.cat(new[] { k_cache, k }, dim: 1);
            v = torch.cat(new[] { v_cache, v }, dim: 1);
        }
        kv_cache = (k.detach(), v.detach());

        q = q.permute(0, 2, 1, 3);
        k = k.permute(0, 2, 3, 1);
        v = v.permute(0, 2, 1, 3);

        if (n_groups > 1)
        {
            k = repeat_kv(k, n_groups);
            v = repeat_kv(v, n_groups);
        }

        // (n_batch, n_head, n_seq, n_seq_past)
        var qk = torch.matmul(q, k) / Math.Sqrt(d_head);
        if (attention_mask is not null)
            qk = qk + attention_mask;

        var scores = F.softmax(qk, dim: -1, dtype: torch.float32).type_as(x);
        scores = dropout.call(scores);

        var output = torch.matmul(scores, v);

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

public class LlamaBlock : ILlamaBlock
{
    public IModule attn_ln;
    public ILlamaAttention attn;
    public IModule ffn_ln;
    public IModule ffn;

    public LlamaBlock(LlamaBuilder builder) : base("LlamaBlock")
    {
        attn_ln = builder.create_ln();
        attn = builder.create_llama_attention();
        ffn_ln = builder.create_ln();
        ffn = builder.create_llama_ffn();

        RegisterComponents();
    }

    public override (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache) forward(
        Tensor x,
        (Tensor cos, Tensor sin) freqs_cis,
        Tensor? attention_mask,
        (Tensor k_cache, Tensor v_cache)? kv_cache)
    {
        using var scope = torch.NewDisposeScope();
        var (h, new_kv_cache) = attn.call(
            attn_ln.call(x),
            freqs_cis,
            attention_mask,
            kv_cache
        );
        x = x + h;
        h = ffn.call(ffn_ln.call(x));
        x = x + h;
        return (
            scope.MoveToOuter(x),
            (
                scope.MoveToOuter(new_kv_cache.k_cache),
                scope.MoveToOuter(new_kv_cache.v_cache)
            )
        );
    }
}

public record LlamaModelInput : IBatchEncoding
{
    public Tensor? input_ids { get; set; }
    public Tensor? input_embeddings { get; set; }
    public Tensor? attention_mask { get; set; }
    public Tensor? position_ids { get; set; }
    public Tensor? labels { get; set; }
    public List<(Tensor k_cache, Tensor v_cache)>? past_key_values { get; set; }
}

public record LlamaModelOutput : IBatchEncoding
{
    public Tensor? loss { get; set; }
    public Tensor logits { get; set; } = null!;
    public List<(Tensor k_cache, Tensor v_cache)> current_key_values { get; set; } = null!;
}

public class LlamaModel : nn.Module<LlamaModelInput, LlamaModelOutput>
{
    public LlamaConfig config;
    public IModule word_embedding;
    public IModule dropout;
    public ModuleList<ILlamaBlock> layers;
    public IModule final_ln;
    public IModule lm_head;
    public IRotaryEmbedding rotary;

    public LlamaModel(LlamaBuilder builder) : base("LlamaModel")
    {
        config = builder.config;
        word_embedding = builder.create_embedding(config.vocab_size, config.hidden_size);
        dropout = builder.create_dropout();
        layers = nn.ModuleList(
            Enumerable.Range(0, config.num_layers)
                .Select(index => builder.create_llama_block()).ToArray());
        final_ln = builder.create_ln();
        lm_head = builder.create_linear(config.hidden_size, config.vocab_size, hasBias: false);
        rotary = builder.create_rotary_embedding();

        RegisterComponents();
    }

    public (
        Tensor input_embeddings,
        Tensor attention_mask,
        (Tensor cos, Tensor sin) freqs_cis
    ) prepare_input(LlamaModelInput input)
    {
        using var scope = torch.NewDisposeScope();

        torch.Device device;
        long n_batch, n_seq, n_seq_new, n_seq_past;

        var input_ids = input.input_ids;
        var input_embeddings = input.input_embeddings;
        var attention_mask = input.attention_mask;
        var position_ids = input.position_ids;
        var labels = input.labels;
        var past_key_values = input.past_key_values;

        if (input_embeddings is null)
        {
            if (input_ids is null)
                throw new ArgumentException(nameof(input_ids));
            device = input_ids.device;
            input_embeddings = word_embedding.call(input_ids);
            (n_batch, n_seq_new) = input_ids.shape;
        }
        else
        {
            if (input_ids is not null)
                throw new ArgumentException(nameof(input_embeddings));
            device = input_embeddings.device;
            (n_batch, n_seq_new, _) = input_embeddings.shape;
        }

        if (past_key_values is not null)
            n_seq_past = past_key_values[0].k_cache.shape[1];
        else
            n_seq_past = 0;

        n_seq = n_seq_new + n_seq_past;

        attention_mask ??= torch.ones(n_batch, n_seq, dtype: torch.int64, device: device);
        position_ids ??= torch.cumsum(attention_mask, dim: 1);

        // cast to float (n_batch, n_seq_new, n_seq) mask
        var causal_mask = torch.tril(
            torch.ones(n_seq_new, n_seq, dtype: torch.int64, device: device),
            n_seq_past
        );
        attention_mask = causal_mask[TensorIndex.None] & attention_mask[.., TensorIndex.None];
        attention_mask = (attention_mask - 1) * 1e10;

        // stripe to new seq length
        position_ids = position_ids[.., ^(int)n_seq_new..];
        // unsqueeze n_head dim
        attention_mask = attention_mask[.., TensorIndex.None];

        var freqs_cis = rotary.call(position_ids);

        return (
            scope.MoveToOuter(input_embeddings),
            scope.MoveToOuter(attention_mask),
            (
                scope.MoveToOuter(freqs_cis.cos),
                scope.MoveToOuter(freqs_cis.sin)
            )
        );
    }

    public override LlamaModelOutput forward(LlamaModelInput input)
    {
        using var outer_scope = torch.NewDisposeScope();

        var (
            prepared_input_embeddings,
            prepared_attention_mask,
            freqs_cis
        ) = prepare_input(input);

        // forward layers
        var h = dropout.call(prepared_input_embeddings);
        var current_key_values = new List<(Tensor k_cache, Tensor v_cache)>();
        foreach (var (i, layer) in layers.Select((v, i) => (i, v)))
        {
            using var scope = torch.NewDisposeScope();

            var kv_cache = input.past_key_values?[i];
            (h, kv_cache) = layer.call(
                h,
                freqs_cis,
                prepared_attention_mask,
                kv_cache
            );
            current_key_values.Add(kv_cache.Value);

            scope.MoveToOuter(h);
            scope.MoveToOuter(kv_cache.Value.k_cache);
            scope.MoveToOuter(kv_cache.Value.v_cache);
        }

        h = final_ln.call(h);
        var output = lm_head.call(h);

        Tensor? loss = null;
        if (input.labels is not null)
        {
            using var scope = torch.NewDisposeScope();

            var n_classes = config.vocab_size;
            var shift_logits = output[.., ..-1, ..].contiguous().to(torch.float32);
            var shift_labels = input.labels[.., 1..].contiguous();
            loss = F.cross_entropy(shift_logits.view(-1, n_classes), shift_labels.view(-1));

            scope.MoveToOuter(loss);
        }

        outer_scope.MoveToOuter(current_key_values.SelectMany(x => new[] { x.k_cache, x.v_cache }));
        return new()
        {
            logits = outer_scope.MoveToOuter(output),
            loss = loss is not null ? outer_scope.MoveToOuter(loss) : null,
            current_key_values = current_key_values
        };
    }
}

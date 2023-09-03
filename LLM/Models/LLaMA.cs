using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Models;

using nn = torch.nn;
using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using F = torch.nn.functional;

public record LLaMAConfig
{
    public virtual long hidden_size { get; set; } = 4096;
    public virtual long inner_hidden_size { get; set; } = 11008;
    public virtual long head_hidden_size { get; set; } = 128;

    public virtual long num_attention_heads { get; set; } = 32;
    public virtual int num_layers { get; set; } = 32;

    public virtual long vocab_size { get; set; } = 151936;
    public virtual double dropout_rate { get; set; } = 0.0;
    public virtual double layernorm_epsilon { get; set; } = 1e-05;
    public virtual long max_sequence_length { get; set; } = 2048;
}

public class CustomLinear : nn.Module<Tensor, Tensor>
{
    public Parameter weight;
    public Parameter? bias;
    public CustomLinear(
        long inputSize,
        long outputSize,
        bool hasBias = true,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("CustomLinear")
    {
        weight = new Parameter(torch.empty(outputSize, inputSize, dtype, device));
        if (hasBias)
            bias = new Parameter(torch.empty(outputSize, dtype, device));
        // skips init
        RegisterComponents();
    }
    public override Tensor forward(Tensor x)
    {
        return F.linear(x, weight, bias);
    }
}

public class CustomEmbedding : nn.Module<Tensor, Tensor>
{
    public Parameter weight;
    public CustomEmbedding(
        long num_embeddings,
        long embedding_dim,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("CustomEmbedding")
    {
        weight = new Parameter(torch.empty(num_embeddings, embedding_dim, dtype, device));
        // skips init
        RegisterComponents();
    }
    public override Tensor forward(Tensor x)
    {
        return weight[x];
    }
}

public class RotaryEmbedding : nn.Module<Tensor, (Tensor cos, Tensor sin)>
{
    public static (Tensor cos, Tensor sin) precompute_freqs_cis(long dim, long length, double theta = 10000.0)
    {
        using var scope = torch.NewDisposeScope();
        if (dim % 2 != 0)
            throw new Exception("dim should be multiple of 2");
        var freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32) / dim);
        freqs = torch.outer(torch.arange(length).to(torch.float32), freqs);
        freqs = torch.cat(new[] { freqs, freqs }, dim: -1);
        return scope.MoveToOuter(torch.cos(freqs), torch.sin(freqs));
    }

    public static Tensor apply_rotary_emb(Tensor x, (Tensor cos, Tensor sin) freqs_cis)
    {
        using var scope = torch.NewDisposeScope();
        var half_dim = x.shape.Last() / 2;
        var splitSizes = Enumerable.Repeat(half_dim, 2).ToArray();
        var (x_r, x_i) = torch.split(x, splitSizes, dim: -1);
        var rotated = torch.cat(new[] { -x_i, x_r }, dim: -1);
        return scope.MoveToOuter(x * freqs_cis.cos + rotated * freqs_cis.sin);
    }

    public Tensor cos;
    public Tensor sin;

    public RotaryEmbedding(
        long num_embeddings,
        long embedding_dims,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("RotaryEmbedding")
    {
        (cos, sin) = precompute_freqs_cis(embedding_dims, num_embeddings);
        if (dtype is not null)
            (cos, sin) = (cos.to(dtype.Value), sin.to(dtype.Value));
        if (device is not null)
            (cos, sin) = (cos.to(device), sin.to(device));

        RegisterComponents();
    }

    public override (Tensor cos, Tensor sin) forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var (n_batch, n_seq) = x.shape;
        return scope.MoveToOuter(
            cos[x].view(n_batch, n_seq, 1, -1),
            sin[x].view(n_batch, n_seq, 1, -1)
        );
    }

    public override Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor>? destination = null, string? prefix = null)
    {
        // omit from state_dict
        return destination ?? new();
    }
}

public class RMSNorm : nn.Module<Tensor, Tensor>
{
    public Parameter weight;
    public double eps;

    public RMSNorm(
        long[] normalized_shape,
        double eps = 1e-5,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("RMSNorm")
    {
        weight = nn.Parameter(torch.ones(normalized_shape, dtype: dtype, device: device));
        this.eps = eps;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var h = x.to(torch.float32);
        var norm = h * torch.rsqrt(h.pow(2).mean(new[] { -1L }, keepdim: true) + (float)eps);
        return scope.MoveToOuter(norm.type_as(x) * weight);
    }
}

public class LLaMAAttention : nn.Module<
    Tensor, // x
    (Tensor cos, Tensor sin),
    Tensor?, // attention mask
    (Tensor k_cache, Tensor v_cache)?,
    (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache)>
{
    public long n_head;
    public long d_head;
    public CustomLinear qkv_proj;
    public CustomLinear o_proj;
    public Dropout dropout;

    public LLaMAAttention(
        long n_state,
        long n_head,
        long d_head,
        double dropout_rate = 0.0,
        bool qkv_bias = true,
        bool o_bias = false,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("LLaMAAttention")
    {
        this.n_head = n_head;
        this.d_head = d_head;
        if (this.d_head % (4 * n_head) != 0)
            throw new Exception("d_head should be multiple of n_head");
        qkv_proj = new CustomLinear(n_state, d_head * n_head * 3, hasBias: qkv_bias, dtype: dtype, device: device);
        o_proj = new CustomLinear(d_head * n_head, n_state, hasBias: o_bias, dtype: dtype, device: device);
        dropout = nn.Dropout(dropout_rate);

        RegisterComponents();
    }

    public override (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache) forward(
        Tensor x,
        (Tensor cos, Tensor sin) freqs_cis,
        Tensor? attention_mask,
        (Tensor k_cache, Tensor v_cache)? kv_cache)
    {
        using var scope = torch.NewDisposeScope();

        var (n_batch, n_seq, _) = x.shape;

        var splitSizes = Enumerable.Repeat(d_head * n_head, 3).ToArray();
        var (q, k, v) = torch.split(qkv_proj.call(x), splitSizes, dim: -1);

        q = q.view(n_batch, n_seq, n_head, d_head);
        k = k.view(n_batch, n_seq, n_head, d_head);
        v = v.view(n_batch, n_seq, n_head, d_head);

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

public class GatedFeedForward : nn.Module<Tensor, Tensor>
{
    public long hidden_dim;
    public CustomLinear w_in;
    public CustomLinear w_gate;
    public CustomLinear w_out;
    public Dropout dropout;
    public Func<Tensor, Tensor> act_fn;

    public GatedFeedForward(
        long dim,
        long? hidden_dim = null,
        double dropout_rate = 0.0,
        bool bias = false,
        torch.ScalarType? dtype = null,
        torch.Device? device = null,
        Func<Tensor, Tensor>? act_fn = null
    ) : base("GatedFeedForward")
    {
        this.hidden_dim = hidden_dim ?? dim * 4;
        w_in = new CustomLinear(dim, this.hidden_dim, hasBias: bias, dtype: dtype, device: device);
        w_gate = new CustomLinear(dim, this.hidden_dim, hasBias: bias, dtype: dtype, device: device);
        w_out = new CustomLinear(this.hidden_dim, dim, hasBias: bias, dtype: dtype, device: device);
        dropout = nn.Dropout(dropout_rate);
        this.act_fn = act_fn ?? F.SiLU;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var h = act_fn(w_gate.call(x)) * w_in.call(x);
        return scope.MoveToOuter(w_out.call(dropout.call(h)));
    }
}

public class LLaMABlock : nn.Module<
    Tensor, // x
    (Tensor cos, Tensor sin),
    Tensor?, // attention mask
    (Tensor k_cache, Tensor v_cache)?,
    (Tensor h, (Tensor k_cache, Tensor v_cache) kv_cache)>
{
    public RMSNorm attn_ln;
    public LLaMAAttention attn;
    public RMSNorm ffn_ln;
    public GatedFeedForward ffn;

    public LLaMABlock(LLaMAConfig config, torch.ScalarType? dtype, torch.Device? device = null) : base("LLaMABlock")
    {
        attn_ln = new RMSNorm(
            new[] { config.hidden_size },
            eps: config.layernorm_epsilon,
            dtype: dtype, device: device);
        attn = new LLaMAAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.head_hidden_size,
            dropout_rate: config.dropout_rate,
            dtype: dtype, device: device);
        ffn_ln = new RMSNorm(
            new[] { config.hidden_size },
            eps: config.layernorm_epsilon,
            dtype: dtype, device: device);
        ffn = new GatedFeedForward(
            config.hidden_size,
            config.inner_hidden_size,
            config.dropout_rate,
            dtype: dtype, device: device);

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

public record LLaMAModelInput : BatchEncoding
{
    public Tensor? input_ids { get; set; }
    public Tensor? input_embeddings { get; set; }
    public Tensor? attention_mask { get; set; }
    public Tensor? position_ids { get; set; }
    public Tensor? labels { get; set; }
    public List<(Tensor k_cache, Tensor v_cache)>? past_key_values { get; set; }
}

public record LLaMAModelOutput : BatchEncoding
{
    public Tensor? loss { get; set; }
    public Tensor logits { get; set; } = null!;
    public List<(Tensor k_cache, Tensor v_cache)> current_key_values { get; set; } = null!;
}

public class LLaMAModel : nn.Module<LLaMAModelInput, LLaMAModelOutput>
{
    public LLaMAConfig config;
    public CustomEmbedding word_embedding;
    public Dropout dropout;
    public ModuleList<LLaMABlock> layers;
    public RMSNorm final_ln;
    public CustomLinear lm_head;
    public RotaryEmbedding rotary;

    public (Tensor cos, Tensor sin) freqs_cis;

    public LLaMAModel(LLaMAConfig config, torch.ScalarType? dtype = null, torch.Device? device = null) : base("LLaMAModel")
    {
        this.config = config;
        word_embedding = new CustomEmbedding(
            config.vocab_size, config.hidden_size, dtype: dtype, device: device
        );
        dropout = nn.Dropout(config.dropout_rate);
        layers = nn.ModuleList(
            Enumerable.Range(0, config.num_layers)
                .Select(index => new LLaMABlock(config, dtype: dtype, device: device)).ToArray());
        final_ln = new RMSNorm(
            new[] { config.hidden_size }, eps: config.layernorm_epsilon, dtype: dtype, device: device);
        lm_head = new CustomLinear(
            config.hidden_size, config.vocab_size, hasBias: false, dtype: dtype, device: device);
        rotary = new RotaryEmbedding(
            config.max_sequence_length, config.head_hidden_size, dtype: dtype, device: device);

        RegisterComponents();
    }

    public (
        Tensor input_embeddings,
        Tensor attention_mask,
        (Tensor cos, Tensor sin) freqs_cis
    ) prepare_input(LLaMAModelInput input)
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

    public override LLaMAModelOutput forward(LLaMAModelInput input)
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

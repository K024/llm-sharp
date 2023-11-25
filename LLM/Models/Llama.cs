using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Layers;
using llm_sharp.LLM.Pretrained;
using llm_sharp.LLM.Tokenizers;

namespace llm_sharp.LLM.Models;

using nn = torch.nn;
using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using F = torch.nn.functional;

/// input => output
using IModule = torch.nn.Module<torch.Tensor, torch.Tensor>;

/// input, attention_mask, rotary_embedding, (k_cache, v_cache) => output
using ILlamaAttention = torch.nn.Module<
    torch.Tensor, // x
    torch.Tensor?, // attention mask
    IRotary?,
    IKvCache?,
    torch.Tensor
>;
using ILlamaBlock = torch.nn.Module<
    torch.Tensor, // x
    torch.Tensor?, // attention mask
    IRotary?,
    IKvCache?,
    torch.Tensor
>;

/// position_ids => (cos, sin)
using IRotaryEmbedding = torch.nn.Module<torch.Tensor, IRotary>;

/// q_seq_len, kv_seq_len => mask
using IAlibi = torch.nn.Module<long, long, torch.Tensor>;

public record LlamaConfig
{
    public long hidden_size { get; set; } = 4096;
    public long inner_hidden_size { get; set; } = 11008;
    public long head_hidden_size { get; set; } = 128;
    public string hidden_act { get; set; } = "silu";

    public long num_attention_heads { get; set; } = 32;
    public long num_key_value_heads { get; set; } = 32;
    public int num_layers { get; set; } = 32;

    public bool qkv_bias { get; set; } = false;
    public bool o_bias { get; set; } = false;

    public long vocab_size { get; set; } = 32000;
    public double dropout_rate { get; set; } = 0.0;
    public double layernorm_epsilon { get; set; } = 1e-06;
    public long max_sequence_length { get; set; } = 2048;
    public double rope_theta { get; set; } = 10000.0;
    public bool use_alibi { get; set; } = false;
}

public class LlamaBuilder : AbstractBuilder
{
    public LlamaBuilder(LlamaConfig config)
    {
        this.config = config;
    }

    public virtual new LlamaConfig config { get => (LlamaConfig)base.config; set => base.config = value; }

    public virtual IModule create_ln()
        => new RMSNorm(new[] { config.hidden_size }, config.layernorm_epsilon, dtype: dtype, device: device);

    public virtual IModule create_dropout() => nn.Dropout(config.dropout_rate);

    public virtual ILlamaAttention create_llama_attention() => new LlamaAttention(this);

    public virtual IModule create_llama_ffn()
        => new GatedFeedForward(this, config.hidden_size, config.inner_hidden_size, config.dropout_rate, bias: false, act_fn_name: config.hidden_act);

    public virtual ILlamaBlock create_llama_block() => new LlamaBlock(this);

    public virtual IRotaryEmbedding create_rotary_embedding()
        => new RotaryEmbedding(config.max_sequence_length, config.head_hidden_size, config.rope_theta, dtype: dtype, device: device);

    public virtual IAlibi create_alibi()
        => new Alibi(config.num_attention_heads, config.max_sequence_length, dtype: dtype, device: device);

    public virtual List<IKvCache> create_kv_cache(long batch_size)
        => Enumerable.Range(0, config.num_layers).Select(_ => (IKvCache)new KvCache()).ToList();
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

        return scope.MoveToOuter(output);
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

    public override Tensor forward(
        Tensor x,
        Tensor? attention_mask,
        IRotary? freqs_cis,
        IKvCache? kv_cache)
    {
        using var scope = torch.NewDisposeScope();
        var h = attn.call(
            attn_ln.call(x),
            attention_mask,
            freqs_cis,
            kv_cache
        );
        x = x + h;
        h = ffn.call(ffn_ln.call(x));
        x = x + h;
        return scope.MoveToOuter(x);
    }
}

public record LlamaModelInput : IBatchEncoding
{
    public Tensor? input_ids { get; set; }
    public Tensor? input_embeddings { get; set; }
    public Tensor? attention_mask { get; set; }
    public Tensor? position_ids { get; set; }
    public Tensor? labels { get; set; }
    public List<IKvCache>? past_key_values { get; set; }
}

public record LlamaModelOutput : IBatchEncoding
{
    public Tensor? loss { get; set; }
    public Tensor logits { get; set; } = null!;
}

public class LlamaModel : nn.Module<LlamaModelInput, LlamaModelOutput>
{
    public LlamaConfig config;
    public IModule word_embedding;
    public IModule dropout;
    public ModuleList<ILlamaBlock> layers;
    public IModule final_ln;
    public IModule lm_head;
    public IRotaryEmbedding? rotary;
    public IAlibi? alibi;

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

        if (!config.use_alibi)
            rotary = builder.create_rotary_embedding();
        else
            alibi = builder.create_alibi();

        kv_cache_factory = builder.create_kv_cache;
        RegisterComponents();
    }

    protected Func<long, List<IKvCache>> kv_cache_factory;

    public List<IKvCache> create_kv_cache(long batch_size) => kv_cache_factory(batch_size);

    public (
        Tensor input_embeddings,
        Tensor attention_mask,
        IRotary? freqs_cis
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
            n_seq_past = past_key_values[0].size;
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
        attention_mask = attention_mask[.., TensorIndex.None].type_as(input_embeddings);

        IRotary? freqs_cis = null;
        if (rotary is not null)
        {
            freqs_cis = rotary.call(position_ids);
            scope.MoveToOuter(freqs_cis.weights);
        }
        else if (alibi is not null)
        {
            // alibi attention bias
            var alibi_bias = alibi.call(n_seq_new, n_seq);
            attention_mask = attention_mask + alibi_bias;
        }

        return (
            scope.MoveToOuter(input_embeddings),
            scope.MoveToOuter(attention_mask),
            freqs_cis
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
        foreach (var (i, layer) in layers.Select((v, i) => (i, v)))
        {
            using var scope = torch.NewDisposeScope();
            var kv_cache = input.past_key_values?[i];
            h = layer.call(
                h,
                prepared_attention_mask,
                freqs_cis,
                kv_cache
            );
            scope.MoveToOuter(h);
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

        return new()
        {
            logits = outer_scope.MoveToOuter(output),
            loss = loss is not null ? outer_scope.MoveToOuter(loss) : null,
        };
    }
}

public abstract class AbstractLlama : GenerativeLM<AbstractLlama.LlamaState>
{
    public class LlamaState : IDisposable
    {
        public List<IKvCache> past_key_values { get; set; } = new();
        public torch.Generator? generator { get; set; }

        public void Dispose()
        {
            past_key_values.ForEach(x => x.Dispose());
            generator?.Dispose();
        }
    }

    public torch.Device? device { get; protected set; }
    public LlamaModel model { get; init; }
    public LlamaConfig model_config { get; init; }

    public override int max_sequence_length => (int)model_config.max_sequence_length;

#nullable disable
    protected AbstractLlama() { }
#nullable restore

    public virtual void to(torch.Device? device)
    {
        model.to(device);
        this.device = device;
    }

    protected override LlamaState prepare_init_state(List<int> input_tokens, GenerationConfig config)
    {
        return new() {
            past_key_values = model.create_kv_cache(1),
            generator = config.seed.HasValue ? new torch.Generator((ulong)config.seed.Value, device) : null
        };
    }

    protected override int generate_step(List<int> tokens, List<int> generated_tokens, LlamaState state, GenerationConfig config)
    {
        using var scope = torch.NewDisposeScope();
        using var no_grad = torch.no_grad();

        if (generated_tokens.Count > 0)
            tokens = generated_tokens.TakeLast(1).ToList();

        var input_ids = torch.tensor(tokens, dtype: torch.int64, device: device).unsqueeze(0);

        model.eval();
        var output = model.call(new()
        {
            input_ids = input_ids,
            past_key_values = state.past_key_values,
        });
        var logits = output.logits[0, ^1];

        logits_bias(logits, generated_tokens, config.frequency_penalty, config.presence_penalty);
        var next = top_p_sampling(logits, config.top_p, config.temperature, state.generator);
        var next_token = (int)next.item<long>();

        return next_token;
    }
}

public class Llama : AbstractLlama
{
    public SentencePieceBPE tokenizer { get; init; }
    public SentencePieceBPEConfig tokenizer_config { get; init; }
    protected override List<int> eos_tokens => tokenizer.eos_ids;

#nullable disable
    protected Llama() { }
#nullable restore

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

    protected override List<int> prepare_input(List<ChatMessage> messages)
    {
        var prompt = "";
        var system = "";
        foreach (var message in messages)
        {
            prompt += message.role switch
            {
                "user" when !string.IsNullOrEmpty(system) =>
                    $"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{message.content} [/INST]",
                "user" => $"<s>[INST] {message.content} [/INST]",
                "assistant" => $" {message.content} </s>",
                _ => "",
            };
            system = message.role switch
            {
                "system" => message.content,
                _ => ""
            };
        }
        return tokenizer.encode_text(prompt);
    }

    protected override string decode_output(List<int> tokens)
    {
        return tokenizer.decode_text(tokens);
    }
}

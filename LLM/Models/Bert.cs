using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Tokenizers;
using System.Text.Json;

namespace llm_sharp.LLM.Models;

using nn = torch.nn;
using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using F = torch.nn.functional;

public record BertConfig
{
    public virtual long hidden_size { get; set; } = 768;
    public virtual long inner_hidden_size { get; set; } = 3072;
    public virtual long head_hidden_size { get; set; } = 64;

    public virtual long num_attention_heads { get; set; } = 12;
    public virtual int num_layers { get; set; } = 12;

    public virtual long vocab_size { get; set; } = 40000;
    public virtual double dropout_rate { get; set; } = 0.0;
    public virtual double layernorm_epsilon { get; set; } = 1e-05;

    public virtual long type_vocab_size { get; set; } = 4;
    public virtual long max_sequence_length { get; set; } = 2048;
    public virtual List<string> pooling_mode { get; set; } = new() { BertPooler.POOLER };
    public virtual int classifier_classes { get; set; } = 0;
    public virtual string classifier_mode { get; set; } = BertModel.SEQUENCE;
}

public class BertEmbeddings : nn.Module<Tensor?, Tensor?, Tensor?, Tensor?, Tensor>
{
    public CustomEmbedding word_embeddings;
    public CustomEmbedding position_embeddings;
    public CustomEmbedding token_type_embeddings;
    public LayerNorm layer_norm;
    public Dropout dropout;

    public BertEmbeddings(
        long vocab_size,
        long hidden_size,
        long max_position_embeddings,
        long type_vocab_size,
        double layer_norm_eps = 1e-5,
        double dropout_rate = 0.0,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("BertEmbeddings")
    {
        word_embeddings = new CustomEmbedding(vocab_size, hidden_size, dtype: dtype, device: device);
        position_embeddings = new CustomEmbedding(max_position_embeddings, hidden_size, dtype: dtype, device: device);
        token_type_embeddings = new CustomEmbedding(type_vocab_size, hidden_size, dtype: dtype, device: device);

        layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps, dtype: dtype, device: device);
        dropout = nn.Dropout(dropout_rate);

        RegisterComponents();
    }
    public override Tensor forward(
        Tensor? input_ids,
        Tensor? input_embeddings,
        Tensor? token_type_ids,
        Tensor? position_ids)
    {
        using var scope = torch.NewDisposeScope();

        long batch_size;
        long sequence_length;
        torch.Device device;
        if (input_ids is not null)
        {
            (batch_size, sequence_length) = input_ids.shape;
            device = input_ids.device;
        }
        else if (input_embeddings is not null)
        {
            (batch_size, sequence_length, _) = input_embeddings.shape;
            device = input_embeddings.device;
        }
        else
        {
            throw new Exception("input_ids or input_embeddings should be defined");
        }

        position_ids ??= torch.arange(sequence_length, torch.int64, device).unsqueeze(0);
        token_type_ids ??= torch.zeros(sequence_length, torch.int64, device).unsqueeze(0);

        input_embeddings ??= word_embeddings.call(input_ids!);
        var type_embeddings = token_type_embeddings.call(token_type_ids);
        var pos_embeddings = position_embeddings.call(position_ids);

        var embeddings = input_embeddings + type_embeddings + pos_embeddings;
        embeddings = layer_norm.call(embeddings);
        embeddings = dropout.call(embeddings);

        return scope.MoveToOuter(embeddings);
    }
}

public class BertAttention : nn.Module<Tensor, Tensor?, Tensor>
{
    public long n_head;
    public long d_head;
    public CustomLinear qkv_proj;
    public CustomLinear o_proj;
    public Dropout dropout;

    public BertAttention(
        long n_state,
        long n_head,
        long d_head,
        double dropout_rate = 0.0,
        bool qkv_bias = true,
        bool o_bias = true,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("BertAttention")
    {
        this.n_head = n_head;
        this.d_head = d_head;

        qkv_proj = new CustomLinear(n_state, d_head * n_head * 3, hasBias: qkv_bias, dtype: dtype, device: device);
        o_proj = new CustomLinear(n_head * d_head, n_state, o_bias, dtype: dtype, device: device);
        dropout = nn.Dropout(dropout_rate);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor? attention_mask)
    {
        using var scope = torch.NewDisposeScope();

        var (n_batch, n_seq, _) = x.shape;

        var splitSizes = Enumerable.Repeat(d_head * n_head, 3).ToArray();
        var (q, k, v) = torch.split(qkv_proj.call(x), splitSizes, dim: -1);

        q = q.view(n_batch, n_seq, n_head, d_head).permute(0, 2, 1, 3);
        k = k.view(n_batch, n_seq, n_head, d_head).permute(0, 2, 3, 1);
        v = v.view(n_batch, n_seq, n_head, d_head).permute(0, 2, 1, 3);

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

public class FeedForward : nn.Module<Tensor, Tensor>
{
    public long hidden_dim;
    public CustomLinear w_in;
    public CustomLinear w_out;
    public Dropout dropout;
    public Func<Tensor, Tensor> act_fn;

    public FeedForward(
        long dim,
        long? hidden_dim = null,
        double dropout_rate = 0.0,
        bool bias = true,
        torch.ScalarType? dtype = null,
        torch.Device? device = null,
        Func<Tensor, Tensor>? act_fn = null
    ) : base("FeedForward")
    {
        this.hidden_dim = hidden_dim ?? dim * 4;
        w_in = new CustomLinear(dim, this.hidden_dim, hasBias: bias, dtype: dtype, device: device);
        w_out = new CustomLinear(this.hidden_dim, dim, hasBias: bias, dtype: dtype, device: device);
        dropout = nn.Dropout(dropout_rate);
        this.act_fn = act_fn ?? F.gelu;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var h = act_fn(w_in.call(x));
        return scope.MoveToOuter(w_out.call(dropout.call(h)));
    }
}

public class BertBlock : nn.Module<Tensor, Tensor?, Tensor>
{
    public BertAttention attn;
    public LayerNorm attn_ln;
    public FeedForward ffn;
    public LayerNorm ffn_ln;

    public BertBlock(BertConfig config, torch.ScalarType? dtype = null, torch.Device? device = null) : base("BertBlock")
    {
        attn = new BertAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.head_hidden_size,
            dropout_rate: config.dropout_rate,
            dtype: dtype, device: device);
        attn_ln = nn.LayerNorm(
            new[] { config.hidden_size },
            eps: config.layernorm_epsilon,
            dtype: dtype, device: device);
        ffn = new FeedForward(
            config.hidden_size,
            config.inner_hidden_size,
            config.dropout_rate,
            dtype: dtype, device: device);
        ffn_ln = nn.LayerNorm(
            new[] { config.hidden_size },
            eps: config.layernorm_epsilon,
            dtype: dtype, device: device);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor? attention_mask)
    {
        using var scope = torch.NewDisposeScope();
        x = attn_ln.call(x + attn.call(x, attention_mask));
        x = ffn_ln.call(x + ffn.call(x));
        return scope.MoveToOuter(x);
    }
}

public class BertPooler : nn.Module<Tensor, Tensor?, Tensor>
{
    public nn.Module<Tensor, Tensor> dense;
    private readonly List<string> pooling_mode;
    public Func<Tensor, Tensor> activation;

    public const string POOLER = "pooler";
    public const string CLS = "cls";
    public const string MEAN = "mean";
    public const string MAX = "max";

    public static List<string> POOLING_MODES = new() { POOLER, CLS, MEAN, MAX };

    public BertPooler(
        long n_state,
        List<string> pooling_mode,
        Func<Tensor, Tensor>? activation = null,
        torch.ScalarType? dtype = null,
        torch.Device? device = null) : base("BertPooler")
    {
        // validate pooling mode
        if (pooling_mode.Count == 0)
            throw new Exception("pooling_mode should not be empty");

        if (pooling_mode.Count > 1 && pooling_mode.Contains(POOLER))
            throw new Exception("pooling_mode should not contain pooler and other modes");

        if (pooling_mode.Any(x => !POOLING_MODES.Contains(x)))
            throw new Exception($"pooling_mode should only contain {string.Join(", ", POOLING_MODES)}");

        dense = pooling_mode.Contains(POOLER)
            ? new CustomLinear(n_state, n_state, dtype: dtype, device: device)
            : nn.Identity();

        this.pooling_mode = pooling_mode;
        this.activation = activation ?? F.tanh;
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor? attention_mask = null)
    {
        using var scope = torch.NewDisposeScope();

        var results = new List<Tensor>();

        if (pooling_mode.Contains(POOLER))
        {
            var first_word = x[.., 0];
            var pooled = dense.call(first_word);
            return scope.MoveToOuter(activation(pooled));
        }

        if (pooling_mode.Contains(CLS))
        {
            var first_word = x[.., 0];
            results.Add(first_word);
        }

        if (pooling_mode.Contains(MEAN))
        {
            Tensor mean;
            if (attention_mask is not null)
            {
                var count = attention_mask.sum(1, keepdim: true);
                var masked = x * attention_mask[.., .., TensorIndex.None];
                mean = torch.sum(masked, 1) / count;
            }
            else
            {
                mean = torch.mean(x, new[] { 1L });
            }
            results.Add(mean);
        }

        if (pooling_mode.Contains(MAX))
        {
            Tensor max;
            if (attention_mask is not null)
            {
                var masked = x * attention_mask[.., .., TensorIndex.None]
                    + (attention_mask[.., .., TensorIndex.None] - 1) * 1e10;
                max = torch.max(masked, new[] { 1L });
            }
            else
            {
                max = torch.max(x, new[] { 1L });
            }
            results.Add(max);
        }
        return scope.MoveToOuter(torch.cat(results, dim: 1));
    }
}

public record BertModelInput : BatchEncoding
{
    public Tensor? input_ids { get; set; }
    public Tensor? input_embeddings { get; set; }
    public Tensor? attention_mask { get; set; }
    public Tensor? token_type_ids { get; set; }
    public Tensor? position_ids { get; set; }
}

public record BertModelOutput : BatchEncoding
{
    public Tensor last_hidden_state { get; set; } = null!;
    public Tensor? pooler_output { get; set; }
    public Tensor? classifier_output { get; set; }
}

public class BertModel : nn.Module<BertModelInput, BertModelOutput>
{
    public BertConfig config;
    public BertEmbeddings embedding;
    public ModuleList<BertBlock> layers;
    public BertPooler? pooler;
    public CustomLinear? classifier;

    public const string SEQUENCE = "sequence";
    public const string TOKEN = "token";

    public BertModel(BertConfig config, torch.ScalarType? dtype = null, torch.Device? device = null) : base("BertModel")
    {
        this.config = config;

        embedding = new BertEmbeddings(
            config.vocab_size,
            config.hidden_size,
            config.max_sequence_length,
            config.type_vocab_size,
            config.layernorm_epsilon,
            config.dropout_rate,
            dtype: dtype, device: device
        );
        layers = nn.ModuleList(
            Enumerable.Range(0, config.num_layers)
                .Select(index => new BertBlock(config, dtype: dtype, device: device)).ToArray());

        if (config.pooling_mode.Count > 0)
            pooler = new BertPooler(config.hidden_size, config.pooling_mode, dtype: dtype, device: device);

        if (config.classifier_classes > 0)
            classifier = new CustomLinear(config.hidden_size, config.classifier_classes, dtype: dtype, device: device);

        RegisterComponents();
    }

    public (Tensor input_embeddings, Tensor? attention_mask) prepare_input(BertModelInput input)
    {
        using var scope = torch.NewDisposeScope();

        var input_ids = input.input_ids;
        var input_embeddings = input.input_embeddings;
        var attention_mask = input.attention_mask;
        var token_type_ids = input.token_type_ids;
        var position_ids = input.position_ids;

        input_embeddings = embedding.call(input_ids, input_embeddings, token_type_ids, position_ids);

        var (n_batch, n_seq, _) = input_embeddings.shape;
        var device = input_embeddings.device;

        if (attention_mask is not null)
        {
            attention_mask = (attention_mask - 1) * 1e10;
            // unsqueeze n_head & n_query dim
            attention_mask = attention_mask[.., TensorIndex.None, TensorIndex.None];
            scope.MoveToOuter(attention_mask);
        }

        return (
            scope.MoveToOuter(input_embeddings),
            attention_mask
        );
    }

    public override BertModelOutput forward(BertModelInput input)
    {
        using var outer_scope = torch.NewDisposeScope();

        var (
            prepared_input_embeddings,
            prepared_attention_mask
        ) = prepare_input(input);

        // forward layers
        var h = prepared_input_embeddings;
        foreach (var (i, layer) in layers.Select((v, i) => (i, v)))
        {
            using var scope = torch.NewDisposeScope();

            h = layer.call(h, prepared_attention_mask);
            scope.MoveToOuter(h);
        }

        Tensor? pooler_output = null, classifier_output = null;

        if (pooler is not null)
        {
            pooler_output = pooler.call(h, input.attention_mask);
            outer_scope.MoveToOuter(pooler_output);
        }

        if (classifier is not null)
        {
            if (config.classifier_mode == TOKEN)
                classifier_output = classifier.call(h);
            else if (config.classifier_mode == SEQUENCE)
                classifier_output = classifier.call(pooler_output
                    ?? throw new Exception("no pooler output for sequence classification"));
            else
                throw new Exception($"Unknown classifier_mode {config.classifier_mode}");

            if (classifier_output is not null)
                outer_scope.MoveToOuter(classifier_output);
        }

        return new()
        {
            last_hidden_state = outer_scope.MoveToOuter(h),
            pooler_output = pooler_output,
            classifier_output = classifier_output,
        };
    }
}

public class BertEncoder : MaskedLM<BertModel, BertConfig, WordPiece, WordPieceConfig>
{
    public override torch.Device? device { get; protected set; }
    public override BertModel model { get; init; }
    public override BertConfig model_config { get; init; }
    public override WordPiece tokenizer { get; init; }
    public override WordPieceConfig tokenizer_config { get; init; }

#nullable disable
    protected BertEncoder() { }
#nullable restore

    public virtual BertEncoder to(torch.Device? device)
    {
        model.to(device);
        this.device = device;
        return this;
    }

    public static BertEncoder from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var (model, model_config) = model_from_pretrained(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained(path);
        return new BertEncoder()
        {
            device = device,
            model = model,
            model_config = model_config,
            tokenizer = tokenizer,
            tokenizer_config = tokenizer_config,
        };
    }

    protected override List<int> prepare_input(string input)
    {
        var list = tokenizer.encode_text(input);
        list.Insert(0, tokenizer["[CLS]"]);
        list.Add(tokenizer["[SEP]"]);
        return list;
    }

    protected override IList<float> encode_tokens(List<int> tokens)
    {
        using var scope = torch.NewDisposeScope();

        var input_ids = torch.tensor(tokens, dtype: torch.int64, device: device).unsqueeze(0);
        var type_ids = torch.zeros_like(input_ids);

        var result = model.call(new BertModelInput()
        {
            input_ids = input_ids,
            token_type_ids = type_ids,
        });

        if (result.pooler_output is null)
            throw new Exception("No pooler for BertDualEncoder");

        return result.pooler_output.cpu().to(torch.float32).data<float>().ToList();
    }
}

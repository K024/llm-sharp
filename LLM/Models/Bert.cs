using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Tokenizers;
using llm_sharp.LLM.Pretrained;
using System.Text.Json;
using llm_sharp.LLM.Layers;

namespace llm_sharp.LLM.Models;

using nn = torch.nn;
using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using F = torch.nn.functional;

/// input => output
using IModule = torch.nn.Module<torch.Tensor, torch.Tensor>;

/// input_ids, input_embeddings, token_type_ids, position_ids => output
using IBertEmbeddings = torch.nn.Module<torch.Tensor?, torch.Tensor?, torch.Tensor?, torch.Tensor?, torch.Tensor>;

/// input, attention_mask => output
using IBertAttention = torch.nn.Module<torch.Tensor, torch.Tensor?, torch.Tensor>;
using IBertPooler = torch.nn.Module<torch.Tensor, torch.Tensor?, torch.Tensor>;
using IBertBlock = torch.nn.Module<torch.Tensor, torch.Tensor?, torch.Tensor>;

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

    // "pooler": tanh(dense([CLS]))
    // "cls", "mean", "max": no extra layers
    public virtual List<string> pooling_mode { get; set; } = new() { BertPooler.POOLER };
    public virtual int classifier_classes { get; set; } = 0;
    public virtual string classifier_mode { get; set; } = BertModel.SEQUENCE;
}

public class BertBuilder : AbstractBuilder
{
    public BertBuilder(BertConfig config)
    {
        this.config = config;
    }

    public virtual new BertConfig config { get => (BertConfig)base.config; set => base.config = value; }

    public virtual IModule create_ln(params long[] shape)
        => nn.LayerNorm(shape, config.layernorm_epsilon, dtype: dtype, device: device);

    public virtual IModule create_dropout() => nn.Dropout(config.dropout_rate);

    public virtual IBertEmbeddings create_bert_embeddings() => new BertEmbeddings(this);

    public virtual IBertAttention create_bert_attention() => new BertAttention(this);

    public virtual IModule create_bert_ffn()
        => new FeedForward(this, config.hidden_size, config.inner_hidden_size, config.dropout_rate, bias: true, act_fn_name: "gelu");

    public virtual IBertPooler create_bert_pooler() => new BertPooler(this);

    public virtual IBertBlock create_bert_block() => new BertBlock(this);
}

public class BertEmbeddings : IBertEmbeddings
{
    public IModule word_embeddings;
    public IModule position_embeddings;
    public IModule token_type_embeddings;
    public IModule layer_norm;
    public IModule dropout;

    public BertEmbeddings(BertBuilder builder) : base("BertEmbeddings")
    {
        var config = builder.config;
        word_embeddings = builder.create_embedding(config.vocab_size, config.hidden_size);
        position_embeddings = builder.create_embedding(config.max_sequence_length, config.hidden_size);
        token_type_embeddings = builder.create_embedding(config.type_vocab_size, config.hidden_size);

        layer_norm = builder.create_ln(config.hidden_size);
        dropout = builder.create_dropout();

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

public class BertAttention : IBertAttention
{
    public long n_head;
    public long d_head;
    public IModule qkv_proj;
    public IModule o_proj;
    public IModule dropout;

    public BertAttention(BertBuilder builder) : base("BertAttention")
    {
        var config = builder.config;
        n_head = config.num_attention_heads;
        d_head = config.head_hidden_size;

        qkv_proj = builder.create_linear(config.hidden_size, d_head * n_head * 3, hasBias: true);
        o_proj = builder.create_linear(n_head * d_head, config.hidden_size, hasBias: true);
        dropout = builder.create_dropout();

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

public class BertBlock : IBertBlock
{
    public IBertAttention attn;
    public IModule attn_ln;
    public IModule ffn;
    public IModule ffn_ln;

    public BertBlock(BertBuilder builder) : base("BertBlock")
    {
        var config = builder.config;
        attn = builder.create_bert_attention();
        attn_ln = builder.create_ln(config.hidden_size);
        ffn = builder.create_bert_ffn();
        ffn_ln = builder.create_ln(config.hidden_size);

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

public class BertPooler : IBertPooler
{
    public IModule dense;
    private readonly List<string> pooling_mode;
    public IModule activation;

    public const string POOLER = "pooler";
    public const string CLS = "cls";
    public const string MEAN = "mean";
    public const string MAX = "max";

    public static List<string> POOLING_MODES = new() { POOLER, CLS, MEAN, MAX };

    public BertPooler(BertBuilder builder) : base("BertPooler")
    {
        pooling_mode = builder.config.pooling_mode;
        validate_pooling_mode();

        dense = pooling_mode.Contains(POOLER)
            ? builder.create_linear(builder.config.hidden_size, builder.config.hidden_size)
            : nn.Identity();

        activation = Activations.get_activation_by_name("tanh");
        RegisterComponents();
    }

    protected void validate_pooling_mode()
    {
        // validate pooling mode
        if (pooling_mode.Count == 0)
            throw new Exception("pooling_mode should not be empty");

        if (pooling_mode.Count > 1 && pooling_mode.Contains(POOLER))
            throw new Exception("pooling_mode should not contain pooler and other modes");

        if (pooling_mode.Any(x => !POOLING_MODES.Contains(x)))
            throw new Exception($"pooling_mode should only contain {string.Join(", ", POOLING_MODES)}");
    }

    public override Tensor forward(Tensor x, Tensor? attention_mask = null)
    {
        using var scope = torch.NewDisposeScope();

        var results = new List<Tensor>();

        if (pooling_mode.Contains(POOLER))
        {
            var first_word = x[.., 0];
            var pooled = dense.call(first_word);
            return scope.MoveToOuter(activation.call(pooled));
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

public record BertModelInput : IBatchEncoding
{
    public Tensor? input_ids { get; set; }
    public Tensor? input_embeddings { get; set; }
    public Tensor? attention_mask { get; set; }
    public Tensor? token_type_ids { get; set; }
    public Tensor? position_ids { get; set; }
}

public record BertModelOutput : IBatchEncoding
{
    public Tensor last_hidden_state { get; set; } = null!;
    public Tensor? pooler_output { get; set; }
    public Tensor? classifier_output { get; set; }
}

public class BertModel : nn.Module<BertModelInput, BertModelOutput>
{
    public BertConfig config;
    public IBertEmbeddings embedding;
    public ModuleList<IBertBlock> layers;
    public IBertPooler? pooler;
    public IModule? classifier;

    public const string SEQUENCE = "sequence";
    public const string TOKEN = "token";

    public BertModel(BertBuilder builder) : base("BertModel")
    {
        config = builder.config;

        embedding = builder.create_bert_embeddings();
        layers = nn.ModuleList(
            Enumerable.Range(0, config.num_layers)
                .Select(index => builder.create_bert_block()).ToArray());

        if (config.pooling_mode.Count > 0)
            pooler = builder.create_bert_pooler();

        if (config.classifier_classes > 0)
            classifier = builder.create_linear(config.hidden_size, config.classifier_classes);

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

public class BertEncoder : MaskedLM
{
    public torch.Device? device { get; protected set; }
    public BertModel model { get; init; }
    public BertConfig model_config { get; init; }
    public WordPiece tokenizer { get; init; }
    public WordPieceConfig tokenizer_config { get; init; }

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
        var (model, model_config) = model_from_pretrained<BertModel, BertConfig, BertBuilder>(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained<WordPiece, WordPieceConfig>(path);
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

    protected override List<float> encode_tokens(List<int> tokens)
    {
        using var scope = torch.NewDisposeScope();

        var input_ids = torch.tensor(tokens, dtype: torch.int64, device: device).unsqueeze(0);
        var type_ids = torch.zeros_like(input_ids);

        model.eval();
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

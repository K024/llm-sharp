using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Microsoft.Extensions.FileSystemGlobbing;
using TorchSharp;

namespace llm_sharp.LLM.Utils;

using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using History = List<(string query, string answer)>;

public record GenerationConfig
{
    public virtual int max_sequence_length { get; set; } = 2000;
    public virtual int max_generated_tokens { get; set; } = 500;
    public virtual int top_k { get; set; } = 100;
    public virtual double top_p { get; set; } = 0.8;
    public virtual double temperature { get; set; } = 1.0;
    public virtual List<string> eos_tokens { get; set; } = new();
}

public abstract class LLM
{
    public virtual bool can_chat => false;

    public virtual IEnumerable<string> chat(History history, string input, GenerationConfig? config = null)
    {
        throw new NotImplementedException();
    }

    public virtual string chat_response(History history, string input, GenerationConfig? config = null)
    {

        var response = new StringBuilder();
        foreach (var output in chat(history, input, config))
        {
            response.Append(output);
        }
        return response.ToString();
    }

    public virtual IAsyncEnumerable<string> chat_async(History history, string input, GenerationConfig? config = null)
    {
        var channel = Channel.CreateUnbounded<string>();
        Task.Factory.StartNew(() =>
        {
            foreach (var output in chat(history, input, config))
            {
                channel.Writer.WriteAsync(output).AsTask().Wait();
            }
            channel.Writer.Complete();

        }, TaskCreationOptions.LongRunning);
        return channel.Reader.ReadAllAsync();
    }

    public virtual bool can_encode => false;

    public virtual IList<float> encode(string text)
    {
        throw new NotImplementedException();
    }

    public virtual Task<IList<float>> encode_async(string text)
    {
        return Task.Factory.StartNew(() =>
        {
            return encode(text);
        }, TaskCreationOptions.LongRunning);
    }

    public virtual bool print_perf()
    {
        throw new NotImplementedException();
    }

    public static LLM from_pretrained(
        string classHint,
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null,
        Assembly? assembly = null)
    {
        var types = (assembly ?? typeof(LLM).Assembly).GetTypes();

        var type = types.Where(x =>
            x.Name == classHint
            && x.IsClass
            && !x.IsAbstract
            && x.IsSubclassOf(typeof(LLM))
        )
            .FirstOrDefault()
            ?? throw new Exception($"Unable to find class '{classHint}'. Try give the correct Assembly");

        var type_from_pretrained = type.GetMethod(
            "from_pretrained",
            BindingFlags.Static | BindingFlags.Public,
            new[] { typeof(string), typeof(torch.ScalarType?), typeof(torch.Device) }
        ) ?? throw new Exception($"The LLM should have a public static from_pretrained(string, torch.ScalarType?, torch.Device?) method");

        var result = type_from_pretrained.Invoke(null, new object?[] { path, dtype, device })
            ?? throw new Exception($"Unable to create class {classHint} with from_pretrained");

        return (LLM)result;
    }
}

public abstract class LLM<TModel, TModelConfig, TTokenizer, TTokenizerConfig> : LLM
    where TModel : class
    where TModelConfig : class
    where TTokenizer : class
    where TTokenizerConfig : class
{
    public const string model_config_file = "model_config.json";
    public const string tokenizer_config_file = "tokenizer_config.json";
    public const string model_weights_pattern = "*.safetensors";

    public static (TModel, TModelConfig) model_from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var createModel = typeof(TModel).GetConstructor(
            new[] { typeof(TModelConfig), typeof(torch.ScalarType?), typeof(torch.Device) }
        ) ?? throw new Exception($"The Model should have a constructor(TModelConfig, torch.ScalarType?, torch.Device?)");

        var modelConfig = JsonSerializer.Deserialize<TModelConfig>(
            File.ReadAllBytes(Path.Combine(path, model_config_file))
        ) ?? throw new ArgumentException(nameof(TModelConfig));

        var model = (TModel)createModel.Invoke(new object?[] { modelConfig, dtype, device })
            ?? throw new NullReferenceException();

        var state_dict = (model as torch.nn.Module)!.state_dict();
        var weight_files = new Matcher().AddInclude(model_weights_pattern).GetResultsInFullPath(path);
        Safetensors.load_state_dict(state_dict, weight_files.ToArray());

        return (model, modelConfig);
    }

    public static (TTokenizer, TTokenizerConfig) tokenizer_from_pretrained(
        string path)
    {
        var createTokenizer = typeof(TTokenizer).GetConstructor(
            new[] { typeof(TTokenizerConfig) }
        ) ?? throw new Exception($"The Model should have a constructor(TTokenizerConfig)");

        var tokenizerConfig = JsonSerializer.Deserialize<TTokenizerConfig>(
            File.ReadAllBytes(Path.Combine(path, tokenizer_config_file))
        ) ?? throw new ArgumentException(nameof(TTokenizerConfig));

        var tokenizer = (TTokenizer)createTokenizer.Invoke(new object?[] { tokenizerConfig })
            ?? throw new NullReferenceException();

        return (tokenizer, tokenizerConfig);
    }
}

public abstract class GenerativeLM<TModel, TModelConfig, TTokenizer, TTokenizerConfig, TState>
        : LLM<TModel, TModelConfig, TTokenizer, TTokenizerConfig>
    where TModel : class
    where TModelConfig : class
    where TTokenizer : class
    where TTokenizerConfig : class
    where TState : class, IDisposable
{
    public override bool can_chat => true;

    public static Tensor top_p_top_k_sampling(
        Tensor logits,
        int top_k = 100,
        double top_p = 0.8,
        double temperature = 1.0)
    {
        using var scope = torch.NewDisposeScope();

        // top_k
        var probs = torch.softmax(logits.to(torch.float32) / temperature, dim: -1);
        var (sorted_probs, indices) = torch.sort(probs, dim: -1, descending: true);

        probs = sorted_probs[TensorIndex.Ellipsis, ..top_k];
        indices = indices[TensorIndex.Ellipsis, ..top_k];

        // top_p
        var cumsum = torch.cumsum(probs, dim: -1);
        probs[(cumsum - probs) > top_p] = 0.0;
        probs = probs / torch.sum(probs, dim: -1, keepdim: true);

        // sample
        var next_token = torch.multinomial(probs, num_samples: 1);

        var output = torch.gather(indices, dim: -1, index: next_token);

        return scope.MoveToOuter(output[TensorIndex.Ellipsis, 0]);
    }

    /// <summary>
    /// Apply template and tokenize input into ids
    /// </summary>
    protected abstract List<int> prepare_input(History history, string input);

    /// <summary>
    /// Return a logits tensor with shape (vocab_size) 
    /// </summary>
    protected abstract (int next_token, TState? state) generate_step(List<int> tokens, TState? state, GenerationConfig config);

    /// <summary>
    /// Decode output
    /// </summary>
    protected abstract string decode_output(List<int> tokens);

    /// <summary>
    /// Return eos token ids
    /// </summary>
    protected abstract List<int> get_eos_tokens();


    protected (int prefix, int gen, List<double> times)? last_generation_perf;

    public override IEnumerable<string> chat(History history, string input, GenerationConfig? config = null)
    {
        var state = (TState?)null;
        var initial = prepare_input(history, input);

        var input_ids = initial.ToList();
        var generated = new List<int>();
        var generated_str = "";
        config ??= new();

        try
        {
            var eos_tokens = get_eos_tokens();
            var generation_time = new List<double>();
            var stop_watch = new Stopwatch();

            while (
                input_ids.Count < config.max_sequence_length &&
                generated.Count < config.max_generated_tokens)
            {
                using var no_grad = torch.no_grad();

                stop_watch.Reset();
                stop_watch.Start();

                var (next_token, next_state) = generate_step(input_ids, state, config);

                stop_watch.Stop();
                generation_time.Add(stop_watch.Elapsed.TotalSeconds);

                state?.Dispose();
                state = next_state;

                if (eos_tokens.Contains(next_token))
                    break;

                input_ids.Add(next_token);
                generated.Add(next_token);

                var decoded = decode_output(generated);
                if (decoded.Length > generated_str.Length && !decoded.EndsWith("�"))
                    yield return decoded[generated_str.Length..];
                generated_str = decoded;
            }
            last_generation_perf = (initial.Count, generated.Count, generation_time);

            var final_decoded = decode_output(generated);
            if (final_decoded.Length > generated_str.Length)
                yield return final_decoded[generated_str.Length..];
        }
        finally
        {
            state?.Dispose();
        }
    }

    public override bool print_perf()
    {
        if (last_generation_perf is null) return false;

        var (prefix, gen, times) = last_generation_perf.Value;

        if (times.Count < 2) return false;

        var first = times.First();
        var rest = times.Skip(1).ToList();

        Console.WriteLine($"Decoder perf:");
        Console.WriteLine($"  len: {prefix}(prefix) + {gen}(gen)");
        Console.WriteLine($" init: {first:0.0000} s");
        Console.WriteLine($"  sum: {rest.Sum():0.0000} s");
        Console.WriteLine($"  gen: {rest.Count / rest.Sum():0.0000} tok/s");
        Console.WriteLine($"  avg: {times.Count / times.Sum():0.0000} tok/s");

        return true;
    }
}

public abstract class MaskedLM<TModel, TModelConfig, TTokenizer, TTokenizerConfig>
        : LLM<TModel, TModelConfig, TTokenizer, TTokenizerConfig>
    where TModel : class
    where TModelConfig : class
    where TTokenizer : class
    where TTokenizerConfig : class
{
    public override bool can_encode => true;

    /// <summary>
    /// Apply template and tokenize input into ids
    /// </summary>
    protected abstract List<int> prepare_input(string input);

    /// <summary>
    /// Return a logits tensor with shape (vocab_size) 
    /// </summary>
    protected abstract IList<float> encode_tokens(List<int> tokens);

    protected List<double> generation_time = new();

    public override IList<float> encode(string text)
    {
        var stop_watch = Stopwatch.StartNew();
        var result = encode_tokens(prepare_input(text));

        stop_watch.Stop();
        generation_time.Add(stop_watch.Elapsed.TotalSeconds);

        while (generation_time.Count > 20)
            generation_time.RemoveAt(0);

        return result;
    }

    public override bool print_perf()
    {
        if (generation_time.Count < 1) return false;

        var times = generation_time;

        Console.WriteLine($"Encoder perf:");
        Console.WriteLine($"  avg: {times.Count / times.Sum():0.0000} sentence/s");

        return true;
    }
}

using System.Diagnostics;
using System.Text;
using System.Threading.Channels;
using TorchSharp;

namespace llm_sharp.LLM.Pretrained;

using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;

public record ChatMessage
{
    public string role { get; set; } = "";
    public string content { get; set; } = "";
}

public record GenerationConfig
{
    public int max_tokens { get; set; } = 500;
    public float top_p { get; set; } = 1f;
    public float temperature { get; set; } = 1f;
    public List<string>? stop_sequences { get; set; }
    public float frequency_penalty { get; set; } = 0f;
    public float presence_penalty { get; set; } = 0f;
    public long? seed { get; set; }
}

public record ChatResponseDelta
{
    public string content { get; set; } = "";
    public int tokens { get; set; } = 0;
    public string? finish_reason { get; set; }
}

public record EncodeResult
{
    public List<float> embedding { get; set; } = new();
    public int tokens { get; set; } = 0;
}

public abstract partial class LanguageModel
{
    public virtual bool can_chat => false;

    public virtual IEnumerable<ChatResponseDelta> chat(List<ChatMessage> messages, GenerationConfig? config = null)
    {
        throw new NotImplementedException();
    }

    public virtual ChatResponseDelta chat_response(List<ChatMessage> messages, GenerationConfig? config = null)
    {
        var delta = new ChatResponseDelta();
        var response = new StringBuilder();
        foreach (var output in chat(messages, config))
        {
            response.Append(output.content);
            delta.tokens += output.tokens;
            delta.finish_reason = output.finish_reason;
        }
        delta.content = response.ToString();
        return delta;
    }

    public virtual IAsyncEnumerable<ChatResponseDelta> chat_async(List<ChatMessage> messages, GenerationConfig? config = null)
    {
        var channel = Channel.CreateUnbounded<ChatResponseDelta>();
        Task.Factory.StartNew(() =>
        {
            foreach (var output in chat(messages, config))
            {
                channel.Writer.WriteAsync(output).AsTask().Wait();
            }
            channel.Writer.Complete();
        }, TaskCreationOptions.LongRunning);
        return channel.Reader.ReadAllAsync();
    }

    public virtual bool can_encode => false;

    public virtual EncodeResult encode(string text)
    {
        throw new NotImplementedException();
    }

    public virtual Task<EncodeResult> encode_async(string text)
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
}

public abstract class GenerativeLM<TState> : LanguageModel
    where TState : class, IDisposable
{
    public override bool can_chat => true;
    public abstract int max_sequence_length { get; }

    public static Tensor top_p_sampling(
        Tensor logits,
        double top_p = 0.8,
        double temperature = 1.0,
        torch.Generator? generator = null)
    {
        using var scope = torch.NewDisposeScope();

        // top_k
        var probs = torch.softmax(logits.to(torch.float32) / temperature, dim: -1);
        var (sorted_probs, indices) = torch.sort(probs, dim: -1, descending: true);

        probs = sorted_probs;
        // probs = sorted_probs[TensorIndex.Ellipsis, ..top_k];
        // indices = indices[TensorIndex.Ellipsis, ..top_k];

        // top_p
        var cumsum = torch.cumsum(probs, dim: -1);
        probs[cumsum > top_p] = 0.0;
        probs = probs / torch.sum(probs, dim: -1, keepdim: true);

        // sample
        var next_token = torch.multinomial(probs, num_samples: 1, generator: generator);

        var output = torch.gather(indices, dim: -1, index: next_token);

        return scope.MoveToOuter(output[TensorIndex.Ellipsis, 0]);
    }

    public static void logits_bias(
        Tensor logits,
        List<int> generated_tokens,
        float frequency_penalty,
        float presence_penalty)
    {
        using var scope = torch.NewDisposeScope();

        if (logits.requires_grad)
            throw new ArgumentException("logits should not require gradients for biasing");

        if (generated_tokens.Count == 0 || (frequency_penalty == 0 && presence_penalty == 0))
            return;

        var generated = generated_tokens.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
        var generated_keys = generated.Keys.ToList();
        var generated_biasis = generated_keys
            .Select(token_id => -frequency_penalty * generated[token_id] - presence_penalty)
            .ToList();

        var indices = torch.tensor(generated_keys, dtype: torch.int64, device: logits.device);
        var biases = torch.tensor(generated_biasis, dtype: logits.dtype, device: logits.device);
        
        logits.scatter_add_(dim: -1, index: indices, src: biases);
    }

    /// <summary>
    /// Apply template and tokenize input into ids
    /// </summary>
    protected abstract List<int> prepare_input(List<ChatMessage> messages);

    /// <summary>
    /// Return a logits tensor with shape (vocab_size) 
    /// </summary>
    protected abstract (int next_token, TState? state) generate_step(List<int> input_tokens, List<int> generated_tokens, TState? state, GenerationConfig config);

    /// <summary>
    /// Decode output
    /// </summary>
    protected abstract string decode_output(List<int> tokens);

    /// <summary>
    /// Return eos token ids
    /// </summary>
    protected abstract List<int> get_eos_tokens();


    protected (int prefix, int gen, List<double> times)? last_generation_perf;

    public override IEnumerable<ChatResponseDelta> chat(List<ChatMessage> messages, GenerationConfig? config = null)
    {
        var state = (TState?)null;
        var input_ids = prepare_input(messages);

        yield return new()
        {
            tokens = input_ids.Count,
        };

        var generated = new List<int>();
        var decoded_str = "";
        var decoded_tokens = 0;
        config ??= new();

        try
        {
            var eos_tokens = get_eos_tokens();
            var generation_time = new List<double>();
            var stop_watch = new Stopwatch();
            var finish_reason = "length";

            while (
                input_ids.Count + generated.Count < max_sequence_length &&
                generated.Count < config.max_tokens)
            {
                stop_watch.Reset();
                stop_watch.Start();

                var (next_token, next_state) = generate_step(input_ids, generated, state, config);

                stop_watch.Stop();
                generation_time.Add(stop_watch.Elapsed.TotalSeconds);

                state?.Dispose();
                state = next_state;

                if (eos_tokens.Contains(next_token))
                {
                    finish_reason = "stop";
                    break;
                }

                generated.Add(next_token);

                var decoded = decode_output(generated);
                if (!decoded.EndsWith("�") && decoded.Length > decoded_str.Length)
                {
                    yield return new()
                    {
                        content = decoded.Substring(decoded_str.Length),
                        tokens = generated.Count - decoded_tokens,
                    };
                    decoded_str = decoded;
                    decoded_tokens = generated.Count;
                }

                if (config.stop_sequences is not null && config.stop_sequences.Any(s => decoded.EndsWith(s)))
                {
                    finish_reason = "stop";
                    break;
                }
            }
            last_generation_perf = (input_ids.Count, generated.Count, generation_time);

            var final_decoded = decode_output(generated);
            if (final_decoded.Length > decoded_str.Length)
            {
                yield return new()
                {
                    content = final_decoded.Substring(decoded_str.Length),
                    tokens = generated.Count - decoded_tokens,
                };
            }
            yield return new()
            {
                finish_reason = finish_reason,
            };
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

public abstract class MaskedLM : LanguageModel
{
    public override bool can_encode => true;

    /// <summary>
    /// Apply template and tokenize input into ids
    /// </summary>
    protected abstract List<int> prepare_input(string input);

    /// <summary>
    /// Return a logits tensor with shape (vocab_size) 
    /// </summary>
    protected abstract List<float> encode_tokens(List<int> tokens);

    protected List<double> generation_time = new();

    public override EncodeResult encode(string text)
    {
        using var no_grad = torch.no_grad();
        var stop_watch = Stopwatch.StartNew();
        var input_ids = prepare_input(text);
        var result = encode_tokens(input_ids);

        stop_watch.Stop();
        generation_time.Add(stop_watch.Elapsed.TotalSeconds);

        while (generation_time.Count > 20)
            generation_time.RemoveAt(0);

        return new()
        {
            embedding = result,
            tokens = input_ids.Count,
        };
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

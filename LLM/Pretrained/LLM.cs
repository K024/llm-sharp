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
    public virtual int max_sequence_length { get; set; } = 2000;
    public virtual int max_generated_tokens { get; set; } = 500;
    public virtual float top_p { get; set; } = 1f;
    public virtual float temperature { get; set; } = 1f;
    public virtual List<string>? stop_sequences { get; set; }
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

    public static Tensor top_p_sampling(
        Tensor logits,
        double top_p = 0.8,
        double temperature = 1.0)
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
    protected abstract List<int> prepare_input(List<ChatMessage> messages);

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

    public override IEnumerable<ChatResponseDelta> chat(List<ChatMessage> messages, GenerationConfig? config = null)
    {
        var state = (TState?)null;
        var initial = prepare_input(messages);

        yield return new()
        {
            tokens = initial.Count,
        };

        var input_ids = initial.ToList();
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
                {
                    finish_reason = "stop";
                    break;
                }

                input_ids.Add(next_token);
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
            }
            last_generation_perf = (initial.Count, generated.Count, generation_time);

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

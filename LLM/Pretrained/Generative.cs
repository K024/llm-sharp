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
    public CancellationToken? cancellation { get; set; }
}

public record ChatResponseDelta
{
    public string content { get; set; } = "";
    public int tokens { get; set; } = 0;
    public string? finish_reason { get; set; }
}

public abstract class GenerativeLM : PretrainedModel
{
    public abstract IEnumerable<ChatResponseDelta> chat(List<ChatMessage> messages, GenerationConfig? config = null);

    public virtual IAsyncEnumerable<ChatResponseDelta> chat_async(List<ChatMessage> messages, GenerationConfig? config = null)
    {
        var channel = Channel.CreateUnbounded<ChatResponseDelta>();
        Task.Factory.StartNew(() =>
        {
            try
            {
                foreach (var output in chat(messages, config))
                {
                    channel.Writer.WriteAsync(output).AsTask().Wait();
                }
                channel.Writer.Complete();
            }
            catch (Exception e)
            {
                channel.Writer.Complete(e);
            }
        }, TaskCreationOptions.LongRunning);
        return channel.Reader.ReadAllAsync();
    }

    public abstract bool print_perf();
}

public abstract class GenerativeLM<TState> : GenerativeLM
    where TState : class, IDisposable
{
    public abstract int max_sequence_length { get; }

    protected abstract List<int> eos_tokens { get; }

    /// <summary>
    /// Apply template and tokenize input into ids
    /// </summary>
    protected abstract List<int> prepare_input(List<ChatMessage> messages);

    /// <summary>
    /// Create initial state for generation
    /// </summary>
    protected abstract TState prepare_init_state(List<int> input_tokens, GenerationConfig config);

    /// <summary>
    /// Return sampled next token id
    /// </summary>
    protected abstract int generate_step(List<int> input_tokens, List<int> generated_tokens, TState state, GenerationConfig config);

    /// <summary>
    /// Decode output
    /// </summary>
    protected abstract string decode_output(List<int> tokens);

    public static Tensor top_p_sampling(
        Tensor logits,
        double top_p = 0.8,
        double temperature = 1.0,
        torch.Generator? generator = null)
    {
        using var scope = torch.NewDisposeScope();

        // top_k
        temperature = Math.Min(Math.Max(temperature, 1e-5), 1e2);
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
        if (generator is not null)
            probs = probs.to(generator.device); // generator won't work on CUDA
        var next_token = torch.multinomial(probs, num_samples: 1, generator: generator);
        next_token = next_token.to(indices.device);

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

    protected (int prefix, int gen, List<double> times)? last_generation_perf;

    public override IEnumerable<ChatResponseDelta> chat(List<ChatMessage> messages, GenerationConfig? config = null)
    {
        config ??= new();
        var input_ids = prepare_input(messages);

        yield return new()
        {
            tokens = input_ids.Count,
        };

        var generated_ids = new List<int>();
        var decoded_str = "";
        var decoded_tokens = 0;

        var eos_tokens = this.eos_tokens;
        var max_sequence_length = this.max_sequence_length;
        var finish_reason = "length";

        var generation_time = new List<double>();
        var stop_watch = new Stopwatch();

        using var state = prepare_init_state(input_ids, config);

        while (
            input_ids.Count + generated_ids.Count < max_sequence_length &&
            generated_ids.Count < config.max_tokens)
        {
            stop_watch.Reset();
            stop_watch.Start();

            var next_token = generate_step(input_ids, generated_ids, state, config);

            stop_watch.Stop();
            generation_time.Add(stop_watch.Elapsed.TotalSeconds);

            if (config.cancellation?.IsCancellationRequested ?? false)
            {
                finish_reason = "stop";
                Console.WriteLine("LLM: chat_async cancelled");
                break;
            }
            if (eos_tokens.Contains(next_token))
            {
                finish_reason = "stop";
                break;
            }

            generated_ids.Add(next_token);

            var decoded = decode_output(generated_ids);
            if (!decoded.EndsWith("�") && decoded.Length > decoded_str.Length)
            {
                yield return new()
                {
                    content = decoded[decoded_str.Length..],
                    tokens = generated_ids.Count - decoded_tokens,
                };
                decoded_str = decoded;
                decoded_tokens = generated_ids.Count;
            }

            if (config.stop_sequences is not null && config.stop_sequences.Any(s => decoded.EndsWith(s)))
            {
                finish_reason = "stop";
                break;
            }
        }
        last_generation_perf = (input_ids.Count, generated_ids.Count, generation_time);

        var final_decoded = decode_output(generated_ids);
        if (final_decoded.Length > decoded_str.Length)
        {
            yield return new()
            {
                content = final_decoded[decoded_str.Length..],
                tokens = generated_ids.Count - decoded_tokens,
            };
        }
        yield return new()
        {
            finish_reason = finish_reason,
        };
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

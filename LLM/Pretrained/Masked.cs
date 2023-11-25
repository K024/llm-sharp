using System.Diagnostics;
using System.Text;
using System.Threading.Channels;
using TorchSharp;

namespace llm_sharp.LLM.Pretrained;

using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;

public record EncodeResult
{
    public List<float> embedding { get; set; } = new();
    public int tokens { get; set; } = 0;
}

public abstract class MaskedLM : PretrainedModel
{
    /// <summary>
    /// Apply template and tokenize input into ids
    /// </summary>
    protected abstract List<int> prepare_input(string input);

    /// <summary>
    /// Return a logits tensor with shape (vocab_size) 
    /// </summary>
    protected abstract List<float> encode_tokens(List<int> tokens);

    protected List<double> generation_time = new();

    public virtual EncodeResult encode(string text)
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

    public virtual Task<EncodeResult> encode_async(string text)
    {
        return Task.Factory.StartNew(() =>
        {
            return encode(text);
        }, TaskCreationOptions.LongRunning);
    }

    public virtual bool print_perf()
    {
        if (generation_time.Count < 1) return false;

        var times = generation_time;

        Console.WriteLine($"Encoder perf:");
        Console.WriteLine($"  avg: {times.Count / times.Sum():0.0000} sentence/s");

        return true;
    }
}

using TorchSharp;
using llm_sharp.LLM.Tokenizers;
using llm_sharp.LLM.Pretrained;

namespace llm_sharp.LLM.Models;

using Tensor = torch.Tensor;

public class Qwen : GenerativeLM<Qwen.QwenState>
{
    public class QwenState : IDisposable
    {
        public List<(Tensor k_cache, Tensor v_cache)> past_key_values { get; set; } = new();

        public void Dispose()
        {
            past_key_values.SelectMany(x => new[] { x.k_cache, x.v_cache })
                .ToList().ForEach(x => x.Dispose());
        }
    }

    public torch.Device? device { get; protected set; }
    public LlamaModel model { get; init; }
    public LlamaConfig model_config { get; init; }
    public TikToken tokenizer { get; init; }
    public TikTokenConfig tokenizer_config { get; init; }

#nullable disable
    protected Qwen() { }
#nullable restore

    public virtual Qwen to(torch.Device? device)
    {
        model.to(device);
        this.device = device;
        return this;
    }

    public static Qwen from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var (model, model_config) = model_from_pretrained<LlamaModel, LlamaConfig, LlamaBuilder>(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained<TikToken, TikTokenConfig>(path);
        return new Qwen()
        {
            device = device,
            model = model,
            model_config = model_config,
            tokenizer = tokenizer,
            tokenizer_config = tokenizer_config,
        };
    }

    protected override List<int> prepare_input(List<(string query, string answer)> history, string input)
    {
        var prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";

        foreach (var (query, answer) in history)
        {
            prompt += $"\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>";
        }
        prompt += $"\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n";

        return tokenizer.encode_text(prompt);
    }

    protected override (int next_token, QwenState? state) generate_step(List<int> tokens, QwenState? state, GenerationConfig config)
    {
        using var scope = torch.NewDisposeScope();

        var past_key_values = state?.past_key_values;
        if (past_key_values is not null)
            tokens = tokens.TakeLast(1).ToList();

        var input_ids = torch.tensor(tokens, dtype: torch.int64, device: device).unsqueeze(0);

        var output = model.call(new()
        {
            input_ids = input_ids,
            past_key_values = past_key_values,
        });

        var next = top_p_top_k_sampling(
            output.logits[0, ^1], config.top_k, config.top_p, config.temperature
        );
        var next_token = (int)next.item<long>();

        scope.MoveToOuter(output.current_key_values.SelectMany(x => new[] { x.k_cache, x.v_cache }));
        return (
            next_token,
            new() { past_key_values = output.current_key_values }
        );
    }

    protected override string decode_output(List<int> tokens)
    {
        return tokenizer.decode_text(tokens);
    }

    protected override List<int> get_eos_tokens()
    {
        return tokenizer.eos_ids;
    }
}

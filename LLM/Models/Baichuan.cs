using TorchSharp;
using llm_sharp.LLM.Tokenizers;
using llm_sharp.LLM.Pretrained;
using llm_sharp.LLM.Layers;
using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;

namespace llm_sharp.LLM.Models;

using nn = torch.nn;
using Tensor = torch.Tensor;
using BaichuanState = Qwen.QwenState;

public class Baichuan : GenerativeLM<BaichuanState>
{
    public torch.Device? device { get; protected set; }
    public LlamaModel model { get; init; }
    public LlamaConfig model_config { get; init; }
    public SentencePieceBPE tokenizer { get; init; }
    public SentencePieceBPEConfig tokenizer_config { get; init; }

#nullable disable
    protected Baichuan() { }
#nullable restore

    public virtual Baichuan to(torch.Device? device)
    {
        model.to(device);
        this.device = device;
        return this;
    }

    public static void norm_head(CustomLinear lm_head)
    {
        using var scope = torch.NewDisposeScope();
        var weight = lm_head.weight;
        scope.Include(weight);

        lm_head.weight = nn.Parameter(weight * (weight.pow(2).sum(dim: 1, keepdim: true) + 1e-7).rsqrt());
        scope.Detach(lm_head.weight);
    }

    public static void enhance_alibi(Alibi alibi)
    {
        using var scope = torch.NewDisposeScope();
        var mask = alibi.mask;
        scope.Include(mask);

        alibi.mask = -torch.flip(mask, 2);
        scope.Detach(alibi.mask);
    }

    public static Baichuan from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var (model, model_config) = model_from_pretrained<LlamaModel, LlamaConfig, LlamaBuilder>(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained<SentencePieceBPE, SentencePieceBPEConfig>(path);

        norm_head((CustomLinear)model.lm_head);
        if (model.alibi is Alibi alibi)
            enhance_alibi(alibi);

        return new Baichuan()
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
        var prompt = "";

        foreach (var (query, answer) in history)
        {
            prompt += $"<reserved_106>{query}<reserved_107>{answer}";
        }
        prompt += $"<reserved_106>{input}<reserved_107>";

        return tokenizer.encode_text(prompt);
    }

    protected override (int next_token, BaichuanState? state) generate_step(List<int> tokens, BaichuanState? state, GenerationConfig config)
    {
        using var scope = torch.NewDisposeScope();

        var past_key_values = state?.past_key_values;
        if (past_key_values is not null)
            tokens = tokens.TakeLast(1).ToList();

        var input_ids = torch.tensor(tokens, dtype: torch.int64, device: device).unsqueeze(0);

        model.eval();
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

public class BaichuanAwq : Baichuan
{
    public static new Baichuan from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var (model, model_config) = model_from_pretrained<LlamaModel, LlamaConfig, LlamaAwqBuilder>(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained<SentencePieceBPE, SentencePieceBPEConfig>(path);

        QwenAwq.convert_turbomind(model);
        norm_head((CustomLinear)model.lm_head);
        if (model.alibi is Alibi alibi)
            enhance_alibi(alibi);

        return new BaichuanAwq()
        {
            device = device,
            model = model,
            model_config = model_config,
            tokenizer = tokenizer,
            tokenizer_config = tokenizer_config,
        };
    }
}

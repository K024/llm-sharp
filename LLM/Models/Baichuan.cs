using TorchSharp;
using llm_sharp.LLM.Tokenizers;
using llm_sharp.LLM.Pretrained;
using llm_sharp.LLM.Layers;

namespace llm_sharp.LLM.Models;

using nn = torch.nn;

public class Baichuan : AbstractLlama
{
    public SentencePieceBPE tokenizer { get; init; }
    public SentencePieceBPEConfig tokenizer_config { get; init; }
    protected override List<int> eos_tokens => tokenizer.eos_ids;

#nullable disable
    protected Baichuan() { }
#nullable restore

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
        var (model, model_config) = model_from_pretrained<LlamaModel, LlamaConfig, LlamaFastBuilder>(path, dtype, device);
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

    protected override List<int> prepare_input(List<ChatMessage> messages)
    {
        var prompt = "";
        foreach (var message in messages)
        {
            prompt += message.role switch
            {
                SYSTEM => $"{message.content}",
                USER => $"<reserved_106>{message.content}",
                ASSISTANT => $"<reserved_107>{message.content}",
                _ => "",
            };
        }
        prompt += "<reserved_107>";
        return tokenizer.encode_text(prompt);
    }

    protected override string decode_output(List<int> tokens)
    {
        return tokenizer.decode_text(tokens);
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

        LlamaAwq.convert_turbomind(model);
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

using TorchSharp;
using llm_sharp.LLM.Tokenizers;
using llm_sharp.LLM.Pretrained;

namespace llm_sharp.LLM.Models;

public class Qwen : AbstractLlama
{
    public TikToken tokenizer { get; init; }
    public TikTokenConfig tokenizer_config { get; init; }

#nullable disable
    protected Qwen() { }
#nullable restore

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

    protected override List<int> prepare_input(List<ChatMessage> messages)
    {
        var prompt = "";
        foreach (var message in messages)
        {
            prompt += message.role switch
            {
                "system" => $"<|im_start|>system\n{message.content}<|im_end|>",
                "user" => $"<|im_start|>user\n{message.content}<|im_end|>",
                "assistant" => $"<|im_start|>assistant\n{message.content}<|im_end|>",
                _ => ""
            };
        }
        prompt += "<|im_start|>assistant\n";
        return tokenizer.encode_text(prompt);
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

public class QwenAwq : Qwen
{
    public static new Qwen from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var (model, model_config) = model_from_pretrained<LlamaModel, LlamaConfig, LlamaAwqBuilder>(path, dtype, device);
        var (tokenizer, tokenizer_config) = tokenizer_from_pretrained<TikToken, TikTokenConfig>(path);

        LlamaAwq.convert_turbomind(model);

        return new QwenAwq()
        {
            device = device,
            model = model,
            model_config = model_config,
            tokenizer = tokenizer,
            tokenizer_config = tokenizer_config,
        };
    }
}

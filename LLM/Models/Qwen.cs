using TorchSharp;
using llm_sharp.LLM.Tokenizers;
using llm_sharp.LLM.Pretrained;

namespace llm_sharp.LLM.Models;

public class Qwen : AbstractLlama
{
    public TikToken tokenizer { get; init; }
    public TikTokenConfig tokenizer_config { get; init; }
    protected override List<int> eos_tokens => tokenizer.eos_ids;

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

        if (messages.All(x => x.role != "system"))
            prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";

        foreach (var message in messages)
        {
            prompt += message.role switch
            {
                "system" => $"<|im_start|>system\n{message.content}<|im_end|>\n",
                "user" => $"<|im_start|>user\n{message.content}<|im_end|>\n",
                "assistant" => $"<|im_start|>assistant\n{message.content}<|im_end|>\n",
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

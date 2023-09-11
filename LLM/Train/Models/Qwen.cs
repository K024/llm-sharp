using TorchSharp;
using llm_sharp.LLM.Models;
using llm_sharp.LLM.Utils;
using System.Threading.Channels;
using TorchSharp.Modules;
using System.Text.Json;

namespace llm_sharp.LLM.Train;


public class QwenTrainer : Trainer
{
    public QwenTrainer(TrainerConfig config)
    {
        _config = config;
        device = torch.device("cuda");

        llm = Qwen.from_pretrained(
            config.model_path,
            config.use_bfloat16 ? torch.bfloat16 : torch.float16,
            device
        );
        dataset = Dataset<TextDataLine>.from_jsonl(config.data_path);
    }
    protected torch.Device device;

    protected TrainerConfig _config;

    protected Qwen llm;

    protected Dataset<TextDataLine> dataset;

    public override TrainerConfig config => _config;

    public override torch.nn.Module model => llm.model;

    public override int compute_batches()
    {
        return dataset.Count / config.batch_size;
    }

    public override torch.Tensor compute_loss(params torch.Tensor[] inputs)
    {
        using var scope = torch.NewDisposeScope();
        var (input_ids, attention_mask) = inputs;

        scope.Include(input_ids);
        scope.Include(attention_mask);

        var outputs = llm.model.call(new LLaMAModelInput()
        {
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = input_ids.masked_fill(attention_mask == 0, -100),
        });
        var loss = outputs.loss!;

        return scope.MoveToOuter(loss);
    }

    public override async Task load_batches(ChannelWriter<torch.Tensor[]> result)
    {
        foreach (var batch in dataset.iterate_batch(config.batch_size, seed: -1))
        {
            var inputs = batch.Select(x => llm.tokenizer.encode_text(x.text)).ToList();
            var masks = inputs.Select(x => x.Select(y => 1).ToList()).ToList();

            var max_length = inputs.Select(x => x.Count).Max();

            foreach (var ids in inputs)
                ids.AddRange(Enumerable.Repeat(0, max_length - ids.Count));

            foreach (var mask in masks)
                mask.AddRange(Enumerable.Repeat(0, max_length - mask.Count));

            var scope = torch.NewDisposeScope();

            var input_ids = torch.tensor(
                inputs.SelectMany(x => x).ToList(),
                torch.int64, device
            ).reshape(inputs.Count, max_length);

            var attention_mask = torch.tensor(
                masks.SelectMany(x => x).ToList(),
                torch.int64, device
            ).reshape(masks.Count, max_length);

            scope.Detach(input_ids, attention_mask);
            await result.WriteAsync(new[] { input_ids, attention_mask });
        }
        result.Complete();
    }

    public override void save_checkpoint(int step, bool finished)
    {
        var save_path = finished ? config.output_dir : Path.Combine(config.output_dir, $"checkpoint_{step}");
        llm.save_pretrained(save_path);
    }

    public override IEnumerable<Parameter> get_trainable_parameters()
    {
        return llm.model.parameters();
    }
}

public class QwenLoRATrainer : QwenTrainer
{
    public const string LORA_SCOPE = "qwen-lora-trainer";
    protected LoRA.LoRAConfig lora_config;

    public QwenLoRATrainer(TrainerConfig config) : base(config)
    {
        lora_config = new LoRA.LoRAConfig();
        var wrapped_modules = new List<string>()
        {
            "word_embedding",
            "layers.{layer}.attn.qkv_proj",
            "lm_head",
        };

        using var lora = LoRA.lora_scope(LORA_SCOPE);

        LoRA.wrap_module(llm.model, wrapped_modules, lora_config);
        var (total, trained) = LoRA.mark_trainable(llm.model);

        Console.WriteLine($"Total params: {total}, trained params: {trained}");
        Console.WriteLine($"Trainable percentage: {trained / (double)total:P3}");
    }

    public override void save_checkpoint(int step, bool finished)
    {
        using var lora = LoRA.lora_scope(LORA_SCOPE);

        var save_path = finished ? config.output_dir : Path.Combine(config.output_dir, $"checkpoint_{step}");
        Directory.CreateDirectory(save_path);

        var state_dict = LoRA.get_lora_state_dict(llm.model);
        var save_name = Path.Combine(save_path, "lora_weights.safetensors");
        Safetensors.save_tensors(state_dict, save_name);

        File.WriteAllText(
            Path.Join(save_path, "lora_config.json"),
            JsonSerializer.Serialize(lora_config, Utils.LLM.TupleJsonSerializerOptions)
        );
    }

    public override IEnumerable<Parameter> get_trainable_parameters()
    {
        using var lora = LoRA.lora_scope(LORA_SCOPE);
        return LoRA.get_lora_state_dict(llm.model).Values.Select(x => torch.nn.Parameter(x));
    }

    public override torch.Tensor compute_loss(params torch.Tensor[] inputs)
    {
        using var lora = LoRA.lora_scope(LORA_SCOPE);
        return base.compute_loss(inputs);
    }
}

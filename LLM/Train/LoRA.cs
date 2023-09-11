using TorchSharp;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Models;
using TorchSharp.Modules;

namespace llm_sharp.LLM.Train;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;


public static class LoRA
{
    public interface ILoRALayer
    {
        public void config_lora(LoRAConfig config);
        public LoRAModule? current_lora { get; }
    }

    public record LoRAConfig
    {
        public long hidden_size { get; set; } = 16;
        public double dropout { get; set; } = 0;
        public double alpha { get; set; } = 1;
    }

    public class LoRAModule : nn.Module
    {
        public LoRAModule() : base("LoRAModule") { }
#nullable disable
        public double dropout;
        public double scaling;
        public Parameter lora_a;
        public Parameter lora_b;
#nullable restore
    }

    private class LoRAScope : IDisposable
    {
        private string? last;
        public LoRAScope(string name)
        {
            last = activeLora.Value;
            activeLora.Value = name;
        }
        public void Dispose()
        {
            activeLora.Value = last;
        }
    }

    private static ThreadLocal<string?> activeLora = new();
    public static string? active_lora => activeLora.Value;
    public static volatile bool debug_mode = false;

    public static IDisposable lora_scope(string name)
    {
        return new LoRAScope(name);
    }

    public static Dictionary<string, Tensor> get_lora_state_dict(nn.Module module)
    {
        var result = new Dictionary<string, Tensor>();
        foreach (var (name, child) in module.named_modules())
        {
            if (child is ILoRALayer lora)
            {
                var lora_module = lora.current_lora
                    ?? throw new Exception("No active LoRA module");
                lora_module.state_dict(result, name);
            }
        }
        return result;
    }

    public static void wrap_module(nn.Module module, IReadOnlyList<string> patterns, LoRAConfig config)
    {
        var name_matcher = new StateDictConverter.TemplateNameConverter(
            patterns.ToDictionary(x => x, x => x));

        var named_modules = module.named_modules().ToList();

        foreach (var (name, child) in named_modules)
        {
            if (child is ILoRALayer lora)
            {
                lora.config_lora(config);
                continue;
            }

            if (name_matcher.TryConvert(name, out _))
            {
                var parent_name = string.Join('.', name.Split('.').SkipLast(1));
                var field_name = name.Split('.').Last();

                var parent = named_modules
                    .Where(x => x.Item1 == parent_name).Select(x => x.Item2)
                    .FirstOrDefault() ?? throw new Exception($"Unable to find parent module '{parent_name}'");

                var field = parent.GetType().GetField(field_name)
                    ?? throw new Exception($"Unable to find field '{field_name}' in module '{parent_name}'");

                if (module is CustomLinear linear)
                {
                    var lora_linear = new LoRALinear(linear);

                    field.SetValue(parent, lora_linear);
                    parent.register_module(field_name, lora_linear);

                    lora_linear.config_lora(config);
                }
                else if (module is CustomEmbedding embedding)
                {
                    var lora_embedding = new LoRAEmbedding(embedding);

                    field.SetValue(parent, lora_embedding);
                    parent.register_module(field_name, lora_embedding);

                    lora_embedding.config_lora(config);
                }
                else
                {
                    throw new Exception($"Unsupported module type '{module.GetType().Name}'");
                }
            }
        }
    }

    public static (int, int) mark_trainable(nn.Module module)
    {
        using var no_grad = torch.no_grad();

        var state_dict = module.state_dict();
        var lora_state_dict = get_lora_state_dict(module);

        foreach (var pair in state_dict)
            pair.Value.requires_grad_(false);

        foreach (var pair in lora_state_dict)
            pair.Value.requires_grad_(true);

        var total_params = (int)state_dict.Select(x => x.Value.numel()).Sum();
        var lora_params = (int)lora_state_dict.Select(x => x.Value.numel()).Sum();

        return (total_params, lora_params);
    }
}

public class LoRALinear : CustomLinear, LoRA.ILoRALayer
{
    public LoRALinear(
        long inputSize,
        long outputSize,
        bool hasBias = true,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base(inputSize, outputSize, hasBias, dtype, device) { }

    public LoRALinear(CustomLinear linear) : base(linear.weight, linear.bias) { }

    public Dictionary<string, LoRA.LoRAModule> lora_modules = new();
    public LoRA.LoRAModule? current_lora =>
        LoRA.active_lora is not null
            ? lora_modules.GetValueOrDefault(LoRA.active_lora)
            : null;

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();

        var result = base.forward(x);

        var lora = current_lora;
        if (lora is not null)
        {
            var input = F.dropout(x, lora.dropout, training);
            var output = F.linear(F.linear(input, lora.lora_a), lora.lora_b) * lora.scaling;
            result += output;
        }
        else if (LoRA.debug_mode)
        {
            throw new Exception("No active LoRA module");
        }
        return scope.MoveToOuter(result);
    }

    public void config_lora(LoRA.LoRAConfig config)
    {
        using var no_grad = torch.no_grad();

        var name = LoRA.active_lora ?? throw new Exception("No active LoRA module");

        if (config.hidden_size <= 0)
            throw new Exception("hidden_size must be > 0");

        var (outputSize, inputSize) = weight.shape;
        var lora = new LoRA.LoRAModule()
        {
            dropout = config.dropout,
            scaling = config.alpha / config.hidden_size,
            lora_a = nn.Parameter(torch.zeros(
                config.hidden_size, inputSize,
                dtype: weight.dtype, device: weight.device)),
            lora_b = nn.Parameter(torch.zeros(
                outputSize, config.hidden_size,
                dtype: weight.dtype, device: weight.device))
        };
        nn.init.kaiming_uniform_(lora.lora_a, a: Math.Sqrt(5));

        lora_modules[name] = lora;
    }

    protected override nn.Module _to(torch.Device device, torch.ScalarType dtype)
    {
        foreach (var (_, lora) in lora_modules)
        {
            lora.to(device, dtype);
        }
        return base._to(device, dtype);
    }
}

public class LoRAEmbedding : CustomEmbedding, LoRA.ILoRALayer
{
    public LoRAEmbedding(
        long num_embeddings,
        long embedding_dim,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base(num_embeddings, embedding_dim, dtype, device) { }

    public LoRAEmbedding(CustomEmbedding embedding) : base(embedding.weight) { }

    public Dictionary<string, LoRA.LoRAModule> lora_modules = new();
    public LoRA.LoRAModule? current_lora =>
        LoRA.active_lora is not null
            ? lora_modules.GetValueOrDefault(LoRA.active_lora)
            : null;

    public override Tensor forward(Tensor input)
    {
        using var scope = torch.NewDisposeScope();

        var result = base.forward(input);

        var lora = current_lora;
        if (lora is not null)
        {
            var hidden = lora.lora_a.T[input];
            var output = F.linear(hidden, lora.lora_b) * lora.scaling;
            result += output;
        }
        else if (LoRA.debug_mode)
        {
            throw new Exception("No active LoRA module");
        }
        return scope.MoveToOuter(result);
    }

    public void config_lora(LoRA.LoRAConfig config)
    {
        using var no_grad = torch.no_grad();

        var name = LoRA.active_lora ?? throw new Exception("No active LoRA module");

        if (config.hidden_size <= 0)
            throw new Exception("hidden_size must be > 0");

        var (num_embeddings, embedding_dim) = weight.shape;
        var lora = new LoRA.LoRAModule()
        {
            dropout = config.dropout,
            scaling = config.alpha / config.hidden_size,
            lora_a = nn.Parameter(torch.zeros(
                config.hidden_size, num_embeddings,
                dtype: weight.dtype, device: weight.device)),
            lora_b = nn.Parameter(torch.zeros(
                embedding_dim, config.hidden_size,
                dtype: weight.dtype, device: weight.device))
        };
        nn.init.normal_(lora.lora_b);

        lora_modules[name] = lora;
    }

    protected override nn.Module _to(torch.Device device, torch.ScalarType dtype)
    {
        foreach (var (_, lora) in lora_modules)
        {
            lora.to(device, dtype);
        }
        return base._to(device, dtype);
    }
}

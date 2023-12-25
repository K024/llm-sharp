using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using F = torch.nn.functional;

public class QLoraLinear : AwqLinear
{
    public long r;
    public float lora_alpha;

    public Tensor lora_A;
    public Tensor lora_B;

    public QLoraLinear(AwqLinear linear, long r, float lora_alpha = 1.0f)
        : base(linear.qweight, linear.qzeros, linear.scales, linear.bias)
    {
        this.r = r;
        this.lora_alpha = lora_alpha;
        var in_features = linear.qweight.shape[0];
        var out_features = linear.qweight.shape[1] * pack_size;
        lora_A = torch.randn(r, in_features, device: linear.scales.device, dtype: linear.scales.dtype);
        lora_B = torch.zeros(out_features, r, device: linear.scales.device, dtype: linear.scales.dtype);
        is_converted_turbomind = linear.is_converted_turbomind;
        register_buffer("lora_A", lora_A);
        register_buffer("lora_B", lora_B);
    }

    public override Tensor forward(Tensor input)
    {
        using var scope = torch.NewDisposeScope();
        var lora_scaling = lora_alpha / r;
        var lora_output = F.linear(F.linear(input, lora_A), lora_B) * lora_scaling;
        return scope.MoveToOuter(base.forward(input) + lora_output);
    }
}

public static class Lora
{
    public static void apply_lora_weights(nn.Module module, string lora_path, float lora_alpha = 1.0f)
    {
        apply_lora_weights(module, new Safetensors(lora_path).load_tensors_dict(), lora_alpha);
    }

    public static void apply_lora_weights(nn.Module module, Dictionary<string, Tensor> lora_weights, float lora_alpha = 1.0f)
    {
        using var nograd = torch.no_grad();
        var dict = module.named_modules().ToDictionary(x => x.name, x => x.module);
        dict[""] = module;

        void set_on_parent(string name, nn.Module value)
        {
            var paths = name.Split('.');
            var module_name = paths.Last();
            var parent_name = string.Join(".", paths.SkipLast(1));
            var parent = dict[parent_name];

            var field = parent.GetType().GetField(module_name);
            if (field is null || !value.GetType().IsAssignableTo(field.FieldType))
                throw new Exception($"Cannot set {name} on {parent_name}");

            field.SetValue(parent, value);
            parent.register_module(module_name, null);
            parent.register_module(module_name, value);
        }

        var warn_qlora = false;

        foreach (var (name, m) in module.named_modules().ToList())
        {
            if (m is AwqLinear awqLinear)
            {
                if (lora_weights.TryGetValue(name + ".lora_A", out var lora_A) &&
                    lora_weights.TryGetValue(name + ".lora_B", out var lora_B))
                {
                    if (!warn_qlora)
                    {
                        Console.WriteLine("Using QLoraLinear for Lora inference. It might be slow.");
                        warn_qlora = true;
                    }
                    // lora_A: (r, in), lora_B: (out, r) => (out, in)
                    var r = lora_A.shape[0];
                    var newLinear = new QLoraLinear(awqLinear, r, lora_alpha);
                    newLinear.lora_A.copy_(lora_A);
                    newLinear.lora_B.copy_(lora_B);
                    awqLinear = newLinear;
                    set_on_parent(name, awqLinear);

                    lora_weights.Remove(name + ".lora_A");
                    lora_weights.Remove(name + ".lora_B");
                }
                if (lora_weights.TryGetValue(name + ".bias", out var bias) && awqLinear.bias is not null)
                {
                    awqLinear.bias.copy_(bias);
                    lora_weights.Remove(name + ".bias");
                }
            }
            else if (m is CustomLinear linear)
            {
                if (lora_weights.TryGetValue(name + ".lora_A", out var lora_A) &&
                    lora_weights.TryGetValue(name + ".lora_B", out var lora_B))
                {
                    using var scope = torch.NewDisposeScope();
                    // lora_A: (r, in), lora_B: (out, r) => (out, in)
                    var r = lora_A.shape[0];
                    var lora_scaling = lora_alpha / r;
                    lora_A = lora_A.to(linear.weight.device);
                    lora_B = lora_B.to(linear.weight.device);
                    linear.weight.add_(lora_B.matmul(lora_A) * lora_scaling);
                    lora_weights.Remove(name + ".lora_A");
                    lora_weights.Remove(name + ".lora_B");
                }
                if (lora_weights.TryGetValue(name + ".bias", out var bias) && linear.bias is not null)
                {
                    linear.bias.copy_(bias);
                    lora_weights.Remove(name + ".bias");
                }
            }
            else if (m is CustomEmbedding embedding)
            {
                if (lora_weights.TryGetValue(name + ".lora_embedding_A", out var lora_A) &&
                    lora_weights.TryGetValue(name + ".lora_embedding_B", out var lora_B))
                {
                    using var scope = torch.NewDisposeScope();
                    // lora_A: (r, num_embeddings), lora_B: (embed_dim, r) => (num_embeddings, embed_dim)
                    var r = lora_A.shape[0];
                    var lora_scaling = lora_alpha / r;
                    lora_A = lora_A.to(embedding.weight.device);
                    lora_B = lora_B.to(embedding.weight.device);
                    embedding.weight.add_(lora_B.matmul(lora_A).t() * lora_scaling);
                    lora_weights.Remove(name + ".lora_embedding_A");
                    lora_weights.Remove(name + ".lora_embedding_B");
                }
            }
        }

        if (lora_weights.Count > 0)
            Console.WriteLine("Unused lora weights: " + string.Join(", ", lora_weights.Keys));
    }
}

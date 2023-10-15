using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;

public static class Activations
{
    public static nn.Module<Tensor, Tensor> get_activation_by_name(string name) {
        switch (name) {
            case "relu":
                return nn.ReLU();

            case "gelu":
                return nn.GELU();

            case "silu":
            case "swish":
                return nn.SiLU();

            case "sigmoid":
                return nn.Sigmoid();

            case "tanh":
                return nn.Tanh();

            default:
                throw new ArgumentException($"Unknown activation function {name}");
        }
    }
}

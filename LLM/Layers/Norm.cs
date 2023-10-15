using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;

public class RMSNorm : nn.Module<Tensor, Tensor>
{
    public Parameter weight;
    public double eps;

    public RMSNorm(
        long[] normalized_shape,
        double eps = 1e-5,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("RMSNorm")
    {
        weight = nn.Parameter(torch.ones(normalized_shape, dtype: dtype, device: device));
        this.eps = eps;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var h = x.to(torch.float32);
        var norm = h * torch.rsqrt(h.pow(2).mean(new[] { -1L }, keepdim: true) + (float)eps);
        return scope.MoveToOuter(norm.type_as(x) * weight);
    }
}

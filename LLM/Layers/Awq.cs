using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;

public class ScaledActivation : nn.Module<Tensor, Tensor>
{
    public Parameter scales;
    public nn.Module<Tensor, Tensor> act_fn;

    public ScaledActivation(Tensor scales, string act_fn_name) : base("ScaledActivation")
    {
        this.scales = new Parameter(scales);
        act_fn = Activations.get_activation_by_name(act_fn_name);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        return scope.MoveToOuter(act_fn.call(x) / scales);
    }
}

public class AwqLinear : nn.Module<Tensor, Tensor>
{
    public Parameter qweight;
    public Parameter qzeros;
    public Parameter scales;
    public Parameter? bias;

    public const int group_size = 128;
    public const int pack_size = 8;

    public AwqLinear(
        long inputSize,
        long outputSize,
        bool hasBias = true,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("AwqLinearGEMM")
    {
        if (inputSize % group_size != 0)
            throw new ArgumentException("inputSize must be a multiple of 128", nameof(inputSize));
        if (outputSize % group_size != 0)
            throw new ArgumentException("outputSize must be a multiple of 128", nameof(outputSize));

        dtype ??= torch.float16;
        if (dtype != torch.float16)
            throw new ArgumentException("dtype must be torch.float16", nameof(dtype));

        qweight = new Parameter(torch.empty(inputSize, outputSize / pack_size, torch.int32, device), false);
        qzeros = new Parameter(torch.zeros(inputSize / group_size, outputSize / pack_size, torch.int32, device), false);
        scales = new Parameter(torch.ones(inputSize / group_size, outputSize, dtype, device), false);
        if (hasBias)
            bias = new Parameter(torch.empty(outputSize, dtype, device));
        // skips init
        RegisterComponents();
    }

    public AwqLinear(
        Parameter qweight,
        Parameter qzeros,
        Parameter scales,
        Parameter? bias = null
    ) : base("AwqLinearGEMM")
    {
        this.qweight = qweight;
        this.qzeros = qzeros;
        this.scales = scales;
        this.bias = bias;
        RegisterComponents();
    }

    protected bool is_converted_turbomind = false;
    public void convert_turbomind()
    {
        var (qw, sz) = Ops.turbomind_convert_s4_k_m8(qweight, scales, qzeros, group_size);

        qweight.Dispose();
        scales.Dispose();
        qzeros.Dispose();

        qweight = new Parameter(qw, false);
        scales = new Parameter(sz, false);
        qzeros = new Parameter(torch.tensor(0), false);

        is_converted_turbomind = true;
    }

    public override Tensor forward(Tensor input)
    {
        using var scope = torch.NewDisposeScope();

        Tensor output;

        if (!is_converted_turbomind)
            output = Ops.awq_gemm_forward(input, qweight, scales, qzeros);
        else
            output = Ops.turbomind_gemm_s4_f16(input, qweight, scales, group_size);

        if (bias is not null)
            output += bias;

        return scope.MoveToOuter(output);
    }
}

using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using TensorIndex = torch.TensorIndex;
using F = torch.nn.functional;

public class CustomLinear : nn.Module<Tensor, Tensor>
{
    public Parameter weight;
    public Parameter? bias;
    public CustomLinear(
        long inputSize,
        long outputSize,
        bool hasBias = true,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("CustomLinear")
    {
        weight = new Parameter(torch.empty(outputSize, inputSize, dtype, device));
        if (hasBias)
            bias = new Parameter(torch.empty(outputSize, dtype, device));
        // skips init
        RegisterComponents();
    }
    public CustomLinear(
        Parameter weight,
        Parameter? bias = null
    ) : base("CustomLinear")
    {
        this.weight = weight;
        this.bias = bias;
        RegisterComponents();
    }
    public override Tensor forward(Tensor x)
    {
        return F.linear(x, weight, bias);
    }
}

public class CustomEmbedding : nn.Module<Tensor, Tensor>
{
    public Parameter weight;
    public CustomEmbedding(
        long num_embeddings,
        long embedding_dim,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("CustomEmbedding")
    {
        weight = new Parameter(torch.empty(num_embeddings, embedding_dim, dtype, device));
        // skips init
        RegisterComponents();
    }
    public CustomEmbedding(
        Parameter weight
    ) : base("CustomEmbedding")
    {
        this.weight = weight;
        RegisterComponents();
    }
    public override Tensor forward(Tensor x)
    {
        return weight[x];
    }
}

public abstract class AbstractBuilder
{
    public virtual object config { get; set; } = new();
    public virtual torch.Device device { get; set; } = torch.CUDA;
    public virtual torch.ScalarType dtype { get; set; } = torch.float32;

    public virtual nn.Module<Tensor, Tensor> create_embedding(long vocab_size, long hidden_size)
        => new CustomEmbedding(vocab_size, hidden_size, dtype: dtype, device: device);

    public virtual nn.Module<Tensor, Tensor> create_linear(long input_size, long output_size, bool hasBias = true)
        => new CustomLinear(input_size, output_size, hasBias: hasBias, dtype: dtype, device: device);
}

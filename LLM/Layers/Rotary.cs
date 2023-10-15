using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;

public class RotaryEmbedding : nn.Module<Tensor, (Tensor cos, Tensor sin)>
{
    public static (Tensor cos, Tensor sin) precompute_freqs_cis(long dim, long length, double theta = 10000.0)
    {
        using var scope = torch.NewDisposeScope();
        if (dim % 2 != 0)
            throw new Exception("dim should be multiple of 2");
        var freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32) / dim);
        freqs = torch.outer(torch.arange(length).to(torch.float32), freqs);
        freqs = torch.cat(new[] { freqs, freqs }, dim: -1);
        return scope.MoveToOuter(torch.cos(freqs), torch.sin(freqs));
    }

    public static Tensor apply_rotary_emb(Tensor x, (Tensor cos, Tensor sin) freqs_cis)
    {
        using var scope = torch.NewDisposeScope();
        var half_dim = x.shape.Last() / 2;
        var (x_r, x_i) = torch.chunk(x, 2, dim: -1);
        var rotated = torch.cat(new[] { -x_i, x_r }, dim: -1);
        return scope.MoveToOuter(x * freqs_cis.cos + rotated * freqs_cis.sin);
    }

    public Tensor cos;
    public Tensor sin;

    public RotaryEmbedding(
        long num_embeddings,
        long embedding_dims,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("RotaryEmbedding")
    {
        (cos, sin) = precompute_freqs_cis(embedding_dims, num_embeddings);
        if (dtype is not null)
            (cos, sin) = (cos.to(dtype.Value), sin.to(dtype.Value));
        if (device is not null)
            (cos, sin) = (cos.to(device), sin.to(device));

        RegisterComponents();
    }

    public override (Tensor cos, Tensor sin) forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var (n_batch, n_seq) = x.shape;
        return scope.MoveToOuter(
            cos[x].view(n_batch, n_seq, 1, -1),
            sin[x].view(n_batch, n_seq, 1, -1)
        );
    }

    public override Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor>? destination = null, string? prefix = null)
    {
        // omit from state_dict
        return destination ?? new();
    }
}

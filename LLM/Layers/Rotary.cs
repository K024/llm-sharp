using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;

public interface IRotary : IDisposable
{
    public (Tensor, Tensor) apply(Tensor query, Tensor key);

    public IEnumerable<Tensor> weights { get; }
}

public class RotaryEmbedding : nn.Module<Tensor, IRotary>
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

    public static (Tensor query, Tensor key) apply_rotary_emb_fused(Tensor query, Tensor key, (Tensor cos, Tensor sin) freqs_cis)
        => Ops.awq_rotary_embedding_neox(query, key, freqs_cis.cos, freqs_cis.sin);

    public class SlowRotary : IRotary
    {
        public Tensor cos;
        public Tensor sin;

        public IEnumerable<Tensor> weights => new[] { cos, sin };

        public SlowRotary(Tensor cos, Tensor sin)
        {
            this.cos = cos;
            this.sin = sin;
        }

        public virtual (Tensor, Tensor) apply(Tensor query, Tensor key)
        {
            return (
                apply_rotary_emb(query, (cos, sin)),
                apply_rotary_emb(key, (cos, sin))
            );
        }

        public void Dispose()
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    public class FastRotary : SlowRotary
    {
        public FastRotary(Tensor cos, Tensor sin) : base(cos, sin)
        {
        }

        public override (Tensor, Tensor) apply(Tensor query, Tensor key)
            => apply_rotary_emb_fused(query, key, (cos, sin));
    }

    public Tensor cos;
    public Tensor sin;

    public virtual (Tensor cos, Tensor sin) precompute(long dim, long length, double theta = 10000.0)
        => precompute_freqs_cis(dim, length, theta);

    public RotaryEmbedding(
        long num_embeddings,
        long embedding_dims,
        double theta = 10000.0,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("RotaryEmbedding")
    {
        (cos, sin) = precompute(embedding_dims, num_embeddings, theta);
        if (dtype is not null)
            (cos, sin) = (cos.to(dtype.Value), sin.to(dtype.Value));
        if (device is not null)
            (cos, sin) = (cos.to(device), sin.to(device));

        RegisterComponents();
    }

    public override IRotary forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var (n_batch, n_seq) = x.shape;
        var rotary = new SlowRotary(
            cos[x].view(n_batch, n_seq, 1, -1),
            sin[x].view(n_batch, n_seq, 1, -1)
        );
        scope.MoveToOuter(rotary.weights);
        return rotary;
    }

    public override Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor>? destination = null, string? prefix = null)
    {
        // omit from state_dict
        // TODO: not working
        return destination ?? new();
    }
}

public class FastRotaryEmbedding : RotaryEmbedding
{
    public FastRotaryEmbedding(
        long num_embeddings,
        long embedding_dims,
        double theta = 10000.0,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base(num_embeddings, embedding_dims, theta, dtype, device)
    {
    }

    public override IRotary forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var (n_batch, n_seq) = x.shape;
        var rotary = new FastRotary(
            cos[x].view(n_batch, n_seq, 1, -1),
            sin[x].view(n_batch, n_seq, 1, -1)
        );
        scope.MoveToOuter(rotary.weights);
        return rotary;
    }
}

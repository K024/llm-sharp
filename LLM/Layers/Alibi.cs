using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;
using TensorIndex = torch.TensorIndex;

public class Alibi : nn.Module<long, long, Tensor>
{
    public static Tensor precompute_alibi_mask(long n_head, long max_len, bool full = false, float alibi_bias_max = 8)
    {
        using var scope = torch.NewDisposeScope();
        var _n_head = (long)Math.Pow(2, Math.Ceiling(Math.Log2(n_head)));

        var slopes = 1 / torch.pow(2, torch.arange(1, _n_head + 1) * (alibi_bias_max / _n_head));
        if (_n_head != n_head)
        {
            slopes = torch.concat(new[]{
                slopes[TensorIndex.Slice(1, null, 2)],
                slopes[TensorIndex.Slice(0, null, 2)]
            })[..(int)n_head];
        }
        slopes = slopes.view(n_head, 1, 1);

        var alibi_bias = torch.arange(1 - max_len, 1);
        if (full)
        {
            // full mask not used
            alibi_bias = alibi_bias.view(1, max_len) - alibi_bias.view(max_len, 1);
            alibi_bias = torch.minimum(alibi_bias, 0);
            alibi_bias.view(1, max_len, max_len);
        }
        else
        {
            alibi_bias.view(1, 1, max_len);
        }
        return scope.MoveToOuter(alibi_bias * slopes);
    }

    public Tensor mask;
    public long n_head;

    public virtual Tensor precompute(long num_heads, long num_embeddings, bool full = false, float alibi_bias_max = 8)
        => precompute_alibi_mask(num_heads, num_embeddings, full, alibi_bias_max);

    public Alibi(
        long num_heads,
        long num_embeddings,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("Alibi")
    {
        n_head = num_heads;
        mask = precompute(num_heads, num_embeddings);
        if (dtype is not null)
            mask = mask.to(dtype.Value);
        if (device is not null)
            mask = mask.to(device);

        RegisterComponents();
    }

    public override Tensor forward(long q_seq_len, long kv_seq_len)
    {
        return mask[
            TensorIndex.None,
            TensorIndex.Colon,
            TensorIndex.Colon,
            // TensorIndex.Slice(kv_seq_len - q_seq_len, kv_seq_len),
            TensorIndex.Slice(0, kv_seq_len)
        ];
    }

    public override Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor>? destination = null, string? prefix = null)
    {
        // omit from state_dict
        // TODO: not working
        return destination ?? new();
    }
}

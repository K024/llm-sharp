using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;

public interface IKvCache : IDisposable
{
    public long size { get; }
    public (Tensor, Tensor) update(Tensor k, Tensor v);
    public IEnumerable<Tensor> weights { get; }
}

public class KvCache : IKvCache
{
    public Tensor? k_cache;
    public Tensor? v_cache;

    public IEnumerable<Tensor> weights => new[] { k_cache, v_cache }.Where(x => x is not null)!;

    public long size => k_cache?.shape[1] ?? 0;

    public void Dispose()
    {
        k_cache?.Dispose();
        v_cache?.Dispose();
    }

    public (Tensor, Tensor) update(Tensor k, Tensor v)
    {
        if (k.requires_grad || v.requires_grad)
            throw new Exception("k and v should not require grad");

        using var scope = torch.NewDisposeScope();

        if (k_cache is not null)
        {
            k = torch.cat(new[] { k_cache, k }, dim: 1);
            k_cache.Dispose();
        }
        if (v_cache is not null)
        {
            v = torch.cat(new[] { v_cache, v }, dim: 1);
            v_cache.Dispose();
        }

        k_cache = k.detach();
        v_cache = v.detach();

        // the creater of kv_cache is responsible for disposing it
        scope.Detach(k_cache, v_cache);
        scope.MoveToOuter(k, v);

        return (k, v);
    }
}

public class FastKvCache : IKvCache

{
    protected const long DEFAULT_CAPABILITY = 200;

    public Tensor k_cache;
    public Tensor v_cache;

    public IEnumerable<Tensor> weights => new[] { k_cache, v_cache };

    protected long capability;
    public long size { get; protected set; } = 0;

    public FastKvCache(long batch_size, long n_head, long d_head, torch.Device device, torch.ScalarType dtype) : base()
    {
        k_cache = torch.zeros(batch_size, DEFAULT_CAPABILITY, n_head, d_head, dtype: dtype, device: device);
        v_cache = torch.zeros(batch_size, DEFAULT_CAPABILITY, n_head, d_head, dtype: dtype, device: device);
    }

    public void Dispose()
    {
        k_cache.Dispose();
        v_cache.Dispose();
    }

    protected void extend(long new_capability)
    {
        if (new_capability <= capability)
            return;

        using var scope = torch.NewDisposeScope();
        var extended_size = new_capability - capability;
        var extends = torch.empty(k_cache.shape[0], extended_size, k_cache.shape[2], k_cache.shape[3], dtype: k_cache.dtype, device: k_cache.device);

        var old_k_cache = k_cache;
        var old_v_cache = v_cache;

        k_cache = torch.cat(new[] { k_cache, extends }, dim: 1);
        v_cache = torch.cat(new[] { v_cache, extends }, dim: 1);
        old_k_cache.Dispose();
        old_v_cache.Dispose();

        scope.Detach(k_cache, v_cache);
        capability = new_capability;
    }

    public (Tensor, Tensor) update(Tensor k, Tensor v)
    {
        if (k.requires_grad || v.requires_grad)
            throw new Exception("k and v should not require grad");

        var updated_size = k.shape[1];
        if (size + updated_size > capability)
            extend(size + updated_size + DEFAULT_CAPABILITY);

        var index_range = torch.TensorIndex.Slice(size, size + updated_size);

        k_cache[.., index_range] = k;
        v_cache[.., index_range] = v;
        size += updated_size;

        var new_index_range = torch.TensorIndex.Slice(0, size);
        return (k_cache[.., new_index_range], v_cache[.., new_index_range]);
    }
}

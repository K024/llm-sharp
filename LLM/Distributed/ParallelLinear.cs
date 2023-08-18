using TorchSharp;
using TorchSharp.Modules;

namespace llm_sharp.LLM.Distributed;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;


public class ColumnParallelLinear : nn.Module<Tensor, Tensor>
{
    public World? world;
    public int rank;
    public bool gatherOutput;

    public Parameter weight;
    public Parameter? bias;

    public ColumnParallelLinear(
        long inputSize,
        long outputSize,
        bool hasBias = true,
        bool gatherOutput = false,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("ColumnParallelLinear")
    {
        world = World.CurrentWorld;
        rank = World.CurrentRank;
        this.gatherOutput = gatherOutput;
        if (world is null)
        {
            weight = new Parameter(torch.empty(outputSize, inputSize, dtype, device));
            if (hasBias)
                bias = new Parameter(torch.empty(outputSize, dtype, device));
        }
        else
        {
            if (outputSize % world.WorldSize != 0)
                throw new Exception($"Unable to split model into {world.WorldSize} parts");

            var targetOutputSize = outputSize / world.WorldSize;
            var localDevice = world.Devices[rank];

            weight = new Parameter(torch.empty(targetOutputSize, inputSize, dtype, localDevice));
            if (hasBias)
                bias = new Parameter(torch.empty(targetOutputSize, dtype, localDevice));
        }
        RegisterComponents();
    }

    public (Tensor weight, Tensor? bias) GatherWeights()
    {
        if (world is null)
            throw new Exception("Not in parallel");
        if (World.CurrentWorld != world)
            throw new Exception("Not in correct distributed context");

        using var scope = torch.NewDisposeScope();
        using var no_grad = torch.no_grad();

        (Tensor weight, Tensor? bias) result = default!;

        var tensors = World.Gather(weight);
        if (World.IsMainThread)
            result.weight = scope.Detach(torch.cat(tensors!, dim: 0));

        if (bias is not null)
        {
            tensors = World.Gather(bias);
            if (World.IsMainThread)
                result.bias = scope.Detach(torch.cat(tensors!, dim: 0));
        }
        return result;
    }

    public override Tensor forward(Tensor x)
    {
        if (world is null)
            return F.linear(x, weight, bias);

        if (world != World.CurrentWorld || rank != World.CurrentRank)
            throw new Exception("Not forwarding in same World constructed");

        using var scope = torch.NewDisposeScope();
        var result = F.linear(x, weight, bias);

        if (!gatherOutput)
            return scope.MoveToOuter(result);

        var gathered = World.AllGather(result);
        var output = torch.cat(gathered, dim: -1);

        return scope.MoveToOuter(output);
    }
}

public class RowParallelLinear : nn.Module<Tensor, Tensor>
{
    public World? world;
    public int rank;
    public bool inputIsParallel;

    public Parameter weight;
    public Parameter? bias;

    public RowParallelLinear(
        long inputSize,
        long outputSize,
        bool hasBias = true,
        bool inputIsParallel = false,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("RowParallelLinear")
    {
        world = World.CurrentWorld;
        rank = World.CurrentRank;
        this.inputIsParallel = inputIsParallel;
        if (world is null)
        {
            weight = new Parameter(torch.empty(outputSize, inputSize, dtype, device));
            if (hasBias)
                bias = new Parameter(torch.empty(outputSize, dtype, device));
        }
        else
        {
            if (inputSize % world.WorldSize != 0)
                throw new Exception($"Unable to split model into {world.WorldSize} parts");

            var targetInputSize = inputSize / world.WorldSize;
            var localDevice = world.Devices[rank];

            weight = new Parameter(torch.empty(outputSize, targetInputSize, dtype, localDevice));
            if (hasBias)
                bias = new Parameter(torch.empty(outputSize, dtype, localDevice));
        }
        RegisterComponents();
    }

    public (Tensor weight, Tensor? bias) GatherWeights()
    {
        if (world is null)
            throw new Exception("Not in parallel");
        if (World.CurrentWorld != world)
            throw new Exception("Not in correct distributed context");

        using var scope = torch.NewDisposeScope();
        using var no_grad = torch.no_grad();

        (Tensor weight, Tensor? bias) result = default!;

        var tensors = World.Gather(weight);
        if (World.IsMainThread)
            result.weight = scope.Detach(torch.cat(tensors!, dim: 1));

        if (bias is not null)
        {
            if (World.IsMainThread)
                result.bias = scope.Detach(bias.clone());
        }
        return result;
    }

    public override Tensor forward(Tensor x)
    {
        if (world is null)
            return F.linear(x, weight, bias);

        if (world != World.CurrentWorld || rank != World.CurrentRank)
            throw new Exception("Not forwarding in same World constructed");

        using var scope = torch.NewDisposeScope();

        if (!inputIsParallel)
        {
            var splits = torch.chunk(x, world.WorldSize, dim: -1);
            x = splits[rank];
        }

        var result = F.linear(x, weight);

        var gathered = World.AllGather(result);
        var output = torch.stack(gathered, dim: 0).sum(dim: 0, keepdim: false);

        if (bias is not null)
            output = output.add(bias);

        return scope.MoveToOuter(output);
    }
}

public class ParallelEmbedding : nn.Module<Tensor, Tensor>
{
    public World? world;
    public int rank;

    public Parameter weight;

    public ParallelEmbedding(
        long num_embeddings,
        long embedding_dim,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("ParallelEmbedding")
    {
        world = World.CurrentWorld;
        rank = World.CurrentRank;
        if (world is null)
        {
            weight = new Parameter(torch.empty(num_embeddings, embedding_dim, dtype, device));
        }
        else
        {
            if (embedding_dim % world.WorldSize != 0)
                throw new Exception($"Unable to split model into {world.WorldSize} parts");

            var targetEmbeddingDim = embedding_dim / world.WorldSize;
            var localDevice = world.Devices[rank];

            weight = new Parameter(torch.empty(num_embeddings, targetEmbeddingDim, dtype, localDevice));
        }
        RegisterComponents();
    }

    public Tensor GatherWeights()
    {
        if (world is null)
            throw new Exception("Not in parallel");
        if (World.CurrentWorld != world)
            throw new Exception("Not in correct distributed context");

        using var scope = torch.NewDisposeScope();
        using var no_grad = torch.no_grad();

        Tensor result = default!;

        var tensors = World.Gather(weight);
        if (World.IsMainThread)
            result = scope.Detach(torch.cat(tensors!, dim: 1));

        return result;
    }

    public override Tensor forward(Tensor x)
    {
        if (world is null)
            return weight[x];

        if (world != World.CurrentWorld || rank != World.CurrentRank)
            throw new Exception("Not forwarding in same World constructed");

        using var scope = torch.NewDisposeScope();

        var result = weight[x];

        var gathered = World.AllGather(result);
        var output = torch.cat(gathered, dim: -1);

        return scope.MoveToOuter(output);
    }
}

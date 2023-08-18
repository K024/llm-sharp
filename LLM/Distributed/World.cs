using System.Threading.Channels;
using llm_sharp.LLM.Utils;
using TorchSharp;

namespace llm_sharp.LLM.Distributed;

using Tensor = torch.Tensor;

public class World
{
    // thread locals
    protected static ThreadLocal<World?> _currentWorld = new();
    protected static ThreadLocal<int> _currentRank = new(() => -1);

    public static World CurrentWorld => _currentWorld.Value
        ?? throw new Exception("Not in a distributed environment.");
    public static int CurrentRank => _currentRank.Value;
    public static bool IsMainThread => CurrentRank == 0;
    public static bool IsInDistributedEnv => CurrentRank >= 0;


    // world definition
    protected torch.Device[] _devices;
    public IReadOnlyList<torch.Device> Devices => _devices;
    public torch.Device MainDecive => _devices[0];
    public int WorldSize => _devices.Length;
    public bool IsDistributed => WorldSize > 1;

    // Each thread only read its own channel
    protected Channel<(Action action, Tensor tensor)>[] _channels;
    protected Channel<int>[] _signalChannels;


    protected World(torch.Device[] devices)
    {
        if (devices.Length == 0)
            throw new ArgumentException(nameof(devices));
        if (devices.Length > 0 && !devices.All(d => d.type == DeviceType.CUDA))
            throw new ArgumentException("Require CUDA devices", nameof(devices));

        _devices = devices;
        _channels = Enumerable.Range(0, WorldSize)
            .Select(x => Channel.CreateUnbounded<(Action action, Tensor tensor)>())
            .ToArray();
        _signalChannels = Enumerable.Range(0, WorldSize)
            .Select(x => Channel.CreateUnbounded<int>())
            .ToArray();
    }

    public static World FromDevices(params int[] devices)
    {
        return new World(devices.Select(x => new torch.Device(x)).ToArray());
    }

    public static World FromDevices(params torch.Device[] devices)
    {
        return new World(devices);
    }

    public T[] ParallelFor<T>(Func<T> func)
    {
        var world = this;
        var results = new T[WorldSize];
        Parallel.ForEach(_devices, (device, _, rank) =>
        {
            try
            {
                _currentWorld.Value = world;
                _currentRank.Value = (int)rank;
                results[rank] = func();
            }
            finally
            {
                _currentWorld.Value = null;
                _currentRank.Value = -1;
            }
        });

        return results;
    }

    protected void SendTensor(int toRank, Action action, Tensor tensor)
    {
        Asserts(tensor is not null);
        var writer = _channels[toRank].Writer;
        SpinWait.SpinUntil(() => writer.TryWrite((action, tensor!)));
    }

    protected Tensor ReceiveTensor(Action action)
    {
        (Action action, Tensor tensor) result = default;
        var reader = _channels[CurrentRank].Reader;
        SpinWait.SpinUntil(() => reader.TryRead(out result));
        Asserts(result.action == action);
        return result.tensor!;
    }

    protected void SendSignal(int toRank, int signal)
    {
        var writer = _signalChannels[toRank].Writer;
        SpinWait.SpinUntil(() => writer.TryWrite(signal));
    }

    protected int ReceiveSignal()
    {
        int result = default;
        var reader = _signalChannels[CurrentRank].Reader;
        SpinWait.SpinUntil(() => reader.TryRead(out result));
        return result;
    }

    protected static void Asserts(bool condition, string? message = null)
    {
        if (!condition)
            throw new Exception(message ?? "Assertion failed");
    }

    /// <summary>
    /// Sync all threads
    /// </summary>
    public static void WaitForAll()
    {
        var world = CurrentWorld;

        if (IsMainThread)
        {
            for (var toRank = 1; toRank < world.WorldSize; toRank++)
                Asserts(world.ReceiveSignal() == -1);

            for (var toRank = 1; toRank < world.WorldSize; toRank++)
                world.SendSignal(toRank, -1);
        }
        else
        {
            world.SendSignal(0, -1);
            Asserts(world.ReceiveSignal() == -1);
        }
    }

    protected enum Action
    {
        Broadcast = 2,
        Scatter = 3,
        Gather = 5,
        AllGather = 11,
    }

    /// <summary>
    /// Broadcast a tensor from main to others.
    /// </summary>
    /// <param name="current">Should be only defined if in main thread</param>
    /// <returns>Local tensor</returns>
    public static Tensor Broadcast(Tensor? current)
    {
        using var no_grad = torch.no_grad();
        using var scope = torch.NewDisposeScope();

        var rank = CurrentRank;
        var world = CurrentWorld;

        var device = world.Devices[rank];

        if (IsMainThread)
        {
            Asserts(current is not null);
            Asserts(current!.device.IsSame(device));

            // move to device in the target thread
            for (var toRank = 1; toRank < world.WorldSize; toRank++)
                world.SendTensor(toRank, Action.Broadcast, current);

            WaitForAll();
            return current;
        }
        else
        {
            Asserts(current is null);

            var tensor = world.ReceiveTensor(Action.Broadcast);

            // currently no grad support
            var result = torch.empty_like(tensor, device: device);
            result.copy_(tensor);

            WaitForAll();
            return scope.MoveToOuter(result);
        }
    }

    /// <summary>
    /// Scatter tensors from main to others.
    /// </summary>
    /// <param name="local">Should be only defined if in main thread</param>
    /// <returns>Local tensor</returns>
    public static Tensor Scatter(Tensor[]? local)
    {
        using var no_grad = torch.no_grad();
        using var scope = torch.NewDisposeScope();

        var rank = CurrentRank;
        var world = CurrentWorld;

        var device = world.Devices[rank];

        if (IsMainThread)
        {
            Asserts(local is not null);
            Asserts(local!.Length == world.WorldSize);
            Asserts(local!.All(x => x.device.IsSame(device)));

            // move to device in the target thread
            for (var toRank = 1; toRank < world.WorldSize; toRank++)
                world.SendTensor(toRank, Action.Scatter, local[toRank]);

            WaitForAll();
            return local[rank];
        }
        else
        {
            Asserts(local is null);

            var tensor = world.ReceiveTensor(Action.Scatter);

            // currently no grad support
            var result = torch.empty_like(tensor, device: device);
            result.copy_(tensor);

            WaitForAll();
            return scope.MoveToOuter(result);
        }
    }

    /// <summary>
    /// Gather tensors from others to main thread
    /// </summary>
    /// <param name="local">The tensor to gather</param>
    /// <returns>Only valid in main thread</returns>
    public static Tensor[]? Gather(Tensor local)
    {
        using var no_grad = torch.no_grad();
        using var scope = torch.NewDisposeScope();

        var rank = CurrentRank;
        var world = CurrentWorld;

        var device = world.Devices[rank];
        Asserts(local is not null);
        Asserts(local!.device.IsSame(device));

        if (IsMainThread)
        {
            var results = new Tensor[world.WorldSize];
            results[rank] = local;

            for (var toRank = 1; toRank < world.WorldSize; toRank++)
            {
                // currently no grad support
                var result = torch.empty_like(local);
                results[toRank] = result;

                // copy to main device in the from thread
                world.SendTensor(toRank, Action.Gather, result);
            }

            WaitForAll();
            scope.MoveToOuter(results);
            return results;
        }
        else
        {
            // copy to main device
            var tensor = world.ReceiveTensor(Action.Gather);
            tensor.copy_(local);

            WaitForAll();
            return null;
        }
    }

    /// <summary>
    /// Gather tensors from all threads to all threads
    /// </summary>
    /// <param name="local">The tensor to gather</param>
    /// <returns>Gathered tensors</returns>
    public static Tensor[] AllGather(Tensor local)
    {
        using var no_grad = torch.no_grad();
        using var scope = torch.NewDisposeScope();

        var rank = CurrentRank;
        var world = CurrentWorld;

        var device = world.Devices[rank];
        Asserts(local is not null);
        Asserts(local!.device.IsSame(device));

        var results = new Tensor[world.WorldSize];
        results[rank] = local;

        for (var step = 1; step < world.WorldSize; step++)
        {
            var fromRank = (rank + step) % world.WorldSize;
            var fromTensor = torch.empty_like(local);
            results[fromRank] = fromTensor;

            // request tensor data from all other devices
            world.SendTensor(fromRank, Action.AllGather, fromTensor);
        }

        for (var step = 1; step < world.WorldSize; step++)
        {
            // copy to target device
            var tensor = world.ReceiveTensor(Action.AllGather);
            tensor.copy_(local);
        }

        WaitForAll();
        scope.MoveToOuter(results);
        return results;
    }
}

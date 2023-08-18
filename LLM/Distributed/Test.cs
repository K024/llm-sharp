using System.Diagnostics;
using llm_sharp.LLM.Utils;
using TorchSharp;

namespace llm_sharp.LLM.Distributed;

public static class DistributedTest
{
    public static void Test()
    {
        LibTorchLoader.EnsureLoaded();

        var world = World.FromDevices(0, 1, 2, 3);

        var models = world.ParallelFor(() =>
        {
            using var no_grad = torch.no_grad();

            var linear = new ColumnParallelLinear(2048, 2048, gatherOutput: false);
            var std = 1 / Math.Sqrt(linear.weight.shape[1]);
            linear.weight.uniform_(-std, std);

            return linear;
        });

        Console.WriteLine(models[0].weight);
        Console.WriteLine(models[1].weight);

        var input = torch.randn(1000, 2048).to(world.MainDecive);

        var parallel_input = world.ParallelFor(() =>
        {
            var parallel_input = World.IsMainThread
                ? input
                : null;

            var local = World.Broadcast(parallel_input);
            return local;
        });

        Stopwatch watch;
        for (var round = 0; round < 10; round++)
        {
            watch = Stopwatch.StartNew();
            for (var i = 0; i < 20; i++)
            {
                var outputs = world.ParallelFor(() =>
                {
                    using var scope = torch.NewDisposeScope();
                    var local = parallel_input[World.CurrentRank];
                    var model = models[World.CurrentRank];
                    var local_result = model.call(local);
                    return scope.MoveToOuter(local_result);
                });
                outputs[0][0, 0].item<float>();
            }
            watch.Stop();
            Console.WriteLine(watch.Elapsed);
        }


        var (weight, bias) = world.ParallelFor(() =>
        {
            var model = models[World.CurrentRank];
            return model.GatherWeights();
        })[0];

        watch = Stopwatch.StartNew();
        for (var i = 0; i < 20; i++)
        {
            using var scope = torch.NewDisposeScope();
            var result = torch.nn.functional.linear(input, weight, bias);
            result[0, 0].item<float>();
        }
        watch.Stop();
        Console.WriteLine("Ref");
        Console.WriteLine(watch.Elapsed);
    }
}

using TorchSharp;
using llm_sharp.LLM.Pretrained;

namespace llm_sharp.LLM.Utils;

public static class CliExtensions
{
    public static void run_torch_test()
    {
        Console.WriteLine($"TorchSharp version: {torch.__version__}");
        Console.WriteLine($"Required libtorch: {LibTorchDownloader.humanVersion}");
        LibTorchLoader.EnsureLoaded();
        Console.WriteLine($"Loaded libtorch from '{LibTorchLoader.LoadedPath}'");
        Console.WriteLine($"Cuda is available: {torch.cuda_is_available()}");
        torch.ones(3, 4).matmul(torch.ones(4, 5));
        Console.WriteLine($"Test passed");
        if (torch.cuda_is_available())
        {
            torch.ones(3, 4).cuda().matmul(torch.ones(4, 5).cuda());
            Console.WriteLine($"Cuda test passed");
            try
            {
                NativeOps.Ops.hello(torch.tensor(0f));
                Console.WriteLine($"Native ops test passed");
            }
            catch (Exception)
            {
                Console.WriteLine($"Native ops test failed");
            }
        }
        else
        {
            Console.WriteLine($"Cuda test & native ops skipped");
        }
    }

    public static void start_chat_cli(this LanguageModel llm)
    {
        var history = new List<ChatMessage>();
        var system_prompt = "";
        var print_perf = false;
        var generation_config = new GenerationConfig();

        Console.WriteLine("Use .help to get help");

        while (true)
        {
            Console.Write("user > ");

            var query = (Console.ReadLine() ?? "").Trim();

            if (string.IsNullOrEmpty(query))
                break;

            if (query == ".help")
            {
                Console.WriteLine(
                  "Commands:\n" +
                  "  .exit: exit chat\n" +
                  "  .clear: clear history\n" +
                  "  .undo: remove last history\n" +
                  "  .system <system_prompt>: set system prompt\n" +
                  "  .top_p <number>: set top_p value\n" +
                  "  .temperature <number>: set temperature value\n" +
                  "  .perf: toggle performance logger\n"
                );
                continue;
            }
            if (query == ".exit")
            {
                break;
            }
            if (query == ".clear")
            {
                history.Clear();
                continue;
            }
            if (query == ".undo")
            {
                if (history.Count > 0)
                    history.RemoveAt(history.Count - 1);
                continue;
            }
            if (query == ".perf")
            {
                print_perf = !print_perf;
                continue;
            }
            if (query.StartsWith(".system"))
            {
                system_prompt = query.Split(' ').Last();
                continue;
            }
            if (query.StartsWith(".top_p"))
            {
                var value = query.Split(' ').Last();
                if (float.TryParse(value, out var top_p))
                    generation_config.top_p = top_p;
                else
                    Console.WriteLine($"Invalid top_p value: {value}");
                continue;
            }
            if (query.StartsWith(".temperature"))
            {
                var value = query.Split(' ').Last();
                if (float.TryParse(value, out var temperature))
                    generation_config.temperature = temperature;
                else
                    Console.WriteLine($"Invalid temperature value: {value}");
                continue;
            }

            var answer = "";

            Console.Write("assistant > ");

            if (history.Count <= 0 && !string.IsNullOrEmpty(system_prompt))
                history.Add(new() { role = "system", content = system_prompt });

            history.Add(new() { role = "user", content = query });
            foreach (var output in llm.chat(history, generation_config))
            {
                answer += output.content;
                Console.Write(output.content);
            }
            Console.WriteLine();
            history.Add(new() { role = "assistant", content = answer });

            if (print_perf)
            {
                llm.print_perf();
                Console.WriteLine();
            }
        }
    }
}

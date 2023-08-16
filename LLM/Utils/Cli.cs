namespace llm_sharp.LLM.Utils;

public static class CliExtensions
{
    public static void start_chat_cli(this LLM llm)
    {
        var history = new List<(string query, string answer)>();
        var print_perf = false;

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

            var answer = "";

            Console.Write("assistant > ");
            foreach (var output in llm.chat(history, query))
            {
                answer += output;
                Console.Write(output);
            }
            Console.WriteLine();

            history.Add((query, answer));

            if (print_perf)
            {
                llm.print_perf();
                Console.WriteLine();
            }
        }
    }
}

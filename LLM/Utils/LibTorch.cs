using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace llm_sharp.LLM.Utils;

public static class LibTorchLoader
{
    private static object load_lock = new object();
    private static volatile bool loaded = false;
    private static string loaded_path = "";

    public static string LoadedPath => loaded_path;

    public static void EnsureLoaded(params string[] args)
    {
        if (loaded) return;

        lock (load_lock)
        {
            if (loaded) return;

            var libtorch =
                FindInArgs(args)
                    ?? FindInEnv()
                    ?? FindInPythonSitePackages()
                    ?? OsDefault();

            try
            {
                NativeLibrary.Load(libtorch);
                loaded = true;
                loaded_path = libtorch;
            }
            catch (DllNotFoundException)
            {
                Console.Error.WriteLine(
                    "Unable to load libtorch.\n" + 
                    "Try:\n" +
                    "  Option 1: run in a python environment with torch installed (by pip or conda)\n" +
                    "  Option 2: specify by command line args (--libtorch-path)\n" +
                    "  Option 3: specify by environment variables (LIBTORCH_PATH)\n" +
                    "  Option 4: add libtorch to PATH (windows) or LD_LIBRARY_PATH (unix/linux)\n"
                );
                throw;
            }
        }
    }

    public static string? RunCommand(string command)
    {
        try
        {
            var shell = "";
            var args = new List<string>();

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                shell = "cmd.exe";
                args.Add("/c");
                args.Add(command);
            }
            else
            {
                shell = "bash";
                args.Add("-c");
                args.Add(command);
            }

            var startInfo = new ProcessStartInfo()
            {
                FileName = shell,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };
            args.ForEach(x => startInfo.ArgumentList.Add(x));

            var process = Process.Start(startInfo) ?? throw new Exception();

            process.WaitForExit(3000);
            if (!process.HasExited || process.ExitCode != 0)
                throw new Exception();

            return process.StandardOutput.ReadToEnd();
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static string? FindInArgs(params string[] args)
    {
        var list = args.ToList();
        var index = list.FindIndex(x => x.StartsWith("--libtorch-path"));
        if (index >= 0)
        {
            if (list[index].StartsWith("--libtorch-path="))
            {
                var splits = list[index].Split('=', 1);
                return splits[1];
            }
            else if (list[index] == "--libtorch-path" && index + 1 < list.Count)
            {
                return list[index + 1];
            }
        }
        return null;
    }

    private static string? FindInEnv()
    {
        return Environment.GetEnvironmentVariable("LIBTORCH_PATH");
    }

    private static string? FindInPythonSitePackages()
    {
        var sysPath = RunCommand("python3 -c \"import sys, json; print(json.dumps(sys.path))\"");
        if (sysPath is null)
            return null;

        var paths = JsonSerializer.Deserialize<List<string>>(sysPath)
            ?? throw new Exception("Unable to parse python output");

        foreach (var path in paths)
        {
            if (string.IsNullOrWhiteSpace(path))
                continue;

            var linuxLibPath = Path.Combine(path, "torch/lib/libtorch.so");
            var winLibPath = Path.Combine(path, "torch/lib/torch.dll");

            if (File.Exists(linuxLibPath))
                return linuxLibPath;

            if (File.Exists(winLibPath))
                return winLibPath;
        }
        return null;
    }

    private static string OsDefault()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return "torch.dll";

        return "libtorch.so";
    }
}

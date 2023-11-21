using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace llm_sharp.LLM.Utils;

public static class LibTorchDownloader
{
    public static string cuda => "cu121";
    public static string torch => "2.1.0";
    public static string arch => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win_amd64" : "linux_x86_64";

    public static string defaultUrl => $"https://download.pytorch.org/whl/{cuda}/torch-{torch}%2B{cuda}-cp310-cp310-{arch}.whl";
    public static string whlFilename => $"torch-{torch}-{cuda}-{arch}-whl.zip";

    public static string cacheDir => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "llm-sharp");

    public static string HumanSize(long size) => size switch
    {
        < 1024 => $"{size} B",
        < 1024 * 1024 => $"{size / 1024.0:F2} KB",
        _ => $"{size / 1024.0 / 1024.0:F2} MB",
    };

    public static async Task DownloadAndExtractLibTorch(bool skipVerification = false, string? optionalUrl = null)
    {
        var url = optionalUrl ?? defaultUrl;
        var filename = whlFilename;
        var cache = cacheDir;

        if (!Directory.Exists(cache))
            Directory.CreateDirectory(cache);

        var path = Path.Combine(cache, filename);

        if (!skipVerification)
        {
            Console.Write($"Download libtorch from {url} to {path}? (y/n) ");
            var answer = Console.ReadLine() ?? "";
            if (!answer.ToLower().StartsWith("y"))
                throw new Exception("Aborted");
        }

        var tmpfilename = Path.Combine(cache, filename + ".tmp");
        {
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(1);

            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            if (!response.IsSuccessStatusCode)
                throw new Exception($"Unable to download libtorch from {url}");

            var contentLength = response.Content.Headers.ContentLength ?? 0;
            var downloaded = 0L;

            using var contentStream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(tmpfilename, FileMode.Create, FileAccess.Write, FileShare.None);

            var buffer = new byte[1024 * 1024];
            int bytesRead;
            var lastPrint = DateTimeOffset.Now;
            while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
            {
                await fileStream.WriteAsync(buffer, 0, bytesRead);
                downloaded += bytesRead;
                if ((DateTimeOffset.Now - lastPrint).TotalMilliseconds > 20)
                {
                    Console.Write($"\rDownloading {HumanSize(downloaded)} of {HumanSize(contentLength)}        ");
                    lastPrint = DateTimeOffset.Now;
                }
            }
            Console.WriteLine();
        }

        File.Move(tmpfilename, path, overwrite: true);
        Console.WriteLine($"Extracting {path}");
        ZipFile.ExtractToDirectory(path, cache, overwriteFiles: true);
        File.Delete(path);

        Console.WriteLine($"LibTorch downloaded to {path}");
    }
}

public static class LibTorchLoader
{
    private static object load_lock = new object();
    private static volatile bool loaded = false;
    private static string loaded_path = "";

    public static string LoadedPath => loaded_path;

    public static void DownloadLibTorch(bool removeLast = false, bool skipVerification = false, string? optionalUrl = null)
    {
        if (!removeLast && FindInDownloadCache() is not null)
        {
            Console.WriteLine("LibTorch already downloaded");
            return;
        }
        if (removeLast)
        {
            var cache = LibTorchDownloader.cacheDir;
            if (!skipVerification)
            {
                Console.Write($"Deleting torch cache in {cache}? (y/n) ");
                var answer = Console.ReadLine() ?? "";
                if (!answer.ToLower().StartsWith("y"))
                    throw new Exception("Aborted");
            }
            Directory.Delete(cache, true);
        }
        LibTorchDownloader.DownloadAndExtractLibTorch(skipVerification, optionalUrl).Wait();

        if (FindInDownloadCache() is null)
            throw new Exception("Unable to find libtorch in download cache");
    }

    public static void EnsureLoaded()
    {
        if (loaded) return;

        lock (load_lock)
        {
            if (loaded) return;

            var libtorch =
                FindInEnv()
                ?? FindInDownloadCache()
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
                    "  Option 1: download libtorch with command:\n" +
                    "              llm-sharp --command download\n" +
                    "  Option 2: run in a python environment with torch installed (by pip or conda)\n" +
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
                shell = "powershell.exe";
                args.Add("-Command");
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

    private static string? FindInEnv()
    {
        return Environment.GetEnvironmentVariable("LIBTORCH_PATH");
    }

    private static string? FindInDownloadCache()
    {
        var path = LibTorchDownloader.cacheDir;
        var linuxLibPath = Path.Combine(path, "torch/lib/libtorch.so");
        var winLibPath = Path.Combine(path, "torch/lib/torch.dll");

        if (File.Exists(linuxLibPath))
            return linuxLibPath;

        if (File.Exists(winLibPath))
            return winLibPath;

        return null;
    }

    private static string? FindInPythonSitePackages()
    {
        var sysPath = RunCommand("python -c \"import sys, json; print(json.dumps(sys.path))\"");
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

using llm_sharp.LLM.Pretrained;
using llm_sharp.LLM.Utils;
using Microsoft.Extensions.Options;
using TorchSharp;

namespace llm_sharp.Services;

public static partial class Extensions
{
    public static IServiceCollection AddLLMService(this IServiceCollection services, IConfiguration config)
    {
        services.Configure<LLMService.LLMConfig>(config);
        services.AddSingleton<LLMService>();
        return services;
    }
}

public class LLMService
{
    public class LLMLoraConfig
    {
        public string weights_path { get; set; } = "";
        public float alpha { get; set; } = 1.0f;
    }
    public class LLMModelConfig
    {
        public string name { get; set; } = "";
        public List<string> aliases { get; set; } = new();
        public string type { get; set; } = "";
        public string path { get; set; } = "";
        public string dtype { get; set; } = "float16";
        public string device { get; set; } = "cuda";
        public LLMLoraConfig? lora { get; set; }
    }
    public class LLMConfig
    {
        public List<LLMModelConfig> models { get; set; } = new();

        public OptimizationConfig optimization { get; set; } = new();
    }

    private string default_model = "";
    private Dictionary<string, PretrainedModel> models = new();
    private Dictionary<string, string> modelByName = new();

    public IEnumerable<string> Models => modelByName.Keys;

    protected ILogger<LLMService> Logger;

    public LLMService(IOptionsMonitor<LLMConfig> optionsMonitor, ILoggerFactory loggerFactory)
    {
        LibTorchLoader.EnsureLoaded();
        Logger = loggerFactory.CreateLogger<LLMService>();
        // load models sync when first construct
        UpdateOptions(optionsMonitor.CurrentValue);
        optionsMonitor.OnChange(UpdateOptionsAsync);
    }

    private void UpdateOptionsAsync(LLMConfig config)
    {
        Task.Factory.StartNew(() => UpdateOptions(config), TaskCreationOptions.LongRunning);
    }
    private void UpdateOptions(LLMConfig config)
    {
        if (config is null)
            return;
        Logger.LogInformation("Updated LLMConfig");
        lock (models)
        {
            OptimizationConfig.current = config.optimization;
            if (!LibTorchLoader.NativeOpsLoaded)
                OptimizationConfig.set_no_native_ops();

            modelByName = new();
            default_model = config.models.Count > 0 ? config.models[0].name : "";

            var to_remove = models.Where(x => config.models.FindIndex(m => m.path == x.Key) < 0).ToList();
            if (to_remove.Count > 0)
            {
                foreach (var model in to_remove)
                {
                    Logger.LogInformation("Removing model {model}", model.Key);
                    models.Remove(model.Key);
                    model.Value.Dispose();
                    Logger.LogInformation("Removed model {model}", model.Key);
                }
                // explicit collect
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.WaitForFullGCComplete();
                GC.Collect();
                if (LibTorchLoader.NativeOpsLoaded && OptimizationConfig.current.auto_empty_cache)
                    NativeOps.Ops.cuda_empty_cache();
            }

            foreach (var model in config.models)
            {
                modelByName.Add(model.name, model.path);
                foreach (var alias in model.aliases)
                    modelByName.Add(alias, model.path);

                if (models.ContainsKey(model.path))
                    continue;

                Logger.LogInformation("Loading model {model}", model.path);
                var model_instance = PretrainedModel.from_pretrained(
                    model.type,
                    model.path,
                    Enum.Parse<torch.ScalarType>(model.dtype, true),
                    torch.device(model.device)
                );
                if (model.lora is not null)
                {
                    Logger.LogInformation("Applying lora weights from {path}", model.lora.weights_path);
                    model_instance.load_lora_weights(model.lora.weights_path, model.lora.alpha);
                }

                models.Add(model.path, model_instance);
                Logger.LogInformation("Loaded model {model}", model.path);
            }

            // explicit collect
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.WaitForFullGCComplete();
            GC.Collect();
            if (LibTorchLoader.NativeOpsLoaded && OptimizationConfig.current.auto_empty_cache)
                NativeOps.Ops.cuda_empty_cache();
        }
    }

    protected T? FindModel<T>(string? model) where T : PretrainedModel
    {
        if (string.IsNullOrWhiteSpace(model))
            model = default_model;

        if (modelByName.TryGetValue(model, out var path))
        {
            var lm = models.GetValueOrDefault(path);
            if (lm is T typed_lm)
                return typed_lm;
            if (lm is not null)
                Logger.LogDebug("Model {model} is not of type {type}", model, typeof(T).Name);
        }

        return null;
    }

    public GenerativeLM? FindGenerativeLM(string? model)
    {
        return FindModel<GenerativeLM>(model);
    }

    public MaskedLM? FindMaskedLM(string? model)
    {
        return FindModel<MaskedLM>(model);
    }
}

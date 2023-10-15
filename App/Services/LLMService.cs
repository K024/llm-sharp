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
    public class LLMModelConfig
    {
        public string name { get; set; } = "";
        public List<string> aliases { get; set; } = new();
        public string type { get; set; } = "";
        public string path { get; set; } = "";
        public bool use_bfloat16 { get; set; } = true;
        public bool use_cuda { get; set; } = true;
        public int device { get; set; } = 0;
    }
    public class LLMConfig
    {
        public string default_model { get; set; } = "";
        public List<LLMModelConfig> models { get; set; } = new();
    }

    private string default_model = "";
    private Dictionary<string, LanguageModel> models = new();
    private Dictionary<string, string> modelByName = new();

    public IEnumerable<string> Models => modelByName.Keys;

    public LLMService(IOptionsMonitor<LLMConfig> optionsMonitor)
    {
        LibTorchLoader.EnsureLoaded();
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
        lock (models)
        {
            modelByName = new();
            default_model = config.default_model;
            foreach (var model in config.models)
            {
                if (models.ContainsKey(model.path))
                    continue;

                var model_instance = LanguageModel.from_pretrained(
                    model.type,
                    model.path,
                    model.use_bfloat16 ? torch.bfloat16 : torch.float16,
                    model.use_cuda ? torch.device(model.device) : torch.CPU
                );

                models.Add(model.path, model_instance);

                modelByName.Add(model.name, model.path);
                foreach (var alias in model.aliases)
                    modelByName.Add(alias, model.path);
            }

            var to_remove = models.Where(x => config.models.FindIndex(m => m.path == x.Key) < 0).ToList();
            if (to_remove.Count > 0)
            {
                foreach (var model in to_remove)
                {
                    models.Remove(model.Key);
                    // explicit collect
                    GC.Collect();
                }
            }
        }
    }

    public LanguageModel? FindModel(string? model)
    {
        if (string.IsNullOrWhiteSpace(model))
            model = default_model;

        if (modelByName.TryGetValue(model, out var path))
            return models.GetValueOrDefault(path);

        return null;
    }

    public IAsyncEnumerable<string>? ChatAsync(
        string? model,
        List<(string query, string answer)> history,
        string input,
        GenerationConfig? config = null)
    {
        var llm = FindModel(model);

        if (llm is null)
            return null;

        return llm.chat_async(history, input, config);
    }
}

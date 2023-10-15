using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.FileSystemGlobbing;
using TorchSharp;
using TupleAsJsonArray;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Tokenizers;
using llm_sharp.LLM.Layers;

namespace llm_sharp.LLM.Pretrained;

using Tensor = torch.Tensor;
using ScalarType = torch.ScalarType;
using Device = torch.Device;

public abstract partial class LanguageModel
{
    public static JsonSerializerOptions TupleJsonSerializerOptions { get; } = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        Converters = { new TupleConverterFactory() },
        WriteIndented = true,
    };

    public const string model_config_file = "model_config.json";
    public const string tokenizer_config_file = "tokenizer_config.json";
    public const string model_weights_pattern = "*.safetensors";

    public static LanguageModel from_pretrained(
        string classHint,
        string path,
        ScalarType? dtype = null,
        Device? device = null,
        Assembly? assembly = null)
    {
        var types = (assembly ?? typeof(LanguageModel).Assembly).GetTypes();

        var type = types.Where(x =>
            x.Name == classHint
            && x.IsClass
            && !x.IsAbstract
            && x.IsSubclassOf(typeof(LanguageModel))
        )
            .FirstOrDefault()
            ?? throw new Exception($"Unable to find class '{classHint}'. Try give the correct Assembly");

        var type_from_pretrained = type.GetMethod(
            "from_pretrained",
            BindingFlags.Static | BindingFlags.Public,
            new[] { typeof(string), typeof(ScalarType?), typeof(Device) }
        ) ?? throw new Exception($"The LLM should have a public static from_pretrained(string, torch.ScalarType?, torch.Device?) method");

        var result = type_from_pretrained.Invoke(null, new object?[] { path, dtype, device })
            ?? throw new Exception($"Unable to create class {classHint} with from_pretrained");

        return (LanguageModel)result;
    }

    public static (TModel, TModelConfig) model_from_pretrained<TModel, TModelConfig, TBuilder>(
        string path,
        ScalarType? dtype = null,
        Device? device = null)
        where TModel : torch.nn.Module
        where TBuilder : AbstractBuilder
    {
        var createModel = typeof(TModel).GetConstructor(new[] { typeof(TBuilder) })
            ?? throw new Exception($"The Model should have a constructor(TBuilder, torch.ScalarType?, torch.Device?)");

        var createBuilder = typeof(TBuilder).GetConstructor(new[] { typeof(TModelConfig) })
            ?? throw new Exception($"The Model should have a constructor(TModelConfig)");

        var modelConfig = JsonSerializer.Deserialize<TModelConfig>(
            File.ReadAllBytes(Path.Combine(path, model_config_file)),
            TupleJsonSerializerOptions
        ) ?? throw new ArgumentException(nameof(TModelConfig));

        var builder = (TBuilder)createBuilder.Invoke(new object?[] { modelConfig })
            ?? throw new NullReferenceException();

        if (device is not null)
            builder.device = device;

        if (dtype is not null)
            builder.dtype = dtype.Value;

        var model = (TModel)createModel.Invoke(new object?[] { builder })
            ?? throw new NullReferenceException();

        var state_dict = (model as torch.nn.Module)!.state_dict();
        var weight_files = new Matcher().AddInclude(model_weights_pattern).GetResultsInFullPath(path);

        load_state_dict(state_dict, weight_files.ToArray());

        return (model, modelConfig);
    }

    public static (TTokenizer, TTokenizerConfig) tokenizer_from_pretrained<TTokenizer, TTokenizerConfig>(
        string path)
        where TTokenizer : ITokenizer
    {
        var createTokenizer = typeof(TTokenizer).GetConstructor(
            new[] { typeof(TTokenizerConfig) }
        ) ?? throw new Exception($"The Model should have a constructor(TTokenizerConfig)");

        var tokenizerConfig = JsonSerializer.Deserialize<TTokenizerConfig>(
            File.ReadAllBytes(Path.Combine(path, tokenizer_config_file)),
            TupleJsonSerializerOptions
        ) ?? throw new ArgumentException(nameof(TTokenizerConfig));

        var tokenizer = (TTokenizer)createTokenizer.Invoke(new object?[] { tokenizerConfig })
            ?? throw new NullReferenceException();

        return (tokenizer, tokenizerConfig);
    }

    private static void load_state_dict(Dictionary<string, Tensor> state_dict, params string[] paths)
    {
        var cloned = new Dictionary<string, Tensor>(state_dict);
        var unusedKeys = new List<string>();

        foreach (var (idx, path) in paths.Select((x, i) => (i + 1, x)))
        {
            Console.Write($"\rLoading {path} [{idx}/{paths.Length}] ");
            using var tensors = new Safetensors(path);
            var unused = tensors.load_to_state_dict(cloned);
            unusedKeys.AddRange(unused);
        }
        Console.WriteLine();

        if (unusedKeys.Count > 0)
            Console.WriteLine($"Unused weights: {string.Join(',', unusedKeys)}");

        if (cloned.Count > 0)
            Console.WriteLine($"Uninitialized states: {string.Join(',', cloned.Keys)}");
    }
}

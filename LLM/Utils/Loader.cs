using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.FileSystemGlobbing;
using TorchSharp;
using TupleAsJsonArray;

namespace llm_sharp.LLM.Utils;

using Tensor = torch.Tensor;
using ScalarType = torch.ScalarType;

public abstract partial class LLM
{
    public static JsonSerializerOptions TupleJsonSerializerOptions { get; } = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        Converters = { new TupleConverterFactory() },
        WriteIndented = true,
    };

    public static LLM from_pretrained(
        string classHint,
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null,
        Assembly? assembly = null)
    {
        var types = (assembly ?? typeof(LLM).Assembly).GetTypes();

        var type = types.Where(x =>
            x.Name == classHint
            && x.IsClass
            && !x.IsAbstract
            && x.IsSubclassOf(typeof(LLM))
        )
            .FirstOrDefault()
            ?? throw new Exception($"Unable to find class '{classHint}'. Try give the correct Assembly");

        var type_from_pretrained = type.GetMethod(
            "from_pretrained",
            BindingFlags.Static | BindingFlags.Public,
            new[] { typeof(string), typeof(torch.ScalarType?), typeof(torch.Device) }
        ) ?? throw new Exception($"The LLM should have a public static from_pretrained(string, torch.ScalarType?, torch.Device?) method");

        var result = type_from_pretrained.Invoke(null, new object?[] { path, dtype, device })
            ?? throw new Exception($"Unable to create class {classHint} with from_pretrained");

        return (LLM)result;
    }
}

public abstract partial class LLM<TModel, TModelConfig, TTokenizer, TTokenizerConfig> : LLM
    where TModel : class
    where TModelConfig : class
    where TTokenizer : class
    where TTokenizerConfig : class
{
    public const string load_config_file = "load_config.json";
    public const string model_config_file = "model_config.json";
    public const string tokenizer_config_file = "tokenizer_config.json";
    public const string model_weights_pattern = "*.safetensors";

    public static (TModel, TModelConfig) model_from_pretrained(
        string path,
        torch.ScalarType? dtype = null,
        torch.Device? device = null)
    {
        var createModel = typeof(TModel).GetConstructor(
            new[] { typeof(TModelConfig), typeof(torch.ScalarType?), typeof(torch.Device) }
        ) ?? throw new Exception($"The Model should have a constructor(TModelConfig, torch.ScalarType?, torch.Device?)");

        var modelConfig = JsonSerializer.Deserialize<TModelConfig>(
            File.ReadAllBytes(Path.Combine(path, model_config_file)),
            TupleJsonSerializerOptions
        ) ?? throw new ArgumentException(nameof(TModelConfig));

        var model = (TModel)createModel.Invoke(new object?[] { modelConfig, dtype, device })
            ?? throw new NullReferenceException();

        var state_dict = (model as torch.nn.Module)!.state_dict();
        var weight_files = new Matcher().AddInclude(model_weights_pattern).GetResultsInFullPath(path);

        var load_config = File.Exists(Path.Combine(path, load_config_file))
            ? JsonSerializer.Deserialize<StateDictConverter.ConvertRules>(
                File.ReadAllBytes(Path.Combine(path, load_config_file)),
                TupleJsonSerializerOptions)
            : null;

        if (load_config is null)
            load_state_dict(state_dict, weight_files.ToArray());
        else
            load_with_converter(state_dict, new(load_config), weight_files.ToArray());

        return (model, modelConfig);
    }

    public static (TTokenizer, TTokenizerConfig) tokenizer_from_pretrained(
        string path)
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

    private static void load_with_converter(Dictionary<string, Tensor> state_dict, StateDictConverter converter, params string[] paths)
    {
        var cloned = new Dictionary<string, Tensor>(state_dict);
        var unusedKeys = new List<string>();

        IEnumerable<(string, Tensor)> load_tensors()
        {
            foreach (var (idx, path) in paths.Select((x, i) => (i + 1, x)))
            {
                Console.Write($"\rLoading {path} [{idx}/{paths.Length}] ");
                using var tensors = new Safetensors(path);
                foreach (var (name, tensor) in tensors.load_tensors())
                {
                    yield return (name, tensor);
                }
            }
        }

        Console.Write($"Using load_config.json to convert weights");

        foreach (var pair in converter.Convert(load_tensors()))
        {
            using var no_grad = torch.no_grad();
            using var scope = torch.NewDisposeScope();

            var (key, data) = pair;
            scope.Include(data);

            if (state_dict.TryGetValue(key, out var tensor))
            {
                if (tensor.dtype is ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16)
                    data = data.type_as(tensor);
                tensor.copy_(data);
                cloned.Remove(key);
            }
            else
            {
                unusedKeys.Add(key);
            }
        }

        if (unusedKeys.Count > 0)
            Console.WriteLine($"Unused weights: {string.Join(',', unusedKeys)}");

        if (cloned.Count > 0)
            Console.WriteLine($"Uninitialized states: {string.Join(',', cloned.Keys)}");
    }


    public void save_pretrained(string basePath, int? shard_size = null)
    {
        string template(int shard, int total)
        {
            if (total <= 1)
                return Path.Join(basePath, "model_weights.safetensors");
            return Path.Join(basePath, $"model_weights_{shard}.safetensors");
        }
        save_pretrained(basePath, template, shard_size);
    }

    public void save_pretrained(string basePath, Func<int, int, string> shardTemplate, int? shard_size = null)
    {
        Directory.CreateDirectory(basePath);

        var state_dict = (model as torch.nn.Module)!.state_dict();
        var max_size = shard_size ?? int.MaxValue;

        var weight_shard = new Dictionary<string, int>();

        var current_shard = 0;
        var current_size = 0;
        foreach (var pair in state_dict)
        {
            var size = pair.Value.bytes.Length;
            if (size > max_size)
                throw new Exception($"Unable to save tensor {pair.Key} with shard size limit {max_size}");

            if (current_size + size > max_size)
            {
                current_shard += 1;
                current_size = 0;
            }

            current_size += size;
            weight_shard.Add(pair.Key, current_shard);
        }

        var total_shards = current_shard + 1;
        foreach (var shard in Enumerable.Range(0, total_shards))
        {
            var path = shardTemplate(shard, total_shards);
            Console.Write($"\rSaving {path} [{shard + 1}/{total_shards}] ");

            var tensors = weight_shard.Where(x => x.Value == shard)
                .ToDictionary(x => x.Key, x => state_dict[x.Key]);
            Safetensors.save_tensors(tensors, path);
        }
        Console.WriteLine();

        File.WriteAllText(
            Path.Join(basePath, model_config_file),
            JsonSerializer.Serialize(model_config, TupleJsonSerializerOptions)
        );

        File.WriteAllText(
            Path.Join(basePath, tokenizer_config_file),
            JsonSerializer.Serialize(tokenizer_config, TupleJsonSerializerOptions)
        );
    }
}

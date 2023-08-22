using System.Text;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Utils;
using TupleAsJsonArray;

namespace llm_sharp.LLM.Utils;

using Tensor = torch.Tensor;
using ScalarType = torch.ScalarType;

public class Safetensors : IDisposable
{
    public static class TensorDtype
    {
        [AttributeUsage(AttributeTargets.Field)]
        class TorchDtypeAttribute : Attribute
        {
            public ScalarType type;
            public TorchDtypeAttribute(ScalarType type)
            {
                this.type = type;
            }
        }

        [TorchDtype(ScalarType.Bool)]
        public const string BOOL = "BOOL";
        [TorchDtype(ScalarType.Byte)]
        public const string U8 = "U8";
        [TorchDtype(ScalarType.Int8)]
        public const string I8 = "I8";
        [TorchDtype(ScalarType.Int16)]
        public const string I16 = "I16";
        [TorchDtype(ScalarType.Int16)]
        public const string U16 = "U16";
        [TorchDtype(ScalarType.Float16)]
        public const string F16 = "F16";
        [TorchDtype(ScalarType.BFloat16)]
        public const string BF16 = "BF16";
        [TorchDtype(ScalarType.Int32)]
        public const string I32 = "I32";
        [TorchDtype(ScalarType.Int32)]
        public const string U32 = "U32";
        [TorchDtype(ScalarType.Float32)]
        public const string F32 = "F32";
        [TorchDtype(ScalarType.Float64)]
        public const string F64 = "F64";
        [TorchDtype(ScalarType.Int64)]
        public const string I64 = "I64";
        [TorchDtype(ScalarType.Int64)]
        public const string U64 = "U64";

        public static Tensor create_tensor(string dtype, long[] shape)
        {
            var field = typeof(TensorDtype).GetField(dtype)
                ?? throw new ArgumentException(nameof(dtype));

            var attr = field.GetCustomAttributes(typeof(TorchDtypeAttribute), false)
                .Cast<TorchDtypeAttribute>().First();

            return torch.empty(shape, dtype: attr.type);
        }

        public static string get_type(Tensor tensor)
        {
            var field = typeof(TensorDtype).GetFields().First(x =>
                x.GetCustomAttributes(typeof(TorchDtypeAttribute), false)
                    .Cast<TorchDtypeAttribute>().First().type == tensor.dtype);

            return (string?)field.GetRawConstantValue()
                ?? throw new ArgumentException(nameof(tensor));
        }
    }

    public record TensorInfo
    {
        public string dtype { get; set; } = TensorDtype.F32;
        public List<long> shape { get; set; } = new();
        public (long begin, long end) data_offsets { get; set; } = new();
    }


    protected long header_size;
    protected FileStream stream;
    protected BinaryReader reader;

    public Dictionary<string, string> metadata { get; protected set; } = new();
    public OrderedDict<string, TensorInfo> tensors { get; protected set; } = new();

    public ICollection<string> Keys => tensors.Keys;

    public Safetensors(string path)
    {
        stream = File.Open(path, FileMode.Open);
        reader = new BinaryReader(stream);
        read_head();
    }

    protected void read_head()
    {
        stream.Seek(0, SeekOrigin.Begin);
        header_size = reader.ReadInt64();
        var json = JsonDocument.Parse(reader.ReadBytes((int)header_size));
        var props = json.RootElement.EnumerateObject().ToList();
        foreach (var prop in props)
        {
            if (prop.Name == "__metadata__")
            {
                metadata = prop.Value.Deserialize<Dictionary<string, string>>()
                    ?? throw new ArgumentException(nameof(metadata));
            }
            else
            {
                var tensorInfo = prop.Value.Deserialize<TensorInfo>(serializerOptions)
                    ?? throw new ArgumentException(nameof(tensors));
                tensors.Add(prop.Name, tensorInfo);
            }
        }
    }

    public Tensor read_tensor(string key)
    {
        var tensorInfo = tensors[key];
        var start = 8 + header_size + tensorInfo.data_offsets.begin;
        var size = tensorInfo.data_offsets.end - tensorInfo.data_offsets.begin;

        if (stream.Position != start)
            stream.Seek(start, SeekOrigin.Begin);

        // currently not able to directly create with bytes
        var tensor = TensorDtype.create_tensor(tensorInfo.dtype, tensorInfo.shape.ToArray());
        var raw_data = tensor.bytes;

        if (raw_data.Length != size)
            throw new Exception("Invalid size to read");

        reader.Read(raw_data);

        return tensor;
    }

    public void Dispose()
    {
        reader.Dispose();
        stream.Dispose();
    }

    protected List<string> load_to_state_dict(Dictionary<string, Tensor> state_dict)
    {
        using var nograd = torch.no_grad();
        var unusedKeys = new List<string>();
        foreach (var key in Keys.ToList())
        {
            using var scope = torch.NewDisposeScope();
            if (state_dict.TryGetValue(key, out var tensor))
            {
                var data = read_tensor(key);
                // auto cast float types
                if (tensor.dtype is ScalarType.Float32 or ScalarType.Float16 or ScalarType.BFloat16)
                    data = data.type_as(tensor);

                // cross device copy is allowed
                tensor.copy_(data);
                state_dict.Remove(key);
            }
            else
            {
                unusedKeys.Add(key);
            }
        }
        return unusedKeys;
    }

    public static void load_state_dict(Dictionary<string, Tensor> state_dict, params string[] paths)
    {
        var cloned = new Dictionary<string, Tensor>(state_dict);
        var unusedKeys = new List<string>();

        foreach (var (idx, path) in paths.Select((x, i) => (i + 1, x)))
        {
            Console.Write($"\rloading {path} [{idx}/{paths.Length}] ");
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


    protected static JsonSerializerOptions serializerOptions = new JsonSerializerOptions
    {
        Converters = { new TupleConverterFactory() },
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    protected static int nextMultipleOfN(int num, int n)
    {
        num += n - 1;
        return num - num % n;
    }

    public static void save_tensors(Dictionary<string, Tensor> tensors, string path, Dictionary<string, string>? metadata = null)
    {
        var headerJson = new Dictionary<string, object?>();

        var tensorList = tensors.ToList();
        tensorList.Sort((a, b) => a.Key.CompareTo(b.Key));

        var offset = 0;
        foreach (var pair in tensorList)
        {
            var size = pair.Value.bytes.Length;
            var begin = offset;
            var end = begin + size;
            var info = new TensorInfo()
            {
                dtype = TensorDtype.get_type(pair.Value),
                shape = pair.Value.shape.ToList(),
                data_offsets = (begin, end),
            };
            headerJson.Add(pair.Key, info);
            offset = end;
        }

        if (metadata is not null)
            headerJson.Add("__metadata__", metadata);

        var header = Encoding.UTF8.GetBytes(
            JsonSerializer.Serialize(headerJson, serializerOptions)
        ).ToList();

        // pad to multiple of 8 with spaces
        var expectedSize = nextMultipleOfN(header.Count, 8);
        while (header.Count < expectedSize)
            header.Add((byte)' ');

        using var file = File.OpenWrite(path);
        using var writer = new BinaryWriter(file);

        writer.Write((long)header.Count);
        writer.Write(header.ToArray());

        foreach (var pair in tensorList)
            writer.Write(pair.Value.bytes);
    }

    public static void save_state_dict(Dictionary<string, Tensor> state_dict, Func<int, int, string> pathTemplate, int? shard_size = null)
    {
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
            var path = pathTemplate(shard, total_shards);
            Console.Write($"\rsaving {path} [{shard + 1}/{total_shards}] ");

            var tensors = new Dictionary<string, Tensor>(
                weight_shard.Where(x => x.Value == shard)
                    .Select(x => KeyValuePair.Create(x.Key, state_dict[x.Key]))
            );
            save_tensors(tensors, path);
        }
        Console.WriteLine();
    }
}

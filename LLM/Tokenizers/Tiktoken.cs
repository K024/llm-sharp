using System.Collections;
using System.Text;
using System.Text.RegularExpressions;

namespace llm_sharp.LLM.Tokenizers;

public class ByteArrayComparer : IEqualityComparer<byte[]>
{
    public bool Equals(byte[]? obj1, byte[]? obj2) =>
      StructuralComparisons.StructuralEqualityComparer.Equals(obj1, obj2);
    public int GetHashCode(byte[] obj) =>
      StructuralComparisons.StructuralEqualityComparer.GetHashCode(obj);
}

public record TikTokenConfig
{
    public List<string> eos_tokens { get; set; } = new();
    public Dictionary<string, int> ranks { get; set; } = new();
    public Dictionary<string, int> special_tokens { get; set; } = new();
    public string pattern { get; set; } = "";
}

public class TikToken
{
    // modified from https://github.com/openai/tiktoken/blob/main/src/lib.rs
    protected static List<T> byte_pair_merge<T>(
        byte[] piece,
        IReadOnlyDictionary<byte[], int> ranks,
        Func<Range, T> f)
    {
        var parts = Enumerable.Range(0, piece.Length + 1)
          .Select(i => (index: i, rank: int.MaxValue)).ToList();

        int? GetRank(int startIdx, int skip)
        {
            if (startIdx + skip + 2 < parts.Count)
            {
                var range = piece[parts[startIdx].index..parts[startIdx + skip + 2].index];
                if (ranks.TryGetValue(range, out int rank))
                    return rank;
            }
            return null;
        }

        for (var i = 0; i < parts.Count - 2; i++)
        {
            var rank = GetRank(i, 0);
            if (rank.HasValue)
            {
                if (rank.Value == int.MaxValue)
                    throw new Exception("Rank value cannot be int.MaxValue.");
                parts[i] = (parts[i].index, rank.Value);
            }
        }

        while (parts.Count > 1)
        {
            if (parts.Count == 1)
                break;

            var minRank = (rank: int.MaxValue, index: 0);
            for (var i = 0; i < parts.Count - 1; i++)
            {
                if (parts[i].rank < minRank.rank)
                    minRank = (parts[i].rank, i);
            }

            if (minRank.rank != int.MaxValue)
            {
                var i = minRank.index;

                parts[i] = (parts[i].index, GetRank(i, 1) ?? int.MaxValue);
                if (i > 0)
                    parts[i - 1] = (parts[i - 1].index, GetRank(i - 1, 1) ?? int.MaxValue);

                parts.RemoveAt(i + 1);
            }
            else
            {
                break;
            }
        }

        var outList = new List<T>(parts.Count - 1);
        for (int i = 0; i < parts.Count - 1; i++)
        {
            outList.Add(f(new Range(parts[i].index, parts[i + 1].index)));
        }

        return outList;
    }

    protected static List<int> byte_pair_encode(byte[] piece, IReadOnlyDictionary<byte[], int> ranks)
    {
        if (piece.Length == 1)
            return new() { ranks[piece] };
        return byte_pair_merge(piece, ranks, range => ranks[piece[range]]);
    }

    protected static List<byte[]> byte_pair_split(byte[] piece, IReadOnlyDictionary<byte[], int> ranks)
    {
        if (piece.Length == 1)
            return new() { piece };
        return byte_pair_merge(piece, ranks, range => piece[range]);
    }

    public virtual TikTokenConfig config { get; }

    protected Regex pattern;
    protected Regex special_pattern;
    protected Dictionary<byte[], int> encoder;
    protected Dictionary<int, byte[]> decoder;
    protected Dictionary<byte[], int> special_token_encoder;
    protected Dictionary<int, byte[]> special_token_decoder;

    public TikToken(TikTokenConfig config)
    {
        this.config = config;

        pattern = new Regex(config.pattern);
        special_pattern = new Regex(string.Join(
            "|",
            config.special_tokens.Keys.Select(x => Regex.Escape(x))
        ));

        encoder = new(
            config.ranks.ToList().Select(pair =>
                KeyValuePair.Create(Convert.FromBase64String(pair.Key), pair.Value)),
            new ByteArrayComparer()
        );

        decoder = new(
            encoder.ToList().Select(x => KeyValuePair.Create(x.Value, x.Key))
        );

        if (encoder.Count != decoder.Count)
            throw new Exception("Possible duplicated rank id");

        special_token_encoder = new(
            config.special_tokens.ToList().Select(pair =>
                KeyValuePair.Create(Encoding.UTF8.GetBytes(pair.Key), pair.Value)),
            new ByteArrayComparer()
        );

        special_token_decoder = new(
            special_token_encoder.ToList().Select(x => KeyValuePair.Create(x.Value, x.Key))
        );

        if (special_token_encoder.Count != special_token_decoder.Count)
            throw new Exception("Possible duplicated special_token id");
    }

    public virtual int Count => encoder.Count + special_token_encoder.Count;
    public virtual int OrdinaryTokens => encoder.Count;

    public virtual int this[string token]
    {
        get
        {
            var piece = Encoding.UTF8.GetBytes(token);
            if (encoder.TryGetValue(piece, out var id))
                return id;
            return special_token_encoder[piece];
        }
    }

    public virtual string this[int id]
    {
        get
        {
            var piece = decoder.GetValueOrDefault(id)
                ?? special_token_decoder.GetValueOrDefault(id)
                    ?? throw new Exception($"Unable to decode id {id}");
            return Encoding.UTF8.GetString(piece);
        }
    }

    public virtual List<int> eos_ids => config.eos_tokens.Select(x => this[x]).ToList();

    public virtual List<int> encode_ordinary_text(string text)
    {
        var ret = new List<int>();
        foreach (var match in pattern.Matches(text).ToList())
        {
            var piece = Encoding.UTF8.GetBytes(match.Value);
            if (encoder.TryGetValue(piece, out var value))
            {
                ret.Add(value);
                continue;
            }
            ret.AddRange(byte_pair_encode(piece, encoder));
        }
        return ret;
    }

    public virtual List<int> encode_text(string text)
    {
        var ret = new List<int>();
        var start = 0;
        foreach (var match in special_pattern.Matches(text).ToList())
        {
            var special_start = match.Index;

            if (start < special_start)
                ret.AddRange(encode_ordinary_text(text[start..special_start]));

            var special = Encoding.UTF8.GetBytes(match.Value);
            ret.Add(special_token_encoder[special]);

            start = match.Index + match.Length;
        }
        if (start < text.Length)
            ret.AddRange(encode_ordinary_text(text[start..]));
        return ret;
    }

    public virtual string decode_text(IReadOnlyList<int> ids)
    {
        var buffer = new List<byte>(ids.Count * 2);
        foreach (var id in ids)
        {
            var bytes = decoder.GetValueOrDefault(id)
                ?? special_token_decoder.GetValueOrDefault(id)
                    ?? throw new Exception($"Unable to decode id {id}");
            buffer.AddRange(bytes);
        }
        return Encoding.UTF8.GetString(buffer.ToArray());
    }
}

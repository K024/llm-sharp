using System.Collections;
using System.Text;
using System.Text.RegularExpressions;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Tokenizers;

public record BPEConfig
{
    public List<string> eos_tokens { get; set; } = new();
    public Dictionary<string, int> vocab { get; set; } = new();
    public List<string> merges { get; set; } = new();
    public Dictionary<string, int> special_tokens { get; set; } = new();
    public string pattern { get; set; } = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
}

public class BPE
{
    // modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py

    // BPE works on UTF-8 bytes representation.
    // To avoid saving corrupted string segments, map each UTF-8 byte to valid unicode character.
    protected static (Dictionary<byte, char>, Dictionary<char, byte>) bytes_to_unicode()
    {
        IEnumerable<int> range(int start, int end) => Enumerable.Range(start, end - start);

        // visible chars (except space) in Latin-1 charset kept same
        var bs = range('!', '~' + 1)
            .Concat(range('¡', '¬' + 1))
            .Concat(range('®', 'ÿ' + 1))
            .ToList();

        var cs = bs.ToList();

        // map these bytes to outside of Latin-1 charset (eg. ' ' to 'Ġ')
        var bs_others = Enumerable.Range(0, 256).Where(x => !bs.Contains(x)).ToList();
        var cs_others = bs_others.Select((x, index) => index + 256);

        bs.AddRange(bs_others);
        cs.AddRange(cs_others);

        var pairs = new Dictionary<byte, char>(Enumerable.Zip(bs, cs).Select(pair =>
            KeyValuePair.Create((byte)pair.First, (char)pair.Second)));

        return (pairs, pairs.ReverseDictionary());
    }

    protected static Lazy<(Dictionary<byte, char>, Dictionary<char, byte>)> _bytes_to_unicode =>
        new(bytes_to_unicode);

    protected static Dictionary<byte, char> byte_encoder => _bytes_to_unicode.Value.Item1;
    protected static Dictionary<char, byte> byte_decoder => _bytes_to_unicode.Value.Item2;

    protected static List<string> byte_pair_merge(
        string[] pieces,
        IReadOnlyDictionary<(string, string), int> bpe_ranks)
    {
        IEnumerable<(string, string)> get_pairs(IList<string> list)
        {
            var last = list.First();
            foreach (var next in list.Skip(1))
            {
                yield return (last, next);
                last = next;
            }
        }

        var list = pieces.ToList();
        while (list.Count > 1)
        {
            var (index, bigram, rank) = get_pairs(list)
                .Select((x, index) => (
                    index, x,
                    rank: bpe_ranks.GetValueOrDefault(x, int.MaxValue)))
                .MinBy(x => x.rank);

            if (rank == int.MaxValue)
                break;

            var (first, second) = bigram;

            list[index] = first + second;
            list.RemoveAt(index + 1);
        }
        return list;
    }

    public virtual BPEConfig config { get; }

    protected Regex pattern;
    protected Regex special_pattern;
    protected Dictionary<string, int> encoder;
    protected Dictionary<int, string> decoder;
    protected Dictionary<string, int> special_token_encoder;
    protected Dictionary<int, string> special_token_decoder;
    protected Dictionary<(string, string), int> bpe_ranks;

    public BPE(BPEConfig config)
    {
        this.config = config;

        pattern = new Regex(config.pattern);
        special_pattern = new Regex(string.Join(
            "|",
            config.special_tokens.Keys.Select(x => Regex.Escape(x))
        ));

        encoder = config.vocab;
        decoder = encoder.ReverseDictionary();

        special_token_encoder = config.special_tokens;
        special_token_decoder = special_token_encoder.ReverseDictionary();

        bpe_ranks = new(
            config.merges
                .Select(x => x.Split(' ', 2)).Select(x => (x[0], x[1]))
                .Select((x, rank) => KeyValuePair.Create(x, rank))
        );
    }

    public virtual int Count => encoder.Count + special_token_encoder.Count;
    public virtual int OrdinaryTokens => encoder.Count;

    public virtual int this[string token]
    {
        get
        {
            if (encoder.TryGetValue(token, out var id))
                return id;
            return special_token_encoder[token];
        }
    }

    public virtual string this[int id]
    {
        get
        {
            var piece = special_token_decoder.GetValueOrDefault(id)
                ?? decoder.GetValueOrDefault(id)
                    ?? throw new Exception($"Unable to decode id {id}");
            return piece;
        }
    }

    public virtual List<int> eos_ids => config.eos_tokens.Select(x => this[x]).ToList();

    public virtual List<int> encode_ordinary_text(string text)
    {
        var ret = new List<int>();
        foreach (var match in pattern.Matches(text).ToList())
        {
            var piece = Encoding.UTF8.GetBytes(match.Value)
                .Select(x => byte_encoder[x]).ToArray();

            // it is safe to convert a char (in utf-16) to string
            // because it's just mapped from utf-8 bytes (with max ranging 0 ~ 512)
            var merged = byte_pair_merge(piece.Select(x => x.ToString()).ToArray(), bpe_ranks);
            ret.AddRange(merged.Select(x => encoder[x]));
        }
        return ret;
    }

    public virtual List<int> encode_text(string text)
    {
        var ret = new List<int>();
        var start = 0;
        foreach (var match in special_pattern.Matches(text).ToList())
        {
            if (match.Length <= 0)
                continue;

            var special_start = match.Index;

            if (start < special_start)
                ret.AddRange(encode_ordinary_text(text[start..special_start]));

            ret.Add(special_token_encoder[match.Value]);

            start = match.Index + match.Length;
        }
        if (start < text.Length)
            ret.AddRange(encode_ordinary_text(text[start..]));
        return ret;
    }

    public virtual string decode_text(IReadOnlyList<int> ids)
    {
        var buffer = new StringBuilder();
        foreach (var id in ids)
        {
            var piece = special_token_decoder.GetValueOrDefault(id)
                ?? decoder.GetValueOrDefault(id)
                    ?? throw new Exception($"Unable to decode id {id}");

            var decoded = Encoding.UTF8.GetString(
                piece.Select(x => byte_decoder[x]).ToArray()
            );
            buffer.Append(decoded);
        }
        return buffer.ToString();
    }
}

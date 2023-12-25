using System.Collections;
using System.Text;
using System.Text.RegularExpressions;

namespace llm_sharp.LLM.Tokenizers;

public record SentencePieceBPEConfig
{
    public List<string> eos_tokens { get; set; } = new();
    public Dictionary<string, int> vocab { get; set; } = new();
    public List<string> merges { get; set; } = new();
    public Dictionary<string, int> special_tokens { get; set; } = new();
    public bool add_dummy_prefix { get; set; } = true;
    public bool remove_extra_whitespaces { get; set; } = true;
}

public class SentencePieceBPE : BPE
{
    // SentencePieceBPE works directly on unicode characters
    public new SentencePieceBPEConfig config;
    public SentencePieceBPE(SentencePieceBPEConfig config)
        : base(new()
        {
            eos_tokens = config.eos_tokens,
            vocab = config.vocab,
            merges = config.merges,
            special_tokens = config.special_tokens,
            pattern = "^$", // no pattern
        })
    {
        // SentencePiece BPE is expected to have byte fallback
        foreach (var b in Enumerable.Range(0, 256))
        {
            var str = $"<0x{b:X2}>";
            var token = encoder[str];
            byte_encoder[(byte)b] = token;
            byte_decoder[token] = (byte)b;
        }
        this.config = config;
    }

    protected new Dictionary<byte, int> byte_encoder = new();
    protected new Dictionary<int, byte> byte_decoder = new();

    public override List<int> encode_ordinary_text(string text)
    {
        if (config.remove_extra_whitespaces)
            text = text.Trim();
        if (string.IsNullOrEmpty(text))
            return new();

        if (config.add_dummy_prefix && !text.StartsWith(" "))
            text = " " + text;
        // replace all spaces to '▁'
        var piece = text.Replace(' ', '▁');

        // split by runes when working with sentencepiece
        var merged = byte_pair_merge(
            piece.EnumerateRunes().Select(x => x.ToString()).ToArray(),
            bpe_ranks);

        var ret = new List<int>();

        foreach (var word in merged)
        {
            if (encoder.TryGetValue(word, out var token))
            {
                ret.Add(token);
            }
            else
            {
                // byte fallback
                ret.AddRange(Encoding.UTF8.GetBytes(word)
                    .Select(x => byte_encoder[x]));
            }
        }
        return ret;
    }

    public override string decode_text(IReadOnlyList<int> ids)
    {
        var continuous_bytes = new List<byte>();
        var buffer = new StringBuilder();

        void flush_bytes()
        {
            if (continuous_bytes.Count > 0)
            {
                buffer.Append(Encoding.UTF8.GetString(continuous_bytes.ToArray()));
                continuous_bytes.Clear();
            }
        }

        foreach (var id in ids)
        {
            if (byte_decoder.ContainsKey(id))
            {
                // decode byte fallback
                continuous_bytes.Add(byte_decoder[id]);
            }
            else
            {
                flush_bytes();
                var piece = special_token_decoder.GetValueOrDefault(id)
                    ?? decoder.GetValueOrDefault(id)
                        ?? throw new Exception($"Unable to decode id {id}");

                buffer.Append(piece);
            }
        }
        flush_bytes();

        // remove prepended space and replace back spaces
        var result = buffer.ToString().Replace('▁', ' ');
        if (config.add_dummy_prefix && result.StartsWith(" "))
            result = result[1..];
        return result;
    }
}

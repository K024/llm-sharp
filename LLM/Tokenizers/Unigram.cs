using System.Collections;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using llm_sharp.LLM.Utils;
using Microsoft.AspNetCore.WebUtilities;

namespace llm_sharp.LLM.Tokenizers;

public record UnigramConfig
{
    public List<string> eos_tokens { get; set; } = new();
    public string? unk_token { get; set; }
    public List<(string, double)> vocab { get; set; } = new();
    public Dictionary<string, int> special_tokens { get; set; } = new();
    public string? precompiled_charsmap { get; set; }
}

public class Unigram : ITokenizer
{
    public static List<string> unigram_viterbi(
        string sequence,
        IReadOnlyDictionary<string, double> log_probs,
        Trie<Rune> trie,
        double min_log_prob)
    {
        // sentencepiece works on unicode code points
        var runes = sequence.EnumerateRunes().ToArray();
        var state = Enumerable.Range(0, runes.Length + 1)
            .Select((x, index) => (log_prob: (min_log_prob - 1) * index, track: index - 1))
            .ToArray();

        state[0] = (0, 0);
        foreach (var index in Enumerable.Range(0, runes.Length))
        {
            foreach (var length in trie.PrefixSearch(runes.Skip(index)))
            {
                var substr = string.Join(null, runes[index..(index + length)]);

                var next_index = index + length;
                var next_prob = state[index].log_prob + log_probs[substr];
                if (next_prob > state[next_index].log_prob)
                {
                    state[next_index] = (next_prob, index);
                }
            }
        }
        var backtrack = new List<int>();
        var cur = runes.Length;
        while (cur != 0)
        {
            backtrack.Add(cur);
            cur = state[cur].track;
        }

        var result = new List<string>();
        var start = 0;
        foreach (var end in backtrack.AsEnumerable().Reverse())
        {
            result.Add(string.Join(null, runes[start..end]));
            start = end;
        }
        return result;
    }

    public virtual UnigramConfig config { get; }

    protected Regex special_pattern;
    protected Dictionary<string, int> encoder;
    protected Dictionary<int, string> decoder;
    protected Dictionary<string, int> special_token_encoder;
    protected Dictionary<int, string> special_token_decoder;
    protected Dictionary<string, double> log_probs;
    protected Trie<Rune> trie;
    protected double min_log_prob;
    protected SentencePiecePrecompiledCharsMap? normalizer;

    protected Dictionary<byte, int> byte_encoder = new();
    protected Dictionary<int, byte> byte_decoder = new();

    public Unigram(UnigramConfig config)
    {
        this.config = config;

        special_pattern = new Regex(string.Join(
            "|",
            config.special_tokens.Keys.Select(x => Regex.Escape(x))
        ));

        encoder = config.vocab.Select((pair, index) => (key: pair.Item1, index))
            .ToDictionary(x => x.key, x => x.index);
        decoder = encoder.ReverseDictionary();

        special_token_encoder = config.special_tokens;
        special_token_decoder = special_token_encoder.ReverseDictionary();

        trie = new();
        trie.AddRange(encoder.Keys.Select(x => x.EnumerateRunes().AsEnumerable()));

        log_probs = config.vocab.ToDictionary(x => x.Item1, x => x.Item2);
        min_log_prob = log_probs.Values.Min();

        if (config.precompiled_charsmap is not null)
        {
            normalizer = new(config.precompiled_charsmap);
        }

        // detect byte fallback
        if (encoder.ContainsKey("<0x00>"))
        {
            foreach (var b in Enumerable.Range(0, 256))
            {
                var str = $"<0x{b:X2}>";
                var token = encoder[str];
                byte_encoder[(byte)b] = token;
                byte_decoder[token] = (byte)b;
            }
        }
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
            var token = special_token_decoder.GetValueOrDefault(id)
                ?? decoder.GetValueOrDefault(id)
                    ?? throw new Exception($"Unable to decode id {id}");
            return token;
        }
    }

    public virtual List<int> eos_ids => config.eos_tokens.Select(x => this[x]).ToList();

    public virtual List<int> encode_ordinary_text(string text)
    {
        text = text.Trim();
        if (string.IsNullOrEmpty(text))
            return new();

        if (normalizer is not null)
            text = normalizer.normalize(text);

        // prepend space and replace all spaces to '▁'
        var piece = (" " + text).Replace(' ', '▁');

        var tokenized = unigram_viterbi(piece, log_probs, trie, min_log_prob);

        var result = new List<int>();
        var unknowns = new List<string>();

        void flush_unknowns()
        {
            if (unknowns.Count <= 0)
                return;

            if (byte_encoder.Count > 0)
            {
                // byte fallback
                result.AddRange(
                    Encoding.UTF8.GetBytes(string.Join(null, unknowns))
                        .Select(x => byte_encoder[x]));
            }
            else
            {
                // merge to single unk token
                result.Add(special_token_encoder[config.unk_token
                    ?? throw new Exception("'unk_token' is null and no byte fallback tokens")]);
            }
            unknowns.Clear();
        }

        foreach (var token in tokenized)
        {
            if (encoder.TryGetValue(token, out var id))
            {
                flush_unknowns();
                result.Add(id);
            }
            else
            {
                unknowns.Add(token);
            }
        }
        flush_unknowns();

        return result;
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
        var result = buffer.ToString().Replace('▁', ' ').TrimStart(' ');
        return result;
    }
}

// It's ...confusing
// from https://github.com/huggingface/spm_precompiled/blob/master/src/lib.rs
public class SentencePiecePrecompiledCharsMap
{
    protected class DoubleArray : List<uint>
    {
        static bool has_leaf(uint value) => ((value >> 8) & 1) == 1;
        static uint value(uint value) => value & ((1u << 31) - 1);
        static uint label(uint value) => value & ((1u << 31) | 0xFF);
        static uint offset(uint value) => (value >> 10) << (int)((value & (1u << 9)) >> 6);

        public List<int> common_prefix_search(byte[] key)
        {
            var node_pos = 0u;
            var results = new List<int>();

            var unit = this[(int)node_pos];
            node_pos ^= offset(unit);
            foreach (var c in key)
            {
                if (c == 0)
                    break;

                node_pos ^= (uint)c;
                unit = this[(int)node_pos];
                if (label(unit) != c)
                    return results;

                node_pos ^= offset(unit);
                if (has_leaf(unit))
                    results.Add((int)value(this[(int)node_pos]));
            }
            return results;
        }
    }

    protected DoubleArray trie;
    protected byte[] normalized;

    public SentencePiecePrecompiledCharsMap(string charsmap)
    {
        var bytes = Base64UrlTextEncoder.Decode(charsmap);
        using var reader = new BinaryReader(new MemoryStream(bytes));

        var trieSize = reader.ReadInt32() / 4;
        trie = new DoubleArray();
        for (var i = 0; i < trieSize; i++)
        {
            trie.Add(reader.ReadUInt32());
        }
        normalized = bytes[(int)reader.BaseStream.Position..];
    }

    public string? transform(string chunk)
    {
        var results = trie.common_prefix_search(Encoding.UTF8.GetBytes(chunk));

        if (results.Count <= 0)
            return null;

        var index = results[0];
        var index2 = index;
        while (index2 < normalized.Length &&
            normalized[index2] != 0)
            index2++;

        return Encoding.UTF8.GetString(normalized[index..index2]);
    }

    public string normalize(string original)
    {
        var result = new StringBuilder();
        var e = StringInfo.GetTextElementEnumerator(original);
        while (e.MoveNext())
        {
            var grapheme = e.GetTextElement();
            if (Encoding.UTF8.GetByteCount(grapheme) < 6)
            {
                if (transform(grapheme) is string norm)
                {
                    result.Append(norm);
                    continue;
                }
            }
            foreach (var part in grapheme.EnumerateRunes())
            {
                if (transform(part.ToString()) is string norm)
                    result.Append(norm);
                else
                    result.Append(part);
            }
        };
        return result.ToString();
    }
}

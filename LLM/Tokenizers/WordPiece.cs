using System.Collections;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Tokenizers;

public record WordPieceConfig
{
    public string unk_token { get; set; } = "";
    public string continuing_prefix { get; set; } = "";
    public BasicTokenizeConfig? basic_tokenize { get; set; }
    public Dictionary<string, int> vocab { get; set; } = new();
    public Dictionary<string, int> special_tokens { get; set; } = new();

    public record BasicTokenizeConfig
    {
        public bool do_lower_case { get; set; } = true;
        public bool tokenize_chinese_chars { get; set; } = true;
        public bool strip_accents { get; set; } = true;
    }
}

public class WordPiece
{
    // modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py

    protected static List<int> encode_wordpiece(
        string pretokenized,
        Trie<char> trie,
        IReadOnlyDictionary<string, int> vocab,
        string prefix,
        int unk_id)
    {
        var words = pretokenized.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var output = new List<int>();
        foreach (var word in words)
        {
            var is_bad = false;
            var start = 0;
            var sub_tokens = new List<int>();

            while (start < word.Length)
            {
                var sequence = word.Skip(start);
                if (start > 0)
                    sequence = prefix.Concat(sequence);

                // find longest sub token
                var length = trie.PrefixSearch(sequence).LastOrDefault(-1);
                if (start > 0)
                    length -= prefix.Length;

                if (length <= 0)
                {
                    is_bad = true;
                    break;
                }

                var substr = word[start..(start + length)];
                sub_tokens.Add(vocab[substr]);
                start += length;
            }

            if (is_bad)
                output.Add(unk_id);
            else
                output.AddRange(sub_tokens);
        }
        return output;
    }

    public virtual WordPieceConfig config { get; }

    protected Regex special_pattern;
    protected Dictionary<string, int> encoder;
    protected Dictionary<int, string> decoder;
    protected Dictionary<string, int> special_token_encoder;
    protected Dictionary<int, string> special_token_decoder;
    protected Trie<char> trie;

    public WordPiece(WordPieceConfig config)
    {
        this.config = config;

        special_pattern = new Regex(string.Join(
            "|",
            config.special_tokens.Keys.Select(x => Regex.Escape(x))
        ));

        encoder = config.vocab;
        decoder = encoder.ReverseDictionary();

        special_token_encoder = config.special_tokens;
        special_token_decoder = special_token_encoder.ReverseDictionary();

        trie = new();
        trie.AddRange(encoder.Keys);
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

    public virtual List<int> encode_ordinary_text(string text)
    {
        var ret = new List<int>();
        var tokens = new List<string>() { text };

        if (config.basic_tokenize is not null)
            tokens = WordPieceBasicTokenizer.pre_tokenize(config.basic_tokenize, text).ToList();

        foreach (var token in tokens)
        {
            ret.AddRange(encode_wordpiece(
                token,
                trie,
                encoder,
                config.continuing_prefix,
                special_token_encoder[config.unk_token]
            ));
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
        var ret = new List<string>();
        foreach (var id in ids)
        {
            var str = special_token_decoder.GetValueOrDefault(id)
                ?? decoder.GetValueOrDefault(id)
                    ?? throw new Exception($"Unable to decode id {id}");
            ret.Add(str);
        }
        var result = string.Join(' ', ret).Replace(" " + config.continuing_prefix, "");
        if (config.basic_tokenize is not null)
            result = WordPieceBasicTokenizer.cleanup_decoded(config.basic_tokenize, result);
        return result;
    }
}

static class WordPieceBasicTokenizer
{
    public static bool is_chinese_char(Rune cp)
    {
        if (cp.Value
                is (>= 0x4E00 and <= 0x9FFF)
                or (>= 0x3400 and <= 0x4DBF)
                or (>= 0x20000 and <= 0x2A6DF)
                or (>= 0x2A700 and <= 0x2B73F)
                or (>= 0x2B740 and <= 0x2B81F)
                or (>= 0x2B820 and <= 0x2CEAF)
                or (>= 0xF900 and <= 0xFAFF)
                or (>= 0x2F800 and <= 0x2FA1F))
            return true;
        return false;
    }

    public static string clean_text(string text)
    {
        var runes = text.EnumerateRunes()
            .Where(x => x.Value is not (0 or 0xFFFD) && !Rune.IsControl(x))
            .Select(x => Rune.IsWhiteSpace(x) ? new Rune(' ') : x);

        return string.Join(null, runes);
    }

    static string tokenize_chinese_chars(string text)
    {
        var runes = text.EnumerateRunes()
            .SelectMany(x =>
                is_chinese_char(x) ? new[] { new(' '), x, new(' ') } : new[] { x }
            );

        return string.Join(null, runes);
    }

    static string strip_accents(string text)
    {
        var runes = text.Normalize(NormalizationForm.FormD)
            .EnumerateRunes()
            .Where(x =>
                Rune.GetUnicodeCategory(x) != UnicodeCategory.NonSpacingMark // Mn
            );

        return string.Join(null, runes);
    }

    static IEnumerable<string> split_on_punctuation(string text)
    {
        var tokens = new List<List<Rune>>();

        var start_new = true;
        foreach (var c in text.EnumerateRunes())
        {
            if (Rune.IsPunctuation(c))
            {
                tokens.Add(new() { c });
                start_new = true;
            }
            else
            {
                if (start_new)
                    tokens.Add(new());
                start_new = false;
                tokens[^1].Add(c);
            }
        }

        return tokens.Select(x => string.Join(null, x));
    }

    public static IEnumerable<string> pre_tokenize(WordPieceConfig.BasicTokenizeConfig config, string text)
    {
        text = clean_text(text);

        if (config.tokenize_chinese_chars)
            text = tokenize_chinese_chars(text);

        text = text.Normalize(NormalizationForm.FormC);

        var tokens = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        tokens = tokens.SelectMany(token =>
        {
            if (config.do_lower_case)
                token = token.ToLowerInvariant();

            if (config.strip_accents)
                token = strip_accents(token);

            return split_on_punctuation(token);
        }).ToArray();

        text = string.Join(' ', tokens);

        return text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    }

    public static string cleanup_decoded(WordPieceConfig.BasicTokenizeConfig config, string decoded)
    {
        var runes = decoded.EnumerateRunes().ToList();
        for (var i = runes.Count - 2; i >= 0; i--)
        {
            if (Rune.IsWhiteSpace(runes[i]) &&
                Rune.IsPunctuation(runes[i + 1]))
                runes.RemoveAt(i);
        }
        if (config.tokenize_chinese_chars)
        {
            for (var i = runes.Count - 2; i > 0; i--)
            {
                if (is_chinese_char(runes[i - 1]) &&
                    Rune.IsWhiteSpace(runes[i]) &&
                    is_chinese_char(runes[i + 1]))
                    runes.RemoveAt(i);
            }
        }
        return string.Join(null, runes).Trim();
    }
}

// from https://code-maze.com/csharp-using-trie-class-for-efficient-text-pattern-searching/
public class Trie<T> where T : notnull
{
    public class TrieNode
    {
        public bool IsWord { get; set; } = false;
        public Dictionary<T, TrieNode> Children { get; } = new();
    }

    private readonly TrieNode _root = new TrieNode();

    public void Add(IEnumerable<T> word)
    {
        var node = _root;
        foreach (var c in word)
        {
            if (!node.Children.ContainsKey(c))
                node.Children[c] = new TrieNode();
            node = node.Children[c];
        }
        node.IsWord = true;
    }

    public void AddRange(IEnumerable<IEnumerable<T>> words)
    {
        foreach (var word in words)
            Add(word);
    }

    public bool Contains(IEnumerable<T> word)
    {
        var node = _root;
        foreach (var c in word)
        {
            if (!node.Children.ContainsKey(c))
                return false;
            node = node.Children[c];
        }
        return node.IsWord;
    }

    public IEnumerable<int> PrefixSearch(IEnumerable<T> word)
    {
        var depth = 0;
        var node = _root;
        foreach (var c in word)
        {
            if (!node.Children.ContainsKey(c))
                yield break;

            node = node.Children[c];
            depth += 1;

            if (node.IsWord)
                yield return depth;
        }
    }
}

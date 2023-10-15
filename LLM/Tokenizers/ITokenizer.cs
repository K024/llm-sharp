
namespace llm_sharp.LLM.Tokenizers;

public interface ITokenizer
{
    public abstract int Count { get; }
    public abstract int OrdinaryTokens { get; }

    public abstract int this[string token] { get; }

    public abstract string this[int id] { get; }

    public abstract List<int> encode_ordinary_text(string text);

    public abstract List<int> encode_text(string text);

    public abstract string decode_text(IReadOnlyList<int> ids);
}

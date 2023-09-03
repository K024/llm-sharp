using TorchSharp;

namespace llm_sharp.LLM.Utils;

public static class Extensions
{
    public static Dictionary<TValue, TKey> ReverseDictionary<TKey, TValue>(this IReadOnlyDictionary<TKey, TValue> dict)
        where TValue : notnull
    {
        // will throws if has duplicated key
        return new(dict.ToDictionary(x => x.Value, x => x.Key));
    }

    public static void Deconstruct<T>(this T[] srcArray, out T a0)
    {
        if (srcArray == null || srcArray.Length != 1)
            throw new ArgumentException(nameof(srcArray));

        a0 = srcArray[0];
    }

    public static void Deconstruct<T>(this T[] srcArray, out T a0, out T a1)
    {
        if (srcArray == null || srcArray.Length != 2)
            throw new ArgumentException(nameof(srcArray));

        a0 = srcArray[0];
        a1 = srcArray[1];
    }

    public static void Deconstruct<T>(this T[] srcArray, out T a0, out T a1, out T a2)
    {
        if (srcArray == null || srcArray.Length != 3)
            throw new ArgumentException(nameof(srcArray));

        a0 = srcArray[0];
        a1 = srcArray[1];
        a2 = srcArray[2];
    }

    public static void Deconstruct<T>(this T[] srcArray, out T a0, out T a1, out T a2, out T a3)
    {
        if (srcArray == null || srcArray.Length != 4)
            throw new ArgumentException(nameof(srcArray));

        a0 = srcArray[0];
        a1 = srcArray[1];
        a2 = srcArray[2];
        a3 = srcArray[3];
    }

    public static void Deconstruct<T>(this T[] srcArray, out T a0, out T a1, out T a2, out T a3, out T a4)
    {
        if (srcArray == null || srcArray.Length != 5)
            throw new ArgumentException(nameof(srcArray));

        a0 = srcArray[0];
        a1 = srcArray[1];
        a2 = srcArray[2];
        a3 = srcArray[3];
        a4 = srcArray[4];
    }

    public static bool IsSame(this torch.Device a, torch.Device b)
    {
        return a.type == b.type && a.index == b.index;
    }
}

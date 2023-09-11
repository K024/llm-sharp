using System.Text.Json;

namespace llm_sharp.LLM.Train;


public record TextDataLine
{
    public string text { get; set; } = "";
}

public record TextPairDataLine
{
    public string text_a { get; set; } = "";
    public string text_b { get; set; } = "";
    public double label { get; set; }
}


public class Dataset<T> : List<T> where T : class
{
    public static Dataset<T> from_jsonl(string path)
    {
        var reader = new StreamReader(File.OpenRead(path));
        var dataset = new Dataset<T>();

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            var data = JsonSerializer.Deserialize<T>(line)
                ?? throw new Exception("Failed to deserialize jsonl file.");
            dataset.Add(data);
        }
        return dataset;
    }

    public IEnumerable<List<T>> iterate_batch(int batch_size, int seed = 0, bool drop_last = false)
    {
        var indices = Enumerable.Range(0, Count).ToList();

        if (seed != 0)
        {
            if (seed < 0)
                seed = Random.Shared.Next();
            var rng = new Random(seed);
            indices = indices.OrderBy(x => rng.Next()).ToList();
        }

        var batch = new List<T>();
        foreach (var idx in indices)
        {
            batch.Add(this[idx]);
            if (batch.Count >= batch_size)
            {
                yield return batch;
                batch = new List<T>();
            }
        }
        if (!drop_last && batch.Count > 0)
            yield return batch;
    }
}

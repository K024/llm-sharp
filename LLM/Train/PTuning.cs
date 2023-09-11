using TorchSharp;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Models;
using TorchSharp.Modules;

namespace llm_sharp.LLM.Train;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;


public class PTuningPrefix : nn.Module<List<(Tensor k_cache, Tensor v_cache)>?, List<(Tensor k_cache, Tensor v_cache)>>
{
    public ParameterDict prefix;

    public PTuningPrefix(
        int layers,
        long n_heads,
        long head_dim,
        long length,
        torch.ScalarType? dtype = null,
        torch.Device? device = null
    ) : base("PTuningPrefix")
    {
        prefix = new ParameterDict();
        for (int i = 0; i < layers; i++)
        {
            var key = nn.Parameter(torch.empty(new[] { 1L, length, n_heads, head_dim }, dtype, device));
            var value = nn.Parameter(torch.empty(new[] { 1L, length, n_heads, head_dim }, dtype, device));
            prefix.Add($"{i}.key", key);
            prefix.Add($"{i}.value", value);

            nn.init.normal_(key);
            nn.init.normal_(value);
        }
        RegisterComponents();
    }

    public override List<(Tensor k_cache, Tensor v_cache)> forward(List<(Tensor k_cache, Tensor v_cache)>? input)
    {
        // when past_key_values are not null
        // the prefix should be already concatenated
        if (input is not null)
            return input;

        var output = new List<(Tensor k_cache, Tensor v_cache)>();
        for (int i = 0; i < prefix.Count; i++)
        {
            var key = (Tensor)prefix[$"{i}.key"];
            var value = (Tensor)prefix[$"{i}.value"];
            output.Add((key, value));
        }

        return output;
    }
}

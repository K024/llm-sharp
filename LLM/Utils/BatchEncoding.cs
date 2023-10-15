using TorchSharp;

namespace llm_sharp.LLM.Utils;

public interface IBatchEncoding
{
    public void to(torch.Device device)
    {
        var tensorProps = GetType().GetProperties()
            .Where(x => x.PropertyType == typeof(torch.Tensor)).ToList();

        foreach (var prop in tensorProps)
        {
            var tensor = prop.GetValue(this) as torch.Tensor;
            if (tensor is not null)
                prop.SetValue(this, tensor.to(device));
        }
    }

    public Dictionary<string, torch.Tensor> to_dict()
    {
        var tensorProps = GetType().GetProperties()
            .Where(x => x.PropertyType == typeof(torch.Tensor)).ToList();

        var dict = new Dictionary<string, torch.Tensor>();
        foreach (var prop in tensorProps)
        {
            var tensor = prop.GetValue(this) as torch.Tensor;
            if (tensor is not null)
                dict.Add(prop.Name, tensor);
        }

        return dict;
    }
}

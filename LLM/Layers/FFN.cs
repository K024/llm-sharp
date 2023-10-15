using TorchSharp;
using TorchSharp.Modules;
using llm_sharp.LLM.Utils;

namespace llm_sharp.LLM.Layers;

using nn = torch.nn;
using Tensor = torch.Tensor;
using F = torch.nn.functional;

public class FeedForward : nn.Module<Tensor, Tensor>
{
    public long hidden_dim;
    public nn.Module<Tensor, Tensor> w_in;
    public nn.Module<Tensor, Tensor> w_out;
    public Dropout dropout;
    public nn.Module<Tensor, Tensor> act_fn;

    public FeedForward(
        long dim,
        long? hidden_dim = null,
        double dropout_rate = 0.0,
        bool bias = true,
        torch.ScalarType? dtype = null,
        torch.Device? device = null,
        nn.Module<Tensor, Tensor>? act_fn = null,
        string act_fn_name = "gelu"
    ) : base("FeedForward")
    {
        this.hidden_dim = hidden_dim ?? dim * 4;
        w_in = new CustomLinear(dim, this.hidden_dim, hasBias: bias, dtype: dtype, device: device);
        w_out = new CustomLinear(this.hidden_dim, dim, hasBias: bias, dtype: dtype, device: device);
        dropout = nn.Dropout(dropout_rate);
        this.act_fn = act_fn ?? Activations.get_activation_by_name(act_fn_name);

        RegisterComponents();
    }

    public FeedForward(
        AbstractBuilder builder, 
        long dim,
        long? hidden_dim = null,
        double dropout_rate = 0.0,
        bool bias = true,
        nn.Module<Tensor, Tensor>? act_fn = null,
        string act_fn_name = "gelu"
    ) : base("FeedForward")
    {
        this.hidden_dim = hidden_dim ?? dim * 4;
        w_in = builder.create_linear(dim, this.hidden_dim, hasBias: bias);
        w_out = builder.create_linear(this.hidden_dim, dim, hasBias: bias);
        dropout = nn.Dropout(dropout_rate);
        this.act_fn = act_fn ?? Activations.get_activation_by_name(act_fn_name);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var h = act_fn.call(w_in.call(x));
        return scope.MoveToOuter(w_out.call(dropout.call(h)));
    }
}

public class GatedFeedForward : nn.Module<Tensor, Tensor>
{
    public long hidden_dim;
    public nn.Module<Tensor, Tensor> w_in;
    public nn.Module<Tensor, Tensor> w_gate;
    public nn.Module<Tensor, Tensor> w_out;
    public Dropout dropout;
    public nn.Module<Tensor, Tensor> act_fn;

    public GatedFeedForward(
        long dim,
        long? hidden_dim = null,
        double dropout_rate = 0.0,
        bool bias = false,
        torch.ScalarType? dtype = null,
        torch.Device? device = null,
        nn.Module<Tensor, Tensor>? act_fn = null,
        string act_fn_name = "silu"
    ) : base("GatedFeedForward")
    {
        this.hidden_dim = hidden_dim ?? dim * 4;
        w_in = new CustomLinear(dim, this.hidden_dim, hasBias: bias, dtype: dtype, device: device);
        w_gate = new CustomLinear(dim, this.hidden_dim, hasBias: bias, dtype: dtype, device: device);
        w_out = new CustomLinear(this.hidden_dim, dim, hasBias: bias, dtype: dtype, device: device);
        dropout = nn.Dropout(dropout_rate);
        this.act_fn = act_fn ?? Activations.get_activation_by_name(act_fn_name);

        RegisterComponents();
    }

    public GatedFeedForward(
        AbstractBuilder builder, 
        long dim,
        long? hidden_dim = null,
        double dropout_rate = 0.0,
        bool bias = false,
        nn.Module<Tensor, Tensor>? act_fn = null,
        string act_fn_name = "silu"
    ) : base("GatedFeedForward")
    {
        this.hidden_dim = hidden_dim ?? dim * 4;
        w_in = builder.create_linear(dim, this.hidden_dim, hasBias: bias);
        w_gate = builder.create_linear(dim, this.hidden_dim, hasBias: bias);
        w_out = builder.create_linear(this.hidden_dim, dim, hasBias: bias);
        dropout = nn.Dropout(dropout_rate);
        this.act_fn = act_fn ?? Activations.get_activation_by_name(act_fn_name);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        var h = act_fn.call(w_gate.call(x)) * w_in.call(x);
        return scope.MoveToOuter(w_out.call(dropout.call(h)));
    }
}

using llm_sharp.LLM.Utils;

namespace llm_sharp.Tests;

[TestClass]
public class UtilsTests
{
    [TestMethod]
    public void StateDictConverter_ShouldConvertName()
    {
        var converts = new Dictionary<string, string>(){
            { "layers.{layer}.attn.qkv_proj.{name}", "transformers.layers.{layer}.attention.qkv_linear.{name}" },
        };
        var converter = new StateDictConverter(converts);

        var source = "layers.11.attn.qkv_proj.some_module.bias";

        Assert.IsTrue(converter.TryConvert(source, out var target));
        Assert.AreEqual("transformers.layers.11.attention.qkv_linear.some_module.bias", target);
    }
}

using llm_sharp.LLM.Utils;
using TorchSharp;

namespace llm_sharp.Tests;

[TestClass]
public class NativeOpsTests
{

    [TestInitialize]
    public void Init() {
        LibTorchLoader.EnsureLoaded();
    }

    [TestMethod]
    public void NativeOps_ShouldWork()
    {
        var result = NativeOps.NativeOps.hello(torch.ones(2, 3, 4, device: torch.CUDA));

        Assert.IsTrue((result == 2).all().item<bool>());
    }
}

using llm_sharp.LLM.Utils;
using llm_sharp.NativeOps;
using TorchSharp;

namespace llm_sharp.Tests;

[TestClass]
public partial class NativeOpsTests
{

    [TestInitialize]
    public void Init()
    {
        LibTorchLoader.EnsureLoaded();
    }

    [TestMethod]
    public void NativeOps_ShouldWork()
    {
        var result = Ops.hello(torch.ones(2, 3, 4, device: torch.CUDA));

        Assert.IsTrue((result == 2).all().item<bool>());
    }

    [TestMethod]
    public void EmptyCache_ShouldWork()
    {
        Ops.cuda_empty_cache();
    }
}

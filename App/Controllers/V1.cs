using Microsoft.AspNetCore.Mvc;

using llm_sharp.Services;

namespace llm_sharp.Controllers;

[ApiController]
[Route("v1")]
public partial class V1Controller : ControllerBase
{
    private LLMService llmService;

    public V1Controller(LLMService llmService)
    {
        this.llmService = llmService;
    }
}

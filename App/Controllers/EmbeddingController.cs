using Microsoft.AspNetCore.Mvc;

using llm_sharp.Services;
using System.ComponentModel.DataAnnotations;
using System.Text;

namespace llm_sharp.Controllers;


[ApiController]
[Route("api/[controller]/[action]")]
public class EmbeddingController : ControllerBase
{
    private LLMService llmService;

    public EmbeddingController(LLMService llmService)
    {
        this.llmService = llmService;
    }

    public class EmbeddingRequest
    {
        public string? model { get; set; }
        [Required]
        public string text { get; set; } = "";
    }

    public class EmbeddingResponse
    {
        public IList<float> embedding { get; set; } = new List<float>();
    }

    [HttpPost]
    [ProducesResponseType(typeof(EmbeddingResponse), 200)]
    public async Task<IActionResult> Encode(EmbeddingRequest request)
    {
        var embedding = await llmService.EncodeAsync(request.model, request.text);
        if (embedding is null)
            return NotFound(new { message = "Model not found" });

        return Ok(new EmbeddingResponse() { embedding = embedding });
    }
}

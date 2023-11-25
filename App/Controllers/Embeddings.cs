using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace llm_sharp.Controllers;


public partial class V1Controller : ControllerBase
{
    public record EmbeddingRequest
    {
        [Required]
        public string model { get; set; } = "";
        [Required]
        public string input { get; set; } = "";
        public string encoding_format { get; set; } = "float";
    }

    public record Embedding
    {
        public int index { get; set; } = 0;
        public string @object { get; set; } = "embedding";
        public IList<float> embedding { get; set; } = new List<float>();
    }

    public record EmbeddingUsage
    {
        public int prompt_tokens { get; set; } = 0;
        public int total_tokens { get; set; } = 0;
    }

    public record EmbeddingResult
    {
        public string @object { get; set; } = "list";
        public string model { get; set; } = "";
        public List<Embedding> data { get; set; } = new();
        public EmbeddingUsage usage { get; set; } = new();
    }

    [Authorize]
    [HttpPost("embeddings")]
    [ProducesResponseType(typeof(EmbeddingResult), 200)]
    public async Task<IActionResult> Embed(EmbeddingRequest request)
    {
        var llm = llmService.FindMaskedLM(request.model);
        if (llm is null)
            return NotFound(new { message = "Model not found" });

        if (request.encoding_format != "float")
            return BadRequest(new { message = "Invalid encoding format" });

        var result = await llm.encode_async(request.input);

        return Ok(new EmbeddingResult()
        {
            model = request.model,
            data = new List<Embedding>()
            {
                new Embedding()
                {
                    index = 0,
                    embedding = result.embedding,
                }
            },
            usage = new EmbeddingUsage()
            {
                prompt_tokens = result.tokens,
                total_tokens = result.tokens,
            }
        });
    }
}

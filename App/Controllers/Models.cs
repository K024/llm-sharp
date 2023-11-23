using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace llm_sharp.Controllers;


public partial class V1Controller : ControllerBase
{
    public record ModelResult
    {
        public string id { get; set; } = "";
        public string @object { get; set; } = "model";
        public long created { get; set; } = 0;
        public string owned_by { get; set; } = "llm-sharp";
    }

    public record ModelsResult
    {
        public string @object { get; set; } = "list";
        public List<ModelResult> data { get; set; } = new();
    }

    [Authorize]
    [HttpGet("models")]
    [ProducesResponseType(typeof(EmbeddingResult), 200)]
    public async Task<IActionResult> Models()
    {
        var start_time = System.Diagnostics.Process.GetCurrentProcess().StartTime;
        var models = llmService.Models.Select(x => new ModelResult()
        {
            id = x,
            created = new DateTimeOffset(start_time).ToUnixTimeSeconds(),
        }).ToList();
        return Ok(new ModelsResult()
        {
            data = models,
        });
    }
}

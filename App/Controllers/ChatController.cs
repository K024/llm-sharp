using Microsoft.AspNetCore.Mvc;

using llm_sharp.Services;
using System.ComponentModel.DataAnnotations;
using System.Text;

namespace llm_sharp.Controllers;


[ApiController]
[Route("api/[controller]/[action]")]
public class ChatController : ControllerBase
{
    private LLMService llmService;

    public ChatController(LLMService llmService)
    {
        this.llmService = llmService;
    }

    public record QuestionAnsweringParams
    {
        public string model { get; set; } = "";
        public bool stream { get; set; } = false;
        [Required]
        public string question { get; set; } = "";
        public List<(string question, string answer)> history { get; set; } = new();
        public double temperature { get; set; } = 1.0;
        public double top_p { get; set; } = 0.8;
    }

    public record QuestionAnsweringResult
    {
        public string answer { get; set; } = "";
    }

    [HttpPost]
    [ProducesResponseType(typeof(QuestionAnsweringResult), 200)]
    public async Task<IActionResult> QuestionAnswering(QuestionAnsweringParams body)
    {
        var history = body.history.Select(x => (x.question, x.answer)).ToList();

        var stream = llmService.ChatAsync(body.model, history, body.question, new()
        {
            top_p = body.top_p,
            temperature = body.temperature,
        });

        if (stream is null)
            return NotFound(new { message = "Model not found" });

        async Task<string> chat()
        {
            var output = new StringBuilder();
            await foreach (var str in stream)
                output.Append(str);
            return output.ToString();
        }

        async IAsyncEnumerable<ServerSentEvent> stream_chat()
        {
            await foreach (var str in stream)
            {
                yield return new ServerSentEvent() { Data = str };
            }
            yield return new ServerSentEvent() { Data = "[DONE]" };
        }

        if (!body.stream)
            return Ok(new QuestionAnsweringResult() { answer = await chat() });

        return this.ServerSentEvents(stream_chat());
    }
}

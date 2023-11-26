using System.ComponentModel.DataAnnotations;
using System.Text;
using System.Text.Json;
using llm_sharp.LLM.Pretrained;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace llm_sharp.Controllers;


public partial class V1Controller : ControllerBase
{
    public record ChatMessage
    {
        [Required]
        public string role { get; set; } = "";
        [Required]
        public string content { get; set; } = "";
        public string? name { get; set; }
    }

    public record ChatCompletionRequest
    {
        [Required]
        public List<ChatMessage> messages { get; set; } = new();
        [Required]
        public string model { get; set; } = "";
        public float frequency_penalty { get; set; } = 0f;
        public int max_tokens { get; set; } = 1024;
        [Range(1, 1)]
        public int n { get; set; } = 1;
        public float presence_penalty { get; set; } = 0f;
        public string? stop { get; set; }
        public float temperature { get; set; } = 1f;
        public float top_p { get; set; } = 1f;
        public bool stream { get; set; } = false;
        public long? seed { get; set; }
    }

    public record ChatCompletionChoice
    {
        public int index { get; set; } = 0;
        public ChatMessage message { get; set; } = new();
        public string finish_reason { get; set; } = "stop";
    }

    public record ChatCompletionUsage
    {
        public int prompt_tokens { get; set; } = 0;
        public int completion_tokens { get; set; } = 0;
        public int total_tokens { get; set; } = 0;
    }

    public record ChatCompletionResult
    {
        public string id { get; set; } = "";
        public string @object { get; set; } = "chat.completion";
        public long created { get; set; } = 0;
        public string model { get; set; } = "";
        public List<ChatCompletionChoice> choices { get; set; } = new();
        public ChatCompletionUsage usage { get; set; } = new();
    }

    public record ChatCompletionChunkDelta
    {
        public string? role { get; set; }
        public string? content { get; set; }
    }

    public record ChatCompletionChunkChoice
    {
        public int index { get; set; } = 0;
        public ChatCompletionChunkDelta delta { get; set; } = new();
        public string? finish_reason { get; set; }
    }

    public record ChatCompletionChunk
    {
        public string id { get; set; } = "";
        public string @object { get; set; } = "chat.completion.chunk";
        public long created { get; set; } = 0;
        public string model { get; set; } = "";
        public List<ChatCompletionChunkChoice> choices { get; set; } = new();
    }

    [Authorize]
    [HttpPost("chat/completions")]
    [ProducesResponseType(typeof(ChatCompletionResult), 200)]
    [ProducesResponseType(typeof(ChatCompletionChunk), 206)]
    public async Task<IActionResult> Completions(ChatCompletionRequest body, CancellationToken cancellationToken)
    {
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var id = $"chatcmpl-{created}";

        var config = new GenerationConfig()
        {
            top_p = body.top_p,
            temperature = body.temperature,
            max_tokens = body.max_tokens,
            stop_sequences = body.stop is null ? null : new List<string>() { body.stop },
            frequency_penalty = body.frequency_penalty,
            presence_penalty = body.presence_penalty,
            seed = body.seed,
            cancellation = cancellationToken,
        };

        var llm = llmService.FindGenerativeLM(body.model);
        if (llm is null)
            return NotFound(new { message = "Model not found" });

        var messages = body.messages.Select(m =>
            new LLM.Pretrained.ChatMessage() { role = m.role, content = m.content }).ToList();
        var stream = llm.chat_async(messages, config);

        var prompt_tokens = 0;
        async Task<ChatResponseDelta> collect()
        {
            var delta = new ChatResponseDelta();
            var response = new StringBuilder();
            await foreach (var output in stream)
            {
                if (output.content == "" && output.tokens > 0)
                {
                    prompt_tokens += output.tokens;
                    continue;
                }
                response.Append(output.content);
                delta.tokens += output.tokens;
                delta.finish_reason = output.finish_reason;
            }
            delta.content = response.ToString();
            return delta;
        }

        async IAsyncEnumerable<ServerSentEvent> sse_stream()
        {
            var jsonOptions = new JsonSerializerOptions()
            {
                WriteIndented = false,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
                Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            };

            var initChunk = new ChatCompletionChunk()
            {
                id = id,
                created = created,
                model = body.model,
                choices = new List<ChatCompletionChunkChoice>() {
                    new ChatCompletionChunkChoice() {
                        delta = new ChatCompletionChunkDelta() {
                            role = "assistant",
                        },
                    }
                },
            };
            yield return new ServerSentEvent() { Data = JsonSerializer.Serialize(initChunk, jsonOptions) };
            await foreach (var delta in stream)
            {
                var chunk = new ChatCompletionChunk()
                {
                    id = id,
                    created = created,
                    model = body.model,
                    choices = new List<ChatCompletionChunkChoice>() {
                        new ChatCompletionChunkChoice() {
                            delta = new ChatCompletionChunkDelta() {
                                content = delta.content,
                            },
                            finish_reason = delta.finish_reason,
                        }
                    },
                };
                yield return new ServerSentEvent() { Data = JsonSerializer.Serialize(chunk, jsonOptions) };
            }
            yield return new ServerSentEvent() { Data = "[DONE]" };
        }

        if (!body.stream)
        {
            var delta = await collect();
            return Ok(new ChatCompletionResult()
            {
                id = id,
                created = created,
                model = body.model,
                choices = new List<ChatCompletionChoice>() {
                    new ChatCompletionChoice() {
                        message = new ChatMessage() { content = delta.content, role = "assistant" },
                        finish_reason = delta.finish_reason ?? "stop",
                    }
                },
                usage = new ChatCompletionUsage()
                {
                    prompt_tokens = prompt_tokens,
                    completion_tokens = delta.tokens,
                    total_tokens = prompt_tokens + delta.tokens,
                }
            });
        }

        return this.ServerSentEvents(sse_stream());
    }
}

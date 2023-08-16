using System.Text;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Infrastructure;

namespace llm_sharp.Controllers;

public record ServerSentEvent
{
    public string? Id { get; set; }
    public string? Data { get; set; }
    public string? Event { get; set; }
    public uint? Retry { get; set; }

    protected static string[] lineSplitters = new[] { "\r\n", "\r", "\n" };
    protected static char[] invalidNameChars = new[] { '\r', '\n', ':' };

    protected virtual void ThrowIfInvalid()
    {
        if (Event is not null && Event.IndexOfAny(invalidNameChars) >= 0)
            throw new ArgumentException(nameof(Event));

        if (Id is not null && Id.IndexOfAny(invalidNameChars) >= 0)
            throw new ArgumentException(nameof(Id));
    }

    public async Task WriteToStreamAsync(Stream stream)
    {
        ThrowIfInvalid();
        var writer = new StreamWriter(stream, Encoding.UTF8)
        {
            NewLine = "\n"
        };

        if (Id is not null)
            await writer.WriteLineAsync($"id: {Id}");

        if (Event is not null)
            await writer.WriteLineAsync($"event: {Event}");

        if (Retry is not null)
            await writer.WriteLineAsync($"retry: {Retry}");

        if (Data is not null)
        {
            var lines = Data.Split(lineSplitters, StringSplitOptions.None);
            foreach (var line in lines)
                await writer.WriteLineAsync($"data: {line}");
        }
        // extra eol to mark end of event
        await writer.WriteLineAsync();
        await writer.FlushAsync();
    }
}

public class ServerSentEventsResult : ActionResult, IStatusCodeActionResult, IActionResult
{
    public ServerSentEventsResult(IAsyncEnumerable<ServerSentEvent> eventStream, int? statusCode = null)
    {
        ArgumentNullException.ThrowIfNull(eventStream, nameof(eventStream));

        StatusCode = statusCode;
        EventStream = eventStream;
    }

    public int? StatusCode { get; init; }
    public IAsyncEnumerable<ServerSentEvent> EventStream { get; init; }

    public override async Task ExecuteResultAsync(ActionContext context)
    {
        // set content-type
        context.HttpContext.Response.ContentType = "text/event-stream; charset=utf-8";

        if (StatusCode.HasValue)
            context.HttpContext.Response.StatusCode = StatusCode.Value;

        await foreach (var sse in EventStream)
        {
            await sse.WriteToStreamAsync(context.HttpContext.Response.Body);
        }
    }
}

public static class ControllerExtensios
{
    public static ServerSentEventsResult ServerSentEvents(
        this ControllerBase controller,
        IAsyncEnumerable<ServerSentEvent> eventStream,
        int? statusCode = null)
    {
        return new ServerSentEventsResult(eventStream, statusCode);
    }
}

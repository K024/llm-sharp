using llm_sharp.Services;
using llm_sharp.LLM.Utils;
using TupleAsJsonArray;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers().AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.Converters.Add(new TupleConverterFactory());
    });
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddCors(cors => cors.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin().AllowAnyMethod();
    }));

builder.Services.AddRouting(options => options.LowercaseUrls = true);
builder.Services.AddLLMService(builder.Configuration.GetSection("llm"));

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

app.UseAuthorization();

app.MapControllers();

var command =
    app.Configuration.GetValue("command",
        app.Configuration.GetValue<string>("c"));

if (string.IsNullOrEmpty(command))
{
    app.Run();
}
else if (command == "cli")
{
    var llmService = app.Services.GetRequiredService<LLMService>();
    var llm = llmService.FindModel(null) ?? throw new Exception("No models defined");
    llm.start_chat_cli();
}
else if (command == "test")
{
    CliExtensions.run_torch_test();
}
else
{
    throw new Exception($"Unknown command {command}");
}
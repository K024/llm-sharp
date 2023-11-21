using TorchSharp;
using TupleAsJsonArray;
using llm_sharp.Services;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Pretrained;

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

app.UseRouting();

app.UseCors();
app.UseAuthorization();

app.MapControllers();

var command =
    app.Configuration.GetValue("command",
        app.Configuration.GetValue<string>("c"));

if (command == "download")
{
    LibTorchLoader.DownloadLibTorch(
        app.Configuration.GetValue<bool>("removeLast"),
        app.Configuration.GetValue<bool>("skipVerification"),
        app.Configuration.GetValue<string>("url")
    );
    return;
}

LibTorchLoader.EnsureLoaded();

if (string.IsNullOrEmpty(command))
{
    app.Run();
}
else if (command == "cli")
{
    var model = app.Configuration.GetValue<string>("model");
    var model_path = app.Configuration.GetValue<string>("path");
    var dtype = app.Configuration.GetValue("dtype", "float16");
    var device = app.Configuration.GetValue("device", "cuda");

    LanguageModel llm;
    if (string.IsNullOrEmpty(model) || string.IsNullOrEmpty(model_path))
    {
        var llmService = app.Services.GetRequiredService<LLMService>();
        llm = llmService.FindModel(null)
            ?? throw new Exception("No model found. Use --model <model_type> and --path <model_path> to specify a model.");
    }
    else
    {
        llm = LanguageModel.from_pretrained(
            model,
            model_path,
            Enum.Parse<torch.ScalarType>(dtype, true),
            torch.device(device)
        );
    }
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

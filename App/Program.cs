using TorchSharp;
using TupleAsJsonArray;
using llm_sharp.Services;
using llm_sharp.LLM.Utils;
using llm_sharp.LLM.Pretrained;
using System.Text.RegularExpressions;

var builder = WebApplication.CreateBuilder(args);
var config = builder.Configuration;

builder.Services.AddAuthentication(BearerAuthentication.SchemeName)
    .AddBearerAuthentication();
builder.Services.Configure<BearerAuthenticationOptions>(config.GetSection("Bearer"));
builder.Services.AddAuthorization();
builder.Services.AddControllers().AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.Converters.Add(new TupleConverterFactory());
        options.JsonSerializerOptions.DefaultIgnoreCondition =
            System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
    });
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddCors(cors => cors.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin().AllowAnyMethod();
    }));

builder.Services.AddRouting(options => options.LowercaseUrls = true);
builder.Services.AddLLMService(config.GetSection("llm"));

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

app.UseRouting();

app.UseCors();
app.UseAuthentication();
app.UseAuthorization();

app.Use(async (ctx, next) =>
{
    await next();
    if (ctx.Request.Path == "/" && ctx.Response.StatusCode == 404 &&!ctx.Response.HasStarted)
        ctx.Response.Redirect("/swagger/");
});

app.MapControllers();

if (new[] { "help", "h" }.SelectMany(h => new[] { "", "--", "-", "/" }.Select(p => p + h)).Any(x => args.Contains(x)))
{
    var modelTypes = typeof(LanguageModel).Assembly.GetTypes()
        .Where(type => type.IsClass && !type.IsAbstract && type.IsSubclassOf(typeof(LanguageModel)))
        .Select(type => type.Name).ToList();

    var torchVersion = LibTorchDownloader.humanVersion;

    Console.WriteLine(@$"llm-sharp: run and serve LLMs in C# https://github.com/K024/llm-sharp

Usage: llm-sharp [/option|--option value] ...
    Start a server serving all models defined in appsettings.json >> llm > models.
    All arguments can be set through cli args / config file / env variables.
    For more details, see https://learn.microsoft.com/en-us/aspnet/core/fundamentals/configuration/
    Required libtorch: {torchVersion}

Special commands:
    llm-sharp [/c|/command] <command>

Available commands:
    download [/removeLast false] [/yes false] [/url url_for_pytorch_wheel]
        Download torch wheel to the `~/.cache/llm-sharp/` directory for libtorch loading.

    cli [/model <model_type>] [/path <model_path>] [/dtype <float16|float32|float64>] [/device <cpu|cuda:device_idx>]
        Run a cli chatbot. If no model is specified, will load the first model in appsettings.json >> llm > models.

    test
        Run libtorch tests. Will hint the location where libtorch loaded.

Available model types:
    {string.Join(", ", modelTypes)}
");
    return;
}

var command = config.GetValue("command", config.GetValue<string>("c"));

if (string.IsNullOrWhiteSpace(command) && args.Length > 0)
{
    if (Regex.IsMatch(args[0], @"^[a-zA-Z]+$"))
        command = args[0];
}

if (command == "download")
{
    LibTorchLoader.DownloadLibTorch(
        config.GetValue<bool>("removeLast"),
        config.GetValue<bool>("yes"),
        config.GetValue<string>("url")
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
    var model = config.GetValue<string>("model");
    var model_path = config.GetValue<string>("path");
    var dtype = config.GetValue("dtype", "float16");
    var device = config.GetValue("device", "cuda");

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

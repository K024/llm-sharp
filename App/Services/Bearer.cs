using System.Security.Claims;
using System.Text.Encodings.Web;
using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Options;

namespace llm_sharp.Services;

public class BearerAuthenticationOptions : AuthenticationSchemeOptions
{
    public string Token { get; set; } = "";
    public List<string> Tokens { get; set; } = new();
}

public class BearerAuthentication : AuthenticationHandler<BearerAuthenticationOptions>
{
    public const string SchemeName = "Bearer";

    private List<string> _tokens = new();

    private static bool logged = false;

    public BearerAuthentication(
        IOptionsMonitor<BearerAuthenticationOptions> options,
        ILoggerFactory logger, UrlEncoder encoder, ISystemClock clock
    )
        : base(options, logger, encoder, clock)
    {
        _tokens = new List<string>() { options.CurrentValue.Token }
            .Concat(options.CurrentValue.Tokens).Where(token => !string.IsNullOrWhiteSpace(token)).ToList();
        LogInfo();
    }

    protected void LogInfo()
    {
        if (logged) return;
        logged = true;
        if (_tokens.Count > 0)
            Logger.LogInformation($"{_tokens.Count} tokens configured, enabling authentication");
        else
            Logger.LogInformation($"No token configured, allowing all requests");
    }

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        var claims = new[] { new Claim(ClaimTypes.Name, SchemeName) };
        var identity = new ClaimsIdentity(claims, nameof(BearerAuthentication));
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, Scheme.Name);

        // if no tokens are configured, allow all requests
        if (_tokens.Count == 0)
            return Task.FromResult(AuthenticateResult.Success(ticket));

        if (!Request.Headers.TryGetValue("Authorization", out var authorization))
            return Task.FromResult(AuthenticateResult.NoResult());

        if (!authorization.Any(x => x.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)))
            return Task.FromResult(AuthenticateResult.NoResult());

        var token = authorization.First().Substring("Bearer ".Length).Trim();
        if (!_tokens.Contains(token))
            return Task.FromResult(AuthenticateResult.Fail("Invalid token"));

        return Task.FromResult(AuthenticateResult.Success(ticket));
    }
}

public static class BearerAuthenticationExtensions
{
    public static AuthenticationBuilder AddBearerAuthentication(this AuthenticationBuilder builder, Action<BearerAuthenticationOptions>? configureOptions = null)
    {
        return builder.AddScheme<BearerAuthenticationOptions, BearerAuthentication>(BearerAuthentication.SchemeName, configureOptions);
    }
}

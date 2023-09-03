using System.Text;
using Microsoft.AspNetCore.Routing.Template;
using TorchSharp;

namespace llm_sharp.LLM.Utils;

using Tensor = torch.Tensor;

public class StateDictConverter
{
    public class TemplateNameConverter
    {
        protected IReadOnlyDictionary<string, string> converts;
        protected Dictionary<string, RouteTemplate> sourceTemplates;
        protected Dictionary<string, RouteTemplate> targetTemplates;
        protected Dictionary<string, TemplateMatcher> sourceMatchers;

        public TemplateNameConverter(IReadOnlyDictionary<string, string> converts)
        {
            this.converts = converts;

            sourceTemplates = converts.ToDictionary(
                x => x.Key, x => TemplateParser.Parse("/" + x.Key));

            targetTemplates = converts.ToDictionary(
                x => x.Key, x => TemplateParser.Parse("/" + x.Value));

            sourceMatchers = sourceTemplates.ToDictionary(
                x => x.Key, x => new TemplateMatcher(x.Value, new()));
        }

        protected static string BindTemplateValues(RouteTemplate template, RouteValueDictionary values)
        {
            var result = new StringBuilder();
            foreach (var (index, segment) in template.Segments.Select((x, i) => (i, x)))
            {
                result.Append("/");
                foreach (var part in segment.Parts)
                {
                    if (part.IsLiteral)
                        result.Append(part.Text);
                    else if (part.IsParameter)
                        result.Append(values[part.Name!]);
                    else
                        throw new Exception($"Unsupported template '{template.TemplateText}'");
                }
            }
            return result.ToString();
        }

        public bool TryConvert(string name, out string converted)
        {
            return TryConvert(name, out converted, out _, out _);
        }

        public bool TryConvert(string name, out string converted, out string fromTemplate, out string toTemplate)
        {
            // append "/" to make a valid path
            var path = new PathString("/" + name);
            var values = new RouteValueDictionary();
            foreach (var pair in converts)
            {
                if (sourceMatchers[pair.Key].TryMatch(path, values))
                {
                    var targetTemplate = targetTemplates[pair.Key];
                    var bound = BindTemplateValues(targetTemplate, values);
                    // remove prepended "/"
                    converted = bound[1..];
                    fromTemplate = pair.Key;
                    toTemplate = pair.Value;
                    return true;
                }
            }
            converted = default!;
            fromTemplate = default!;
            toTemplate = default!;
            return false;
        }
    }

    public record MergeRule
    {
        public string target { get; set; } = "";
        public List<string> from { get; set; } = new();
        public int dim { get; set; }
    }

    public record ConvertRules
    {
        public List<string> keeps { get; set; } = new();
        public Dictionary<string, string> renames { get; set; } = new();
        public List<MergeRule> merges { get; set; } = new();
    }

    protected ConvertRules rules;
    protected TemplateNameConverter renameConverter;
    protected TemplateNameConverter mergeConverter;
    protected Dictionary<string, MergeRule> mergeRules;

    public StateDictConverter(ConvertRules rules)
    {
        this.rules = rules;
        renameConverter = new TemplateNameConverter(rules.renames);

        mergeRules = rules.merges.ToDictionary(x => x.target, x => x);
        mergeConverter = new TemplateNameConverter(
            rules.merges.SelectMany(x => x.from.Select(from => (from, x.target)))
                .ToDictionary(x => x.from, x => x.target)
        );
    }

    protected bool TryFindMerge(string name, out string converted, out MergeRule mergeRule, out int index)
    {
        if (mergeConverter.TryConvert(name, out converted, out var fromTemplate, out var toTemplate))
        {
            mergeRule = mergeRules[toTemplate];
            index = mergeRule.from.IndexOf(fromTemplate);
            if (index < 0)
                throw new Exception("Unexpected index");
            return true;
        }
        mergeRule = default!;
        index = -1;
        return false;
    }

    public IEnumerable<(string, Tensor)> Convert(IEnumerable<(string, Tensor)> tensors)
    {
        var toMerge = new Dictionary<string, (string, Tensor?)[]>();

        foreach (var (name, tensor) in tensors)
        {
            if (rules.keeps.Contains(name))
            {
                yield return (name, tensor);
            }
            else if (renameConverter.TryConvert(name, out var converted))
            {
                yield return (converted, tensor);
            }
            else if (TryFindMerge(name, out converted, out var mergeRule, out var index))
            {
                var list = toMerge.GetValueOrDefault(converted)
                    ?? new (string, Tensor?)[mergeRule.from.Count];

                if (list[index].Item2 is not null)
                    throw new Exception($"Duplicated merge for key {name}");

                list[index] = (name, tensor);
                toMerge[converted] = list;

                if (list.All(x => x.Item2 is not null))
                {
                    var merged = torch.cat(list.Select(x => x.Item2).ToArray()!, mergeRule.dim);
                    foreach (var split in list)
                        split.Item2?.Dispose();
                    toMerge[converted] = new (string, Tensor?)[0];
                    yield return (converted, merged);
                }
            }
            else
            {
                Console.WriteLine($"Unconverted state dict key '{name}'");
            }
        }

        foreach (var pair in toMerge)
        {
            if (pair.Value.Length > 0)
            {
                Console.WriteLine($"Partially merged tensor '{pair.Key}'");
                foreach (var (_, tensor) in pair.Value)
                    tensor?.Dispose();
            }
        }
    }
}

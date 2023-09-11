using System.Reflection;
using System.Text.Json;
using TorchSharp;

namespace llm_sharp.LLM.Train;

using nn = torch.nn;
using optim = torch.optim;
using Tensor = torch.Tensor;


public record TrainerConfig
{
    public string trainer { get; set; } = "";
    public string model_path { get; set; } = "";
    public string data_path { get; set; } = "";
    public string eval_data_path { get; set; } = "";
    public string output_dir { get; set; } = "";

    public bool use_bfloat16 { get; set; } = true;

    public int batch_size { get; set; } = 32;
    public int max_epochs { get; set; } = 10;
    public int max_steps { get; set; } = 3000;
    public int warmup_steps { get; set; } = 100;
    public int accumulation_steps { get; set; } = 1;
    public int save_steps { get; set; } = 1000;
    public double learning_rate { get; set; } = 5e-5;
}

public abstract class Trainer
{
    public const string LOG_FILE = "train.log";

    public static Trainer from_config(TrainerConfig config, Assembly assembly)
    {
        var types = (assembly ?? typeof(Utils.LLM).Assembly).GetTypes();

        var type = types.Where(x =>
            x.Name == config.trainer
            && x.IsClass
            && !x.IsAbstract
            && x.IsSubclassOf(typeof(Trainer))
        )
            .FirstOrDefault()
            ?? throw new Exception($"Unable to find class '{config.trainer}'. Try give the correct Assembly");

        var createTrainer = type.GetConstructor(new[] { typeof(TrainerConfig) })
            ?? throw new Exception($"The Trainer should have a public constructor with TrainerConfig");

        var result = createTrainer.Invoke(new object?[] { config });
        return (Trainer)result;
    }

    public record TrainerLog
    {
        public int step { get; set; }
        public double epoch { get; set; }
        public double progress { get; set; }
        public double loss { get; set; }
        public double lr { get; set; }
        public long timestamp { get; set; } // miliseconds
    }

    public abstract TrainerConfig config { get; }

    public abstract nn.Module model { get; }

    public abstract IEnumerable<TorchSharp.Modules.Parameter> get_trainable_parameters();

    public virtual (optim.Optimizer, optim.lr_scheduler.LRScheduler) create_optimizer(int num_training_steps)
    {
        var optimizer = optim.AdamW(get_trainable_parameters(), config.learning_rate);
        var scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda);

        double lr_lambda(int step)
        {
            if (step < config.warmup_steps)
                return (double)step / config.warmup_steps;
            return Math.Max(0.0, (num_training_steps - step) / (double)Math.Max(1, num_training_steps - config.warmup_steps));
        }
        return (optimizer, scheduler);
    }

    public abstract int compute_batches();

    public virtual (int steps, int step_per_epoch) compute_steps()
    {
        var batches = compute_batches();
        batches = batches / config.accumulation_steps;
        var steps = Math.Min(batches * config.max_epochs, config.max_steps);
        return (steps, batches);
    }

    public virtual void log_step(int step, TrainerLog log)
    {
        var log_str = JsonSerializer.Serialize(log);
        Console.WriteLine(log_str);
        File.AppendAllText(Path.Combine(config.output_dir, LOG_FILE), log_str + "\n");
    }

    public abstract IEnumerable<Tensor[]> create_batches();

    public abstract Tensor compute_loss(params Tensor[] inputs);

    public abstract void save_checkpoint(int step, bool finished);

    public virtual void train()
    {
        Directory.CreateDirectory(config.output_dir);

        var (num_training_steps, num_steps_per_epoch) = compute_steps();
        var (optimizer, scheduler) = create_optimizer(num_training_steps);

        var step = 0;
        var epoch = 0.0;
        while (step < num_training_steps)
        {
            var acc_loss = 0.0;
            var acc_step = 0;
            foreach (var batch in create_batches())
            {
                using var loss = compute_loss(batch);
                acc_loss += loss.item<double>();
                acc_step += 1;

                loss.backward();

                if (acc_step % config.accumulation_steps == 0)
                {
                    optimizer.step();
                    scheduler.step();
                    optimizer.zero_grad();

                    var log = new TrainerLog
                    {
                        step = step,
                        epoch = epoch,
                        progress = step / (double)num_training_steps,
                        loss = acc_loss / acc_step,
                        lr = scheduler.get_last_lr().FirstOrDefault(0.0),
                        timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds(),
                    };
                    log_step(step, log);

                    step += 1;
                    epoch += 1.0 / num_steps_per_epoch;
                    acc_loss = 0;
                    acc_step = 0;

                    if (step >= num_training_steps)
                        break;

                    if (step % config.save_steps == 0)
                        save_checkpoint(step, false);
                }
            }
        }
        save_checkpoint(step, true);
    }
}

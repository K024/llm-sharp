import json
import fire
import torch
from pathlib import Path
from convert_utils import pack_u4, unpack_u4
from safetensors.torch import load_file, save_file


def convert_config(input: Path, output: Path):
    config_json = input / "config.json"
    assert config_json.exists()

    config = json.loads(config_json.read_text())
    assert "QWenLMHeadModel" in config["architectures"]

    new_config = dict(
        hidden_size=config["hidden_size"],
        inner_hidden_size=config["intermediate_size"] // 2,
        head_hidden_size=config["kv_channels"],
        hidden_act="silu",
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_attention_heads"],
        num_layers=config["num_hidden_layers"],
        qkv_bias=True,
        o_bias=False,
        vocab_size=config["vocab_size"],
        dropout_rate=0.0,
        layernorm_epsilon=1e-6,
        max_sequence_length=2048,
    )

    model_config_json = output / "model_config.json"
    model_config_json.write_text(json.dumps(new_config, indent=2))


def convert_tokenizer(input: Path, output: Path):
    tiktoken_file = input / "qwen.tiktoken"
    assert tiktoken_file.exists()

    lines = tiktoken_file.read_text().splitlines()
    pairs = [line.split(" ") for line in lines if line]
    ranks = { k.strip(): int(v) for k, v in pairs }

    eos_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    special_tokens = eos_tokens + [
        f"<|extra_{i}|>" for i in range(205)
    ]

    tokenizer_config = dict(
        eos_tokens=eos_tokens,
        special_tokens={
            k: v + len(ranks) for v, k in enumerate(special_tokens)
        },
        ranks=ranks,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    )

    tokenizer_config_json = output / "tokenizer_config.json"
    tokenizer_config_json.write_text(json.dumps(tokenizer_config, indent=2))


def convert_weights(input: Path, output: Path, weight_is_awq: bool):
    name_mapping = {
        'transformer.wte.weight': 'word_embedding.weight',
        'transformer.ln_f.weight': 'final_ln.weight',
        'lm_head.weight': 'lm_head.weight'
    }

    for i in range(128):
        name_mapping.update({
            f'transformer.h.{i}.ln_1.weight': f'layers.{i}.attn_ln.weight',
            f'transformer.h.{i}.attn.c_attn.bias': f'layers.{i}.attn.qkv_proj.bias',
            f'transformer.h.{i}.ln_2.weight': f'layers.{i}.ffn_ln.weight',
        })

        for suffix in ["weight", "qweight", "qzeros", "scales"]:
            name_mapping.update({
                f'transformer.h.{i}.attn.c_attn.{suffix}': f'layers.{i}.attn.qkv_proj.{suffix}',
                f'transformer.h.{i}.attn.c_proj.{suffix}': f'layers.{i}.attn.o_proj.{suffix}',
                f'transformer.h.{i}.mlp.w1.{suffix}': f'layers.{i}.ffn.w_in.{suffix}',
                f'transformer.h.{i}.mlp.w2.{suffix}': f'layers.{i}.ffn.w_gate.{suffix}',
                f'transformer.h.{i}.mlp.c_proj.{suffix}': f'layers.{i}.ffn.w_out.{suffix}',
            })

    weight_files = input.glob("*.safetensors")
    pack_order = [0, 2, 4, 6, 1, 3, 5, 7]

    for weight_file in weight_files:
        weights = load_file(weight_file, device="cuda" if torch.cuda.is_available() else "cpu")
        converted = {}

        for key, value in weights.items():
            if not weight_is_awq:
                if key.endswith(".qweight"):
                    # [in_dim // 8, out_dim] => [in_dim, out_dim // 8]
                    value = pack_u4(unpack_u4(value.T).T, pack_order)

                elif key.endswith(".qzeros"):
                    # [in_dim // group_size, out_dim // 8] same shape but plus 1
                    value = pack_u4(unpack_u4(value) + 1, pack_order)

                elif key.endswith(".g_idx"):
                    # ignore g_idx in gptq
                    continue

            if key.endswith(".bias") and key not in name_mapping:
                if not torch.allclose(value, torch.zeros_like(value), rtol=0.001):
                    raise RuntimeError(f"Non-zero bias in {key}.")

            converted[name_mapping[key]] = value

        save_file(converted, output / weight_file.name)


def convert(input, output, weight_is_awq=False):
    input = Path(input)
    output = Path(output)

    assert input.exists()
    if output.exists():
        print(f"Output path {output} already exists.")

    output.mkdir(parents=True, exist_ok=True)

    convert_config(input, output)
    convert_tokenizer(input, output)
    convert_weights(input, output, weight_is_awq)


if __name__ == "__main__":
    fire.Fire(convert)

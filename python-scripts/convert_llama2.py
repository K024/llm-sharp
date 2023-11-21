import json
import fire
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


def convert_config(input: Path, output: Path):
    config_json = input / "config.json"
    assert config_json.exists()

    config = json.loads(config_json.read_text())
    assert "LlamaForCausalLM" in config["architectures"]
    
    quantize_config_json = input / "quantize_config.json"
    assert not quantize_config_json.exists(), "GPTQ quantization is not supported."

    new_config = dict(
        hidden_size=config["hidden_size"],
        inner_hidden_size=config["intermediate_size"],
        head_hidden_size=config["hidden_size"] // config["num_attention_heads"],
        hidden_act="silu",
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        num_layers=config["num_hidden_layers"],
        qkv_bias=False,
        o_bias=False,
        vocab_size=config["vocab_size"],
        dropout_rate=0.0,
        layernorm_epsilon=1e-6,
        max_sequence_length=config["max_position_embeddings"],
    )

    model_config_json = output / "model_config.json"
    model_config_json.write_text(json.dumps(new_config, indent=2))


def convert_tokenizer(input: Path, output: Path):
    tokenizer_json = input / "tokenizer.json"
    assert tokenizer_json.exists()

    tokenizer = json.loads(tokenizer_json.read_bytes())

    vocab = tokenizer["model"]["vocab"]
    special_tokens = ["<unk>", "<s>", "</s>"]
    special_tokens = { token: vocab[token] for token in special_tokens }
    eos_tokens = ["</s>"]
    merges = tokenizer["model"]["merges"]

    tokenizer_config = dict(
        eos_tokens=eos_tokens,
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        special_tokens=special_tokens,
        vocab=vocab,
        merges=merges,
    )

    tokenizer_config_json = output / "tokenizer_config.json"
    tokenizer_config_json.write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def convert_weights(input: Path, output: Path):
    name_mapping = {
        'model.embed_tokens.weight': 'word_embedding.weight',
        'model.norm.weight': 'final_ln.weight',
        'lm_head.weight': 'lm_head.weight'
    }
    name_merges = {}
    merge_config = {}

    for i in range(128):
        name_mapping.update({
            f'model.layers.{i}.input_layernorm.weight': f'layers.{i}.attn_ln.weight',
            f'model.layers.{i}.post_attention_layernorm.weight': f'layers.{i}.ffn_ln.weight',
        })

        for suffix in ["weight", "qweight", "qzeros", "scales"]:
            name_mapping.update({
                f'model.layers.{i}.self_attn.o_proj.{suffix}': f'layers.{i}.attn.o_proj.{suffix}',
                f'model.layers.{i}.mlp.up_proj.{suffix}': f'layers.{i}.ffn.w_in.{suffix}',
                f'model.layers.{i}.mlp.gate_proj.{suffix}': f'layers.{i}.ffn.w_gate.{suffix}',
                f'model.layers.{i}.mlp.down_proj.{suffix}': f'layers.{i}.ffn.w_out.{suffix}',
            })
            name_merges.update({
                f'model.layers.{i}.self_attn.q_proj.{suffix}': (f'layers.{i}.attn.qkv_proj.{suffix}', 0),
                f'model.layers.{i}.self_attn.k_proj.{suffix}': (f'layers.{i}.attn.qkv_proj.{suffix}', 1),
                f'model.layers.{i}.self_attn.v_proj.{suffix}': (f'layers.{i}.attn.qkv_proj.{suffix}', 2),
            })
            merge_config.update({
                f'layers.{i}.attn.qkv_proj.{suffix}': dict(size=3, dim=1)
            })

    weight_files = list(input.glob("*.safetensors"))
    is_safetensors = True
    if len(weight_files) == 0:
        weight_files = list(input.glob("*.bin"))
        is_safetensors = False
    assert len(weight_files) > 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    merges = {}

    for weight_file in weight_files:
        if is_safetensors:
            weights = load_file(weight_file, device=device)
        else:
            weights = torch.load(weight_file, map_location=device)
        converted = {}

        for key, value in weights.items():

            if key in name_merges:
                new_key, dim = name_merges[key]
                if new_key not in merges:
                    merges[new_key] = [None] * merge_config[new_key]["size"]
                merges[new_key][dim] = value
                if None not in merges[new_key]:
                    converted[new_key] = torch.cat(merges[new_key], dim=merge_config[new_key]["dim"])
                    del merges[new_key]
                continue

            converted[name_mapping[key]] = value

        if is_safetensors:
            file_name = weight_file.name
        else:
            file_name = weight_file.stem + ".safetensors"
        save_file(converted, output / file_name)

    if len(merges) > 0:
        print("Warning: some weights missing for merges:")
        for key in merges:
            print(f"  {key}")


def convert(input, output):
    input = Path(input)
    output = Path(output)

    assert input.exists()
    if output.exists():
        print(f"Output path {output} already exists.")

    output.mkdir(parents=True, exist_ok=True)

    convert_config(input, output)
    convert_tokenizer(input, output)
    convert_weights(input, output)

    print(f"Converted {input} to {output}.")


if __name__ == "__main__":
    fire.Fire(convert)

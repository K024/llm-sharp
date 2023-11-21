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
    assert "BaichuanForCausalLM" in config["architectures"]

    quantize_config_json = input / "quantize_config.json"
    if quantize_config_json.exists():
        quantize_config = json.loads(quantize_config_json.read_text())
        assert quantize_config["bits"] == 4
        assert quantize_config["desc_act"] == False

    new_config = dict(
        hidden_size=config["hidden_size"],
        inner_hidden_size=config["intermediate_size"],
        head_hidden_size=config["hidden_size"] // config["num_attention_heads"],
        hidden_act="silu",
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_attention_heads"],
        num_layers=config["num_hidden_layers"],
        qkv_bias=False,
        o_bias=False,
        vocab_size=config["vocab_size"],
        dropout_rate=0.0,
        layernorm_epsilon=1e-6,
        max_sequence_length=4096,
        use_alibi=config["hidden_size"] == 5120,
    )

    model_config_json = output / "model_config.json"
    model_config_json.write_text(json.dumps(new_config, indent=2))


def convert_tokenizer(input: Path, output: Path):
    tokenizer_json = input / "tokenizer.json"
    assert tokenizer_json.exists(), f"""
    Please run the following command to generate the tokenizer.json file:

    python -m transformers.convert_slow_tokenizers_checkpoints_to_fast \\
        --tokenizer_name LlamaTokenizer \\
        --checkpoint_name {input} \\
        --dump_path .

    After this, you will get the converted tokenizer. Rename it to tokenizer.json and place it in the model folder.
    """

    tokenizer = json.loads(tokenizer_json.read_text())

    vocab = tokenizer["model"]["vocab"]
    special_token_end = vocab["<reserved_999>"]
    special_tokens = { token: id for token, id in vocab.items() if id <= special_token_end }
    eos_tokens = ["</s>"]
    merges = tokenizer["model"]["merges"]

    tokenizer_config = dict(
        eos_tokens=eos_tokens,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        special_tokens=special_tokens,
        vocab=vocab,
        merges=merges,
    )

    tokenizer_config_json = output / "tokenizer_config.json"
    tokenizer_config_json.write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def convert_weights(input: Path, output: Path, weight_is_awq: bool):
    name_mapping = {
        'model.embed_tokens.weight': 'word_embedding.weight',
        'model.norm.weight': 'final_ln.weight',
        'lm_head.weight': 'lm_head.weight'
    }

    for i in range(128):
        name_mapping.update({
            f'model.layers.{i}.input_layernorm.weight': f'layers.{i}.attn_ln.weight',
            f'model.layers.{i}.post_attention_layernorm.weight': f'layers.{i}.ffn_ln.weight',
        })

        for suffix in ["weight", "qweight", "qzeros", "scales"]:
            name_mapping.update({
                f'model.layers.{i}.self_attn.W_pack.{suffix}': f'layers.{i}.attn.qkv_proj.{suffix}',
                f'model.layers.{i}.self_attn.o_proj.{suffix}': f'layers.{i}.attn.o_proj.{suffix}',
                f'model.layers.{i}.mlp.up_proj.{suffix}': f'layers.{i}.ffn.w_in.{suffix}',
                f'model.layers.{i}.mlp.gate_proj.{suffix}': f'layers.{i}.ffn.w_gate.{suffix}',
                f'model.layers.{i}.mlp.down_proj.{suffix}': f'layers.{i}.ffn.w_out.{suffix}',
            })

    weight_files = list(input.glob("*.safetensors"))
    is_safetensors = True
    if len(weight_files) == 0:
        weight_files = list(input.glob("*.bin"))
        is_safetensors = False
    assert len(weight_files) > 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack_order = [0, 2, 4, 6, 1, 3, 5, 7]

    for weight_file in weight_files:
        if is_safetensors:
            weights = load_file(weight_file, device=device)
        else:
            weights = torch.load(weight_file, map_location=device)
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
                continue

            converted[name_mapping[key]] = value

        if is_safetensors:
            file_name = weight_file.name
        else:
            file_name = weight_file.stem + ".safetensors"
        save_file(converted, output / file_name)


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

    print(f"Converted {input} to {output}.")


if __name__ == "__main__":
    fire.Fire(convert)

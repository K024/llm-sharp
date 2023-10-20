import json
import fire
import torch
from pathlib import Path
from safetensors.torch import save_file


def convert_config(input: Path, output: Path):
    config_json = input / "config.json"
    assert config_json.exists()

    config = json.loads(config_json.read_text())

    new_config = dict(
        hidden_size=config["hidden_size"],
        inner_hidden_size=config["intermediate_size"],
        head_hidden_size=config["hidden_size"] // config["num_attention_heads"],
        num_attention_heads=config["num_attention_heads"],
        num_layers=config["num_hidden_layers"],
        vocab_size=config["vocab_size"],
        dropout_rate=0.0,
        layernorm_epsilon=1e-6,
        type_vocab_size=config["type_vocab_size"],
        max_sequence_length=config["max_position_embeddings"],
        pooling_mode=["pooler"],
        classifier_classes=0,
        classifier_mode="sequence",
    )

    model_config_json = output / "model_config.json"
    model_config_json.write_text(json.dumps(new_config, indent=2))


def convert_tokenizer(input: Path, output: Path):
    config_json = input / "tokenizer_config.json"
    assert config_json.exists()

    config = json.loads(config_json.read_text())

    vocab_file = input / "vocab.txt"
    vocab = [l.strip() for l in vocab_file.read_text("utf-8").splitlines() if l]
    vocab = { l: i for i, l in enumerate(vocab) }

    tokenizer_config = dict(
        unk_token="[UNK]",
        continuing_prefix="##",
        basic_tokenize=dict(
            do_lower_case=config["do_lower_case"],
            tokenize_chinese_chars=config["tokenize_chinese_chars"],
            strip_accents=config["strip_accents"],
        ),
        special_tokens={
            "[PAD]": vocab["[PAD]"],
            "[CLS]": vocab["[CLS]"],
            "[SEP]": vocab["[SEP]"],
            "[MASK]": vocab["[MASK]"],
        },
        vocab=vocab,
    )

    tokenizer_config_json = output / "tokenizer_config.json"
    tokenizer_config_json.write_text(json.dumps(tokenizer_config, indent=2, ensure_ascii=False), "utf-8")


def convert_weights(input: Path, output: Path):
    suffixes = ["weight", "bias"]
    name_mapping = {
        'embeddings.word_embeddings.weight': 'embedding.word_embeddings.weight',
        'embeddings.position_embeddings.weight': 'embedding.position_embeddings.weight',
        'embeddings.token_type_embeddings.weight': 'embedding.token_type_embeddings.weight',
    }

    for suffix in suffixes:
        name_mapping.update({
            f'embeddings.LayerNorm.{suffix}': f'embedding.layer_norm.{suffix}',
            f'pooler.dense.{suffix}': f'pooler.dense.{suffix}',
            f'classifier.{suffix}': f'classifier.{suffix}',
        })

    mergers = {}
    for layer_idx in range(32):
        for suffix in suffixes:
            name_mapping.update({
                f'encoder.layer.{layer_idx}.attention.output.dense.{suffix}': f'layers.{layer_idx}.attn.o_proj.{suffix}',
                f'encoder.layer.{layer_idx}.attention.output.LayerNorm.{suffix}': f'layers.{layer_idx}.attn_ln.{suffix}',
                f'encoder.layer.{layer_idx}.intermediate.dense.{suffix}': f'layers.{layer_idx}.ffn.w_in.{suffix}',
                f'encoder.layer.{layer_idx}.output.dense.{suffix}': f'layers.{layer_idx}.ffn.w_out.{suffix}',
                f'encoder.layer.{layer_idx}.output.LayerNorm.{suffix}': f'layers.{layer_idx}.ffn_ln.{suffix}',
            })
            mergers.update({
                f'encoder.layer.{layer_idx}.attention.self.query.{suffix}': (f'layers.{layer_idx}.attn.qkv_proj.{suffix}', 0),
                f'encoder.layer.{layer_idx}.attention.self.key.{suffix}': (f'layers.{layer_idx}.attn.qkv_proj.{suffix}', 1),
                f'encoder.layer.{layer_idx}.attention.self.value.{suffix}': (f'layers.{layer_idx}.attn.qkv_proj.{suffix}', 2),
            })

    state_dict = torch.load(input / "pytorch_model.bin", map_location="cpu")
    to_be_merged = {}
    converted = {}

    for key, value in state_dict.items():
        if key in name_mapping:
            converted[name_mapping[key]] = value
        elif key in mergers:
            merged_name, index = mergers[key]
            to_be_merged.setdefault(merged_name, {})[index] = value
        else:
            print(f"Ignoring key: {key}")

    for key, value in to_be_merged.items():
        # weight: [out, in], bias: [out]
        converted[key] = torch.cat([value[0], value[1], value[2]], dim=0)

    save_file(converted, output / "model_weights.safetensors")


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


if __name__ == "__main__":
    fire.Fire(convert)

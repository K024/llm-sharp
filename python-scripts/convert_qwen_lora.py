import json
import fire
from pathlib import Path
from safetensors.torch import load_file, save_file


def convert(input, output):
    input = Path(input)
    output = Path(output)

    assert input.exists()
    if output.exists():
        print(f"Output path {output} already exists.")
        
    state_dict = load_file(input)

    name_mapping = {}

    for i in range(128):
        for suffix in ["lora_A", "lora_B"]:
            name_mapping.update({
                f'base_model.model.transformer.h.{i}.attn.c_attn.{suffix}.weight': f'layers.{i}.attn.qkv_proj.{suffix}',
                f'base_model.model.transformer.h.{i}.attn.c_proj.{suffix}.weight': f'layers.{i}.attn.o_proj.{suffix}',
                f'base_model.model.transformer.h.{i}.mlp.w1.{suffix}.weight': f'layers.{i}.ffn.w_in.{suffix}',
                f'base_model.model.transformer.h.{i}.mlp.w2.{suffix}.weight': f'layers.{i}.ffn.w_gate.{suffix}',
                f'base_model.model.transformer.h.{i}.mlp.c_proj.{suffix}.weight': f'layers.{i}.ffn.w_out.{suffix}',
            })

    converted = {}
    for key, value in state_dict.items():
        if key in name_mapping:
            converted[name_mapping[key]] = value
        else:
            print(f"Key {key} not found in name mapping.")
            converted[key] = value

    save_file(converted, output)

    print(f"Converted {input} to {output}.")


if __name__ == "__main__":
    fire.Fire(convert)

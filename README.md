# llm-sharp

**🚧 Under very early development**

Run and serve language models in C# with [TorchSharp](https://github.com/dotnet/TorchSharp).

This project aims to write most things natively in C# except for some specialized CUDA kernels where you only have cpp. This offers the best developer experience for production ready apps.


## Features & TODOs

C#:
- Python resources interop
  - [x] Load safetensors in C#
  - [ ] Convert scripts
    - [x] GPTQ & Awq convert scripts
  - [ ] streamlit web ui for api service
- Introduce more models
  - [ ] Llama2 family
    - [ ] Llama2
    - [x] Qwen tested
  - [x] Bert family
    - [x] SentenceTransformer tested
- Introduce more tokenizers
  - [x] BPE
  - [x] BPE (SentencePiece)
  - [x] Tiktoken (BPE)
  - [x] Unigram (SentencePiece)
  - [x] WordPiece
  - [ ] Unit tests
- Serve
  - [x] Basic api with ASP.Core
  - [x] Command line interface
  - [ ] Batched inference

Native cpp:
- Model parallel
  - [ ] Layer parallel
  - [ ] ~~Tensor parallel~~ Needs NCCL support.
- Specialized cuda ops
  - [x] C# cuda op loading
  - [ ] ~~GPTQ int4 ops~~ Can be converted to AWQ format without loss if `des_act` is not used.
  - [x] AWQ int4 ops
  - [ ] Flash attention
  - [x] Fused ops (RMS norm, Rotary embeddings)


## Usage

Modify `appsettings.json` in `App` project or add an environment aware config with:
```json
{
  "llm": {
    "models": [
      {
        "name": "qwen-14b-chat",
        "type": "QwenAwq",
        "path": "path/to/qwen-14b-chat-awq-int4",
        "device": "cuda:0",
        "dtype": "float16"
      }
    ]
  }
}
```

Http api service (visit `http://localhost:5137/swagger/index.html` for api docs):
```
dotnet run --project App
```

Command line interface:
```
dotnet run --project App /c cli
```

## Performance

Single inference on Linux with single RTX 3090 (Qwen-14B-chat, awq int4):

```
Decoder perf:
  len: 670(prefix) + 326(gen)
 init: 0.3439 s
  sum: 6.5723 s
  gen: 49.6019 tok/s
  avg: 47.2804 tok/s
```

## Acknowledgement

- [dotnet/TorchSharp](https://github.com/dotnet/TorchSharp)
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers)
- [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)
- [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

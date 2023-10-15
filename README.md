# llm-sharp

**🚧 Under very early development**

Run and serve language models in C# with [TorchSharp](https://github.com/dotnet/TorchSharp).

TODO list:
- Python resources interop
  - [x] Safetensors
  - [ ] Convert scripts
- Introduce more models
  - [ ] Llama2 family (Qwen tested √)
  - [ ] Bert family (Ernie tested √)
- Introduce more tokenizers
  - [x] BPE
  - [x] BPE (SentencePiece)
  - [x] Tiktoken (BPE)
  - [x] Unigram (SentencePiece)
  - [x] WordPiece
  - [ ] Unit tests
- Model parallel
  - [ ] Layer parallel
  - [ ] ~~Tensor parallel~~ Needs NCCL support.
- Specialized cuda ops
  - [x] C# cuda op loading
  - [ ] GPTQ int4 ops
  - [x] AWQ int4 ops
  - [ ] Flash attention
  - [ ] Fused ops (RMS norm, Rotary embeddings)

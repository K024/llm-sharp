# llm-sharp

**🚧 Under very early development**

Run and serve language models in C# with [TorchSharp](https://github.com/dotnet/TorchSharp).

TODO list:
- Python resources interop
  - [x] Safetensors
  - [ ] Convert scripts
- Introduce more models & tokenizers
  - [ ] LLaMa2 family (Sentencepiece BPE)
  - [x] Qwen 7B (Tiktoken)
  - [ ] Bert family (Wordpiece)
- Model parallel
  - [ ] Layer parallel
  - [ ] ~~Tensor parallel~~ Needs NCCL support.
- Specialized cuda ops
  - [ ] C# cuda op loading
  - [ ] GPTQ int4 ops
  - [ ] Flash attention
  - [ ] Fused ops (RMS norm, Rotary embeddings)

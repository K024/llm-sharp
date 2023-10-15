FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    build-essential git wget curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ \
    transformers datasets safetensors sentencepiece tiktoken \
    ninja fastapi uvicorn[standard] streamlit fire

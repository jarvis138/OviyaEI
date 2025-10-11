FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    faster-whisper>=0.9.0 \
    scipy>=1.10.0 \
    numpy>=1.24.0 \
    onnxruntime>=1.16.1 \
    websockets>=12.0 \
    fastapi>=0.104.1 \
    uvicorn>=0.24.0 \
    pydantic>=2.5.0

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Copy handler
COPY handler.py /workspace/handler.py

# Start the serverless handler
CMD ["python", "-u", "/workspace/handler.py"]



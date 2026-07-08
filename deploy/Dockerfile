FROM python:3.11-slim

# HF Spaces runs the container as user 1000 with only /tmp and $HOME writable.
# triton (torch's CUDA kernel JIT) needs a C compiler and writable cache.
ENV HF_HOME=/tmp/hf \
    TRITON_CACHE_DIR=/tmp/triton \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch transformers accelerate flask numpy

WORKDIR /app
COPY dashboard/ /app/dashboard/
COPY probes/ /app/probes/

EXPOSE 7860
CMD ["python", "dashboard/server.py", \
     "--model", "unsloth/Llama-3.2-3B-Instruct", \
     "--probes", "/app/probes", \
     "--host", "0.0.0.0", "--port", "7860"]

# Start with CUDA runtime base (no PyTorch preinstalled)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /workspace/flux

# Copy application code
COPY . .

RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r torchtitan/experiments/flux/requirements-flux.txt \
    -r requirements-mlperf.txt

RUN pip install -e .


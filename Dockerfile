FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, git, and Python dev headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python and pip3 the default pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --set pip /usr/bin/pip3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./ /app/
COPY assets/ ./assets/
COPY torchtitan/ ./torchtitan/
COPY requirements.txt ./requirements.txt

# Install dependencies from requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install the project itself
RUN python -m pip install --no-cache-dir .

# Copy the rest of the project (scripts, etc.)
COPY . .

# Make the training script executable
RUN chmod +x ./torchtitan/experiments/deepseek_v3/run_training.sh

# Set the default command or entrypoint if needed
# For now, we'll leave it to be specified at runtime 
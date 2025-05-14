# Docker images for TorchTitan CI

This directory contains everything needed to build the Docker images
that are used in TorchTitan CI. The content of this directory are copied
from PyTorch CI https://github.com/pytorch/pytorch/tree/main/.ci/docker.
It also uses the same directory structure as PyTorch.

## Contents

* `build.sh` -- dispatch script to launch all builds
* `common` -- scripts used to execute individual Docker build stages
* `ubuntu` -- Dockerfile for Ubuntu image for CPU build and test jobs

## Usage

```bash
# Generic usage
./build.sh "${IMAGE_NAME}" "${DOCKER_BUILD_PARAMETERS}"

# Build a specific image
./build.sh torchtitan-ubuntu-20.04-clang12 -t myimage:latest
```

### Flux Docker Image

For flux model development, a dedicated Docker image is available that includes the necessary dependencies (transformers, einops, sentencepiece). This image is built automatically when changes are made to the flux code.

To build the flux Docker image manually:

```bash
cd .ci/docker
docker build -f ubuntu/Dockerfile.flux -t torchtitan-flux-ubuntu-20.04-clang12 .
```

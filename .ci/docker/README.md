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
./build.sh torchtitan-ubuntu-22.04-clang12 -t myimage:latest

# Build the CUDA image with RL dependencies (Monarch, TorchStore, and test extras)
./build.sh torchtitan-ubuntu-22.04-clang12:rl -t my-rl-image:latest
```

CI publishes the RL variant to the existing CUDA ECR repository with an
`rl-<docker-hash>` tag so it does not replace the standard CUDA image.
The RL unit and integration test workflows install torch, torchvision, and vLLM
together from the CUDA nightly index. The optional Verifiers example is installed
only by RL unit tests, with MCP 1.x to select a compatible nightly cohort.

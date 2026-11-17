# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# XPU builds on top of pytorch's official XPU CI image, which already ships the
# Intel GPU driver, oneAPI toolchain, conda and the jenkins CI user. Only the
# torchtitan Python dependencies are layered on; torch itself is installed at
# test time from the nightly xpu index, same as the CUDA/ROCm flows.
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# The base image's py_3.11 env pins numpy-1-era pandas/scipy, which the xpu
# nightly torch wheel breaks by upgrading numpy to 2.x. Build a fresh py_3.12
# env instead (matching the CUDA flow's PYTHON_VERSION) so every dependency is
# resolved against the same numpy major.
ARG PYTHON_VERSION=3.12
COPY --chown=jenkins:jenkins requirements.txt requirements-dev.txt /tmp/
USER jenkins
# -p (absolute prefix): this image's conda silently falls back to
# ~/.conda/envs with -n despite /opt/conda/envs being first in envs_dirs
# and writable; the env still registers as py_3.12 for `conda activate`.
RUN /opt/conda/bin/conda create -p "/opt/conda/envs/py_${PYTHON_VERSION}" -y "python=${PYTHON_VERSION}" pip \
    && "/opt/conda/envs/py_${PYTHON_VERSION}/bin/python" -m pip install --no-cache-dir \
        -r /tmp/requirements.txt \
        -r /tmp/requirements-dev.txt \
    # The non-recursive chown in this image build means cache entries the
    # base image's own py_3.11 setup left under /opt/conda/pkgs (owned by a
    # uid jenkins no longer maps to) can't be removed; a 58GB recursive
    # chown to let clean fully succeed made image export take 65+ min and
    # tripped containerd export bugs, so this exit code is expected, not a
    # swallowed real failure.
    && { /opt/conda/bin/conda clean -ya || true; } \
    && rm /tmp/requirements.txt /tmp/requirements-dev.txt

CMD ["bash"]

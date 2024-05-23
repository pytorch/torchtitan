#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_miniconda() {
  BASE_URL="https://repo.anaconda.com/miniconda"
  CONDA_FILE="Miniconda3-py${PYTHON_VERSION//./}_${MINICONDA_VERSION}-Linux-x86_64.sh"

  mkdir -p /opt/conda
  chown ci-user:ci-user /opt/conda

  pushd /tmp
  wget -q "${BASE_URL}/${CONDA_FILE}"
  # Install miniconda
  as_ci_user bash "${CONDA_FILE}" -b -f -p "/opt/conda"
  # Clean up the download file
  rm "${CONDA_FILE}"
  popd

  sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
  export PATH="/opt/conda/bin:$PATH"
}

install_python() {
  pushd /opt/conda
  # Install the correct Python version
  as_ci_user conda create -n "py_${PYTHON_VERSION}" -y --file /opt/conda/conda-env-ci.txt python="${PYTHON_VERSION}"
  popd
}

install_pip_dependencies() {
  pushd /opt/conda
  # Install all Python dependencies
  pip_install -r /opt/conda/dev-requirements.txt
  pip_install -r /opt/conda/requirements.txt
  popd
}

fix_conda_ubuntu_libstdcxx() {
  cat /etc/issue
  # WARNING: This is a HACK from PyTorch core to be able to build PyTorch on 22.04.
  # Specifically, ubuntu-20+ all comes lib libstdc++ newer than 3.30+, but anaconda
  # is stuck with 3.29. So, remove libstdc++6.so.3.29 as installed by
  # https://anaconda.org/anaconda/libstdcxx-ng/files?version=11.2.0
  #
  # PyTorch sev: https://github.com/pytorch/pytorch/issues/105248
  # Ref: https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_conda.sh
  if grep -e "2[02].04." /etc/issue >/dev/null; then
    rm "/opt/conda/envs/py_${PYTHON_VERSION}/lib/libstdc++.so.6"
  fi
}

install_miniconda
install_python
install_pip_dependencies
fix_conda_ubuntu_libstdcxx

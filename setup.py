# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="torchtitan",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    description="Package for training large models using native PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/torchtitan",
)

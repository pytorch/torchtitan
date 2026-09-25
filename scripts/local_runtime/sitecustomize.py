# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare optional local-runtime overrides before Torch is imported."""

import os
import sys


source_root = os.environ.get("CODA_PYTORCH_SOURCE_ROOT")
if source_root:
    for finder in sys.meta_path:
        source_files = getattr(finder, "known_source_files", None)
        if not isinstance(source_files, dict):
            continue
        torch_init = source_files.get("torch")
        if torch_init is None:
            continue
        installed_source_root = os.path.dirname(os.path.dirname(torch_init))
        # pyrefly: ignore [missing-attribute]
        finder.known_source_files = {
            module: source_root + path.removeprefix(installed_source_root)
            for module, path in source_files.items()
        }

extension = os.environ.get("TORCHAO_MXFP8_EXTENSION")
if extension:
    import torch

    torch.ops.load_library(extension)

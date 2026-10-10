# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import pkgutil
import unittest

import torch
import torchtitan.models

from tests.utils import restore_aux_loss_registry


class TestModelBufferDtypes(unittest.TestCase):
    def test_debug_models_keep_buffer_dtypes_across_cast(self):
        # Every buffer in the model zoo is a cache or accumulator built in a
        # fixed dtype; one held by a plain nn.Module (not Module) fails here.
        restore_aux_loss_registry(self)
        for info in pkgutil.iter_modules(torchtitan.models.__path__):
            package = importlib.import_module(f"torchtitan.models.{info.name}")
            for flavor in getattr(package, "MODEL_FLAVORS", {}):
                if not flavor.startswith("debug"):
                    continue
                with self.subTest(model=info.name, flavor=flavor):
                    with torch.device("meta"):
                        model = package.build_model_config(flavor).build()
                    expected = {fqn: b.dtype for fqn, b in model.named_buffers()}
                    model.to(torch.bfloat16)
                    actual = {fqn: b.dtype for fqn, b in model.named_buffers()}
                    self.assertEqual(actual, expected)

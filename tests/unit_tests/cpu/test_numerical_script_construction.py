# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import unittest
from unittest import mock

import torch

from torchtitan.config.parallelism import ParallelismConfig

from tests.utils import restore_aux_loss_registry


class TestNumericalScriptConstruction(unittest.TestCase):
    def setUp(self):
        restore_aux_loss_registry(self)
        # The Qwen script disables compilation at import time.
        patch = mock.patch.object(
            torch._dynamo.config, "disable", torch._dynamo.config.disable
        )
        patch.start()
        self.addCleanup(patch.stop)

    def assert_parameter_dtype(self, parameters, dtype):
        parameters = list(parameters)
        self.assertTrue(parameters)
        for parameter in parameters:
            self.assertEqual(parameter.device.type, "cpu")
            self.assertEqual(parameter.dtype, dtype)

    def assert_initialized_buffer_dtypes(self, model, config):
        # These script configurations use caches and accumulators whose dtypes
        # must match fresh initialization, independent of parameter precision.
        with torch.device("meta"):
            reference = config.build()
        expected = {name: buffer.dtype for name, buffer in reference.named_buffers()}
        self.assertTrue(expected)
        actual = {name: buffer.dtype for name, buffer in model.named_buffers()}
        self.assertEqual(actual, expected)
        self.assertTrue(all(buffer.device.type == "cpu" for buffer in model.buffers()))

    @torch.no_grad()
    def test_qwen_fp16(self):
        script = importlib.import_module(
            "scripts.checkpoint_conversion.numerical_tests_qwen3_5"
        )
        config = script.build_model_config(
            "debugmodel", seq_len=32, parallelism=ParallelismConfig()
        )
        with mock.patch.object(script, "build_model_config", return_value=config):
            model = script._build_tt_model("debugmodel")
        self.assert_parameter_dtype(model.parameters(), torch.float16)
        self.assert_initialized_buffer_dtypes(model, config)

    @torch.no_grad()
    def test_kimi_bf16_with_fp16_vision(self):
        script = importlib.import_module(
            "scripts.checkpoint_conversion.numerical_tests_kimi"
        )
        config = script.build_model_config(
            "debugmodel", seq_len=32, parallelism=ParallelismConfig()
        )
        with mock.patch.object(script, "build_model_config", return_value=config):
            model = script._build_tt_model("debugmodel", torch.bfloat16, torch.float16)
        self.assert_parameter_dtype(model.vision_encoder.parameters(), torch.float16)
        text_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("vision_encoder.")
        ]
        self.assert_parameter_dtype(text_parameters, torch.bfloat16)
        self.assert_initialized_buffer_dtypes(model, config)

    @torch.no_grad()
    def test_kimi_k3_bf16(self):
        script = importlib.import_module(
            "scripts.checkpoint_conversion.numerical_tests_kimi_k3"
        )
        config = script.build_model_config(
            "debugmodel", seq_len=32, parallelism=ParallelismConfig()
        )
        model = script._build_tt_model(config, torch.bfloat16)
        self.assert_parameter_dtype(model.parameters(), torch.bfloat16)
        self.assert_initialized_buffer_dtypes(model, config)

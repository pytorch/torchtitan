# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan_recipes.tests.models.deepseek_v4 import (
    deepseek_v4_debugmodel,
    deepseek_v4_mtp_debugmodel,
)


class TestDeepSeekV4MTPConfig(unittest.TestCase):
    def test_dataloader_mtp_depth_matches_model(self):
        for config in (deepseek_v4_debugmodel(), deepseek_v4_mtp_debugmodel()):
            self.assertEqual(
                config.dataloader.num_mtp_layers, config.model.n_mtp_layers
            )

    def test_mtp_debugmodel_builds_mtp_layers(self):
        config = deepseek_v4_mtp_debugmodel()
        model_config = config.model
        self.assertEqual(model_config.n_mtp_layers, 1)
        self.assertEqual(len(model_config.mtp_layers), 1)

    def test_mtp_metadata_remains_replicated_at_block_boundary(self):
        config = build_model_config(
            "debugmodel",
            n_mtp_layers=1,
            parallelism=ParallelismConfig(
                enable_sequence_parallel=True, expert_parallel_degree=2
            ),
        )

        mtp_config = config.mtp_layers[0].sharding_config
        assert mtp_config is not None
        self.assertIsNotNone(mtp_config.input_shardings)


def _preprocess(model, input_dict, **kwargs):
    # SPMD annotation needs a device mesh; these tests check the plumbing only.
    with patch(
        "torchtitan.models.common.mtp.annotate_input_spmd_types",
        side_effect=lambda _parallelism_context, batch, _input_sharding: batch,
    ):
        return model.preprocess_inputs(
            input_dict,
            parallelism_context=SimpleNamespace(
                cp_enabled=False, activate_spmd=contextlib.nullcontext
            ),
            parallelism=SimpleNamespace(),
            **kwargs,
        )


class TestDeepSeekV4MTPInputs(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(AuxLoss._group_counts, clear=True))
        self.enterContext(patch.dict(AuxLoss.group_acc, clear=True))

    def test_preprocess_inputs_prepares_each_depth(self):
        num_mtp_layers = 4
        with torch.device("meta"):
            model = build_model_config(
                "debugmodel",
                seq_len=128,
                n_mtp_layers=num_mtp_layers,
                parallelism=ParallelismConfig(expert_parallel_degree=1),
            ).build()
        # Two packed documents (70 and 50 tokens) and 8 padding tokens.
        document_1 = torch.arange(71, dtype=torch.long) + 100
        document_2 = torch.arange(51, dtype=torch.long) + 1000
        positions = torch.cat(
            [
                torch.arange(70, dtype=torch.long),
                torch.arange(50, dtype=torch.long),
                torch.arange(8, dtype=torch.long),
            ]
        )
        padding_mask = torch.cat(
            [
                torch.zeros(70, dtype=torch.bool),
                torch.zeros(50, dtype=torch.bool),
                torch.ones(8, dtype=torch.bool),
            ]
        )
        tokens = torch.cat(
            [
                document_1[:-1],
                document_2[:-1],
                torch.zeros(8, dtype=torch.long),
            ]
        )
        labels = torch.cat(
            [
                document_1[1:],
                document_2[1:],
                torch.full((8,), IGNORE_INDEX, dtype=torch.long),
            ]
        )

        inputs, loss_labels, model_kwargs, _ = _preprocess(
            model,
            {
                "input": tokens,
                "labels": labels,
                "positions": positions,
                "padding_mask": padding_mask,
            },
        )

        self.assertEqual(len(inputs), num_mtp_layers + 1)
        self.assertTrue(torch.equal(inputs[0], tokens))
        self.assertTrue(torch.equal(model_kwargs["input_ids_T"], tokens))
        self.assertEqual(len(model_kwargs["mtp_input_valid_masks"]), num_mtp_layers)
        self.assertEqual(len(loss_labels), num_mtp_layers + 1)
        self.assertTrue(torch.equal(loss_labels[0], labels))


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDeepSeekV3MTPRemat(unittest.TestCase):
    def test_mtp_blocks_are_checkpointed_and_match_no_remat(self):
        """Activation remat checkpoints the MTP blocks, with bit-identical outputs and gradients."""
        torch.manual_seed(0)
        device = torch.device("cuda")
        seq_len = 512
        config = build_model_config(
            "debugmodel",
            seq_len=seq_len,
            num_mtp_layers=2,
            parallelism=ParallelismConfig(),
            attn_backend="flex",
        )
        with torch.device("meta"):
            baseline = config.build()
        baseline = baseline.to(torch.bfloat16)
        baseline.to_empty(device=device)
        baseline.init_states()
        baseline.eval()  # no load-balancing statistics

        tokens_T = torch.randint(0, 100, (seq_len,), device=device)
        # Two documents, so the MTP shift is masked at the boundary.
        positions_T = torch.cat([torch.arange(200), torch.arange(312)]).to(device)
        shifted = [
            roll_mtp_sequence(
                tokens_T,
                shift=depth,
                positions=positions_T,
                fill_value=0,
                return_valid_mask=True,
            )
            for depth in (1, 2)
        ]
        attention_metadata = baseline._get_attention_metadata(positions=positions_T)

        def forward_backward(model):
            model.zero_grad(set_to_none=True)
            outputs = model(
                (tokens_T, *(tokens for tokens, _ in shifted)),
                positions_T,
                attention_metadata,
                mtp_input_valid_masks=tuple(valid for _, valid in shifted),
            )
            sum(output.float().logsumexp(-1).mean() for output in outputs).backward()
            grads = {
                name: parameter.grad.clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            return [output.detach() for output in outputs], grads

        expected_outputs, expected_grads = forward_backward(baseline)
        self.assertTrue(any(name.startswith("mtp_layers.") for name in expected_grads))

        for policy in (FullActivationRematConfig(), DefaultActivationRematConfig()):
            with self.subTest(policy=type(policy).__qualname__):
                model = copy.deepcopy(baseline)
                policy.apply(
                    model, block_container_fqns=model._remat_block_container_fqns()
                )
                self.assertTrue(
                    all("forward" in vars(block) for block in model.mtp_layers)
                )
                outputs, grads = forward_backward(model)
                for output, expected in zip(outputs, expected_outputs, strict=True):
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assertEqual(grads.keys(), expected_grads.keys())
                for name, grad in grads.items():
                    torch.testing.assert_close(
                        grad, expected_grads[name], rtol=0, atol=0, msg=name
                    )


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest
from dataclasses import replace

import torch

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestVarlenGraph(unittest.TestCase):
    def test_trace_and_serialization_preserve_sequence_boundaries(self):
        torch.manual_seed(42)
        attention = VarlenInnerAttention.Config().build()
        inputs = [
            torch.randn(64, 4, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
            for _ in range(3)
        ]

        def step(q, k, v, *, attention_metadata):
            output = attention(q, k, v, attention_metadata=attention_metadata)
            loss = output.float().square().mean()
            return output, loss, torch.autograd.grad(loss, (q, k, v))

        boundaries = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
        metadata = VarlenAttentionMetadata(boundaries, boundaries, 64, 64)
        traced = minimal_fx_tracer(step, module=attention)(
            *inputs, attention_metadata=metadata
        )
        restored = replace(
            traced,
            gm=pickle.loads(pickle.dumps(traced.gm)),
            user_inputs_spec=pickle.loads(pickle.dumps(traced.user_inputs_spec)),
        )
        replay = run_traced(restored, module=attention)

        # Replay different boundaries to detect metadata baked into the graph.
        for split in (32, 16):
            with self.subTest(split=split):
                boundaries = torch.tensor(
                    [0, split, 64], device="cuda", dtype=torch.int32
                )
                metadata = VarlenAttentionMetadata(boundaries, boundaries, 64, 64)
                expected = step(*inputs, attention_metadata=metadata)
                actual = replay(*inputs, attention_metadata=metadata)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

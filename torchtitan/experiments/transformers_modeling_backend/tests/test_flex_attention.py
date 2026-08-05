# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-sample leakage test for the flex-attention (packing) path.

When samples are packed into one sequence, attention must not cross document
boundaries. Flex attention with a document BlockMask expresses that. This
verifies that the document mask the backend builds (``get_attention_masks``)
actually blocks cross-document attention through the backend's flex function
(``_flex_attention_torchtitan``): perturbing an earlier document's keys/values
must leave a later document's output bit-identical.

Run with:
    python -m pytest \
      torchtitan/experiments/transformers_modeling_backend/tests/test_flex_attention.py
"""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "flex_attention requires CUDA")
class TestFlexCrossDocumentLeakage(unittest.TestCase):
    def test_packed_documents_do_not_leak(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.experiments.transformers_modeling_backend.model import (
            _flex_attention_torchtitan,
        )
        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_document_mask_mod,
        )

        device = "cuda"
        len_a, len_b, n_heads, head_dim = 8, 12, 4, 32
        seq = len_a + len_b

        # Packed positions: two documents, position resets to 0 at the boundary.
        positions = torch.tensor(
            [list(range(len_a)) + list(range(len_b))], device=device
        )
        # The same document-causal BlockMask the model builds in get_attention_masks.
        mask_mod = and_masks(
            get_causal_mask_mod(),
            get_document_mask_mod(positions),
        )
        block_mask = create_attention_mask(
            mask_mod, 1, None, seq, seq, device=device, BLOCK_SIZE=128
        )

        torch.manual_seed(0)
        shape = (1, n_heads, seq, head_dim)  # (batch, heads, seq, dim)
        q = torch.randn(shape, device=device, dtype=torch.bfloat16)
        k = torch.randn(shape, device=device, dtype=torch.bfloat16)
        v = torch.randn(shape, device=device, dtype=torch.bfloat16)

        module = torch.nn.Module()  # no num_key_value_groups -> no GQA repeat
        # Output layout is (batch, seq, heads, dim).
        out1, _ = _flex_attention_torchtitan(module, q, k, v, block_mask)

        # Perturb document A's keys/values only. Document B is both causally later
        # and in a different document, so it must not attend to A -> B's output is
        # unchanged. (Document A's own output changes, which the sanity check below
        # confirms so the test can't pass trivially.)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, :len_a] = torch.randn_like(k2[:, :, :len_a])
        v2[:, :, :len_a] = torch.randn_like(v2[:, :, :len_a])
        out2, _ = _flex_attention_torchtitan(module, q, k2, v2, block_mask)

        # Document B rows must be bit-identical: masked-out columns contribute
        # exactly nothing to the softmax, so changing them cannot move B's output.
        torch.testing.assert_close(out1[:, len_a:], out2[:, len_a:], rtol=0, atol=0)
        # Sanity: document A's output did change (we perturbed its own k/v).
        self.assertFalse(torch.allclose(out1[:, :len_a], out2[:, :len_a]))


class TestAttnMaskTypeValidation(unittest.TestCase):
    """A block_causal config builds a flex BlockMask and is supported.

    Building the config is lightweight (no model build, no GPU/distributed), so
    this checks that a block_causal config builds without error.
    """

    def test_block_causal_builds(self):
        from torchtitan.experiments.transformers_modeling_backend import (
            TitanModelConfig,
        )
        from torchtitan.experiments.transformers_modeling_backend.model import (
            HFTransformerModel,
        )

        # Flex supports the document BlockMask -- no error.
        HFTransformerModel.Config(
            model_config=TitanModelConfig(
                dim=256,
                n_layers=2,
                n_heads=16,
                n_kv_heads=16,
                attn_mask_type="block_causal",
            ),
        )


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.flux.layers import (
    local_concat_text_image,
    local_pack,
    local_split_text_image,
    local_unpack,
)
from torchtitan.models.flux.sharding import (
    create_flux_attention_metadata,
    flux_inner_attention_sharding_config,
)


class TestFluxSharding(unittest.TestCase):
    def test_context_parallel_attention_uses_gathered_keys_and_values(self):
        sharding = flux_inner_attention_sharding_config()
        assert sharding.in_src_shardings is not None
        self.assertEqual(set(sharding.in_src_shardings), {"q_THK", "k_THK", "v_THV"})
        self.assertEqual(
            _per_axis_types(sharding.in_src_shardings["q_THK"])[MeshAxisName.CP],
            spmd.S(0),
        )
        for name in ("k_THK", "v_THV"):
            self.assertEqual(
                sharding.in_src_shardings[name].local_type[MeshAxisName.CP], spmd.R
            )

    def test_packed_layout_preserves_sample_boundaries(self):
        text = torch.arange(12).reshape(2, 3, 2)
        image = torch.arange(20).reshape(2, 5, 2) + 100
        combined = local_concat_text_image(local_pack(text), local_pack(image), 2)
        torch.testing.assert_close(
            local_unpack(combined, 2), torch.cat((text, image), dim=1)
        )
        unpacked_text, unpacked_image = local_split_text_image(combined, 3, 2)
        torch.testing.assert_close(unpacked_text, text.flatten(0, 1))
        torch.testing.assert_close(unpacked_image, image.flatten(0, 1))
        with patch("torchtitan.models.flux.sharding.spmd_mesh_group") as group:
            group.return_value.size.return_value = 2
            metadata = create_flux_attention_metadata(2, 8, device=torch.device("cpu"))
        torch.testing.assert_close(
            metadata.cu_seq_q, torch.tensor([0, 8, 16], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.cu_seq_k, torch.tensor([0, 16, 32], dtype=torch.int32)
        )
        self.assertEqual((metadata.max_q, metadata.max_k), (8, 16))


if __name__ == "__main__":
    unittest.main()

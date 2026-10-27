# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GDN normalization policy across local, CP, and recomputed backward paths."""

from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest
import torch

from torchtitan.models.common.attention import cp_gdn, gdn
from torchtitan.models.common.attn_gym import ContextParallelRouting


def _sum_kernel(q, k, v, g, beta, **kwargs):
    return q + k + v + g.unsqueeze(-1) + beta.unsqueeze(-1), None


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("path", ["chunk", "recomputed_backward", "context_parallel"])
def test_l2norm_backward_tuning_obeys_determinism(deterministic, path):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        q, k, v = (torch.ones(3, 2, 4) for _ in range(3))
        g, beta = (torch.ones(3, 2) for _ in range(2))
        offsets = torch.tensor([0, 1, 3], dtype=torch.int32)
        with (
            mock.patch.object(gdn, "require_attn_gym"),
            mock.patch.object(cp_gdn, "require_attn_gym"),
            mock.patch.object(gdn, "chunk_kernel_options", return_value=None),
            mock.patch.object(gdn, "chunk_gdn", side_effect=_sum_kernel),
            mock.patch.object(cp_gdn, "context_parallel_gdn", side_effect=_sum_kernel),
            mock.patch.object(cp_gdn, "spmd_mesh_group", return_value=object()),
            mock.patch.object(
                gdn, "l2norm", side_effect=lambda x, **kwargs: x
            ) as local_norm,
            mock.patch.object(
                cp_gdn, "l2norm", side_effect=lambda x, **kwargs: x
            ) as cp_norm,
        ):
            if path == "chunk":
                kernel = gdn.GDNKernel.Config().build()
                kernel(q, k, v, g, beta, cu_seqlens=offsets)
            elif path == "recomputed_backward":
                inputs = tuple(x.unsqueeze(0) for x in (q, k, v, g, beta))
                gradients = gdn._chunk_gdn_gradients(
                    torch.ones_like(inputs[0]), *inputs, offsets
                )
                for gradient in gradients[:3]:
                    torch.testing.assert_close(gradient, torch.ones_like(gradient))
                for gradient in gradients[3:]:
                    torch.testing.assert_close(gradient, torch.full_like(gradient, 4))
            else:
                inner = cp_gdn.ContextParallelInnerGDN.Config(
                    conv_kernel_size=4, kernel=gdn.GDNKernel.Config()
                ).build()
                routing = cast(
                    ContextParallelRouting, SimpleNamespace(cu_seqlens=offsets)
                )
                inner.gdn_core(q, k, v, g, beta, cu_seqlens=offsets, routing=routing)
            normalizer = cp_norm if path == "context_parallel" else local_norm
            assert normalizer.call_count == 2
            for call in normalizer.call_args_list:
                assert call.kwargs["autotune"] is (not deterministic)
                assert call.kwargs["cu_seqlens"] is offsets
    finally:
        torch.use_deterministic_algorithms(previous)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selected-layer FSDP offload wiring and temporary MTP unit registration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v3.mtp import apply_fsdp_to_mtp_decoder, MTPDecoder


def test_mtp_model_forwards_selected_layer_offload_policy():
    """Keep FSDP wrapping consistent with selected-layer materialization."""
    model = MagicMock(spec=MTPDecoder)
    dp_mesh, dp_mesh_dims = object(), object()
    edp_mesh, edp_mesh_dims = object(), object()
    parallelism_context = SimpleNamespace(pp_enabled=False, ep=2)
    training = TrainingConfig(
        mixed_precision_param="bfloat16", mixed_precision_reduce="float32"
    )
    parallelism = ParallelismConfig(
        fsdp_cpu_offload_layer_fqns=("layers.7",),
        fsdp_enable_explicit_backward_prefetch=False,
    )
    with (
        patch(
            "torchtitan.distributed.fsdp.resolve_fsdp_mesh",
            return_value=(dp_mesh, dp_mesh_dims),
        ),
        patch(
            "torchtitan.distributed.fsdp.resolve_sparse_fsdp_mesh",
            return_value=(edp_mesh, edp_mesh_dims),
        ),
        patch(
            "torchtitan.models.deepseek_v3.mtp.apply_fsdp_to_mtp_decoder"
        ) as apply_mtp_fsdp,
    ):
        MTPDecoder._apply_fsdp(
            model,
            parallelism_context=parallelism_context,
            training=training,
            parallelism=parallelism,
        )

    apply_mtp_fsdp.assert_called_once_with(
        model,
        dp_mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        cpu_offload=False,
        cpu_offload_layer_fqns=("layers.7",),
        reshard_after_forward_policy="default",
        enable_explicit_backward_prefetch=False,
        ep_degree=2,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=None,
    )


def test_mtp_fsdp_helper_forwards_selected_layer_offload_policy():
    """Forward the policy while restoring temporary MTP layer aliases."""
    model = SimpleNamespace(
        layers=nn.ModuleDict({"0": nn.Identity()}),
        mtp_layers=nn.ModuleList([nn.Identity()]),
    )
    dp_mesh = object()
    with patch(
        "torchtitan.models.deepseek_v3.mtp.apply_fsdp_to_decoder"
    ) as apply_decoder_fsdp:
        apply_fsdp_to_mtp_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            cpu_offload_layer_fqns=("layers.0",),
        )

    apply_decoder_fsdp.assert_called_once_with(
        model,
        dp_mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        cpu_offload=False,
        cpu_offload_layer_fqns=("layers.0",),
        reshard_after_forward_policy="default",
        enable_explicit_backward_prefetch=True,
        ep_degree=1,
        edp_mesh=None,
        dp_mesh_dims=None,
        edp_mesh_dims=None,
        symm_mem_scope=None,
    )
    assert list(model.layers) == ["0"]

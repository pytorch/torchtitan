# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest import mock

import pytest
import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torchtitan.config.parallelism import ParallelismConfig

from torchtitan.distributed.fsdp import resolve_fsdp_mesh
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.multimodal import (
    build_dummy_vision_inputs,
    MultimodalModel,
)
from torchtitan.models.common.vision_encoder_sharding import (
    multimodal_input_sharding,
    set_vision_encoder_cp_invariant,
)
from torchtitan.models.kimi_k2_7.sharding import set_moonvit_sharding_config
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _compute_2d_rope_cache,
    VisionRotaryEmbedding2D,
)
from torchtitan.models.kimi_k3.flavors import _vision_encoder_config, build_model_config
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.protocols.module import Module


@pytest.mark.parametrize("invariant", [False, True])
def test_encoder_cp_plan_preserves_tp_and_decoder_boundary(invariant: bool) -> None:
    config = _vision_encoder_config(
        text_dim=32,
        dim=16,
        qkv_dim=16,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
    )
    set_moonvit_sharding_config(config, projector_norm="post_norm")
    if invariant:
        set_vision_encoder_cp_invariant(config)
    cp_type = spmd.I if invariant else spmd.R
    for _, child, _, _ in config.traverse(Module.Config, recurse=True):
        sharding = child.sharding_config
        if sharding is None:
            continue
        for layout in sharding.state_shardings.values():
            assert layout.local_type[MeshAxisName.CP] is cp_type
        if sharding.out_src_shardings is not None:
            assert sharding.out_src_shardings.local_type[MeshAxisName.CP] is cp_type
    boundary = config.sharding_config
    assert boundary.out_dst_shardings.local_type[MeshAxisName.CP] is cp_type
    assert boundary.out_dst_shardings.local_type[MeshAxisName.TP] is spmd.R
    assert config.block.attn.wq.sharding_config.state_shardings["weight"].local_type[
        MeshAxisName.TP
    ] == spmd.S(0)


@pytest.mark.parametrize("flavor", ["debugmodel", "Kimi-K3"])
def test_kimi_k3_encoder_is_cp_invariant_by_default(flavor: str) -> None:
    config = build_model_config(flavor, seq_len=128)
    assert "encoder_cp_invariant" not in config.__dataclass_fields__
    config.set_sharding_(ParallelismConfig())
    assert (
        config.vision_encoder.patch_embed_proj.sharding_config.state_shardings[
            "weight"
        ].local_type[MeshAxisName.CP]
        is spmd.I
    )
    assert (
        config.tok_embeddings.sharding_config.state_shardings["weight"].local_type[
            MeshAxisName.CP
        ]
        is spmd.R
    )


@pytest.mark.parametrize("cp_enabled", [False, True])
def test_kimi_k3_vision_inputs_are_cp_invariant(cp_enabled: bool) -> None:
    model = KimiK3Model.__new__(KimiK3Model)
    torch.nn.Module.__init__(model)
    model.tok_embeddings = torch.nn.Embedding(4, 4)
    inputs = {
        "input": torch.tensor([0, 1]),
        "labels": torch.tensor([1, 2]),
        "pixel_values": torch.zeros(4, 12),
        "grid_thw": torch.tensor([[1, 2, 2]]),
        "special_tokens": {"image_id": 0},
    }
    with (
        mock.patch.object(model, "_cp_shard", side_effect=lambda data, **_: data),
        mock.patch(
            "torchtitan.models.kimi_k3.model.annotate_input_spmd_types",
            side_effect=lambda context, data, layouts: data,
        ) as annotate,
    ):
        model.preprocess_inputs(
            inputs,
            parallelism_context=mock.Mock(cp_enabled=cp_enabled),
            parallelism=ParallelismConfig(),
        )
    layouts = annotate.call_args.args[2]
    assert layouts["pixel_values"].local_type[MeshAxisName.CP] is spmd.I
    assert layouts["grid_thw"].local_type[MeshAxisName.CP] is spmd.I
    assert layouts["input"].local_type[MeshAxisName.CP] is spmd.V


@pytest.mark.parametrize("cp_type", [spmd.R, spmd.I])
def test_cp_vision_metadata_and_dummy_inputs(cp_type) -> None:
    axes = {
        "dp": spmd.MeshAxis.of(2, 4),
        "cp": spmd.MeshAxis.of(2, 2),
        "tp": spmd.MeshAxis.of(2, 1),
    }
    rotary = VisionRotaryEmbedding2D.Config(head_dim=8).build()
    with spmd.set_current_mesh(axes, local_axes=(axes["dp"],)):
        spmd.assert_type(rotary.inv_freq, {"dp": spmd.R, "cp": cp_type, "tp": spmd.I})
        with typecheck(strict_mode="strict", local=False):
            table = rotary(4)
            spmd.assert_type(table, {"dp": spmd.R, "cp": cp_type, "tp": spmd.I})
            cache = _compute_2d_rope_cache(table, [[1, 2, 4]], 8)
            spmd.assert_type(cache, {"dp": spmd.V, "cp": cp_type, "tp": spmd.I})
            pixels, grids = build_dummy_vision_inputs(
                patch_dim=12,
                grid_thw=(1, 2, 2),
                device=torch.device("cpu"),
                cp=cp_type,
            )
            layouts = multimodal_input_sharding(cp=cp_type)
            spmd.assert_type(pixels, layouts["pixel_values"])
            spmd.assert_type(grids, layouts["grid_thw"])


@pytest.mark.parametrize("shard_cp", [False, True])
@pytest.mark.parametrize("replicate", [False, True])
def test_encoder_fsdp_keeps_cp_storage_axis_without_reduction(
    shard_cp: bool, replicate: bool
) -> None:
    mesh = mock.Mock()
    mesh.size.return_value = 4
    context = mock.Mock(cp_enabled=True, dp_replicate_enabled=replicate)
    context.get_activated_mesh.return_value = mesh
    storage_mesh, axes = resolve_fsdp_mesh(context, shard_cp=shard_cp)
    assert storage_mesh is mesh
    context.get_activated_mesh.assert_called_once_with(
        ["dp_replicate", "dp_shard", "cp", "tp"]
    )
    assert axes.shard_names == (("dp_shard", "cp") if shard_cp else ("dp_shard",))
    assert axes.replicate_names == (("dp_replicate",) if replicate else ())


@pytest.mark.parametrize("model_type", [MultimodalModel, KimiK3Model])
@pytest.mark.parametrize("pp_enabled", [False, True])
def test_kimi_k3_encoder_fsdp_is_always_dp_only(
    model_type: type[MultimodalModel], pp_enabled: bool
) -> None:
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.multimodal_encoder_fqns = ("vision_encoder",)
    model.vision_encoder = torch.nn.Linear(4, 4)
    context = mock.Mock(pp_enabled=pp_enabled)
    with (
        mock.patch(
            "torchtitan.distributed.fsdp.resolve_fsdp_mesh", return_value=(None, None)
        ) as resolve_mesh,
        mock.patch(
            "torchtitan.distributed.fsdp.apply_fsdp_to_multimodal_encoder"
        ) as wrap_encoder,
        mock.patch.object(Decoder, "_apply_fsdp") as wrap_decoder,
    ):
        model._apply_fsdp(
            parallelism_context=context,
            training=SimpleNamespace(
                mixed_precision_param="bfloat16",
                mixed_precision_reduce="float32",
                enable_cpu_offload=False,
            ),
            parallelism=ParallelismConfig(),
        )
    if model_type is KimiK3Model or not pp_enabled:
        if model_type is KimiK3Model:
            resolve_mesh.assert_called_once_with(context, shard_cp=False)
        else:
            resolve_mesh.assert_called_once_with(context)
        wrap_encoder.assert_called_once()
        assert wrap_encoder.call_args.args[0] is model.vision_encoder
    else:
        wrap_encoder.assert_not_called()
    wrap_decoder.assert_called_once()

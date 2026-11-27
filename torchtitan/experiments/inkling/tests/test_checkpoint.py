# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import re
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint import FileSystemReader

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config.transform import LoRATransform
from torchtitan.experiments.inkling import build_model_config, InklingModel
from torchtitan.experiments.inkling.state_dict_adapter import (
    export_merged_checkpoint,
    InklingStateDictAdapter,
)
from torchtitan.tools.utils import set_default_dtype
from torchtitan_recipes.models.inkling import inkling_2_mini_lora


def test_internal_lora_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(123)
    cfg = cast(
        InklingModel.Config,
        LoRATransform(
            rank=2, alpha=6, target_modules=["layers.0.attn.wq_du"]
        ).transform(build_model_config("debugmodel")),
    )
    with set_default_dtype(torch.bfloat16):
        model = cfg.build()
    model.init_states()
    projection = model.get_submodule("layers.0.attn.wq_du")
    base = {
        f"model.llm.{k}": t.clone()
        for k, t in model.state_dict().items()
        if ".lora_" not in k
    }
    key = "model.llm.layers.0.attn.wq_du.weight"
    base[key] = torch.randn_like(base[key])
    base["model.visual.extra"] = torch.tensor([1.25])
    filename = "model-00001-of-00001.safetensors"
    source = tmp_path / "source"
    source.mkdir()
    # As in the HF cache, the writable export must be independent of the source blob.
    blob = tmp_path / "blob"
    save_file(base, blob)
    (source / filename).symlink_to(blob)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(base, filename)})
    )
    loop = {
        "core_start": 1,
        "core_end": 3,
        "num_passes": 2,
        "gate_rank": 4,
        "is_varloop": False,
    }
    text = {
        "model_type": "inkling2_loop_text",
        "model_max_length": 16,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 8,
        "d_rel": 4,
        "rel_extent": 8,
        "sliding_window_size": 4,
        "local_layer_ids": [0, 2],
        "sconv_kernel_size": 4,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": math.sqrt(512),
        "rel_l2_clip_threshold": 8.0,
        "log_scaling_alpha": 0.1,
        "log_scaling_n_floor": 8192,
        "n_layers_per_attn_res_block": 1,
        "final_logit_softcapping": 30.0,
        "q_bias": False,
        "o_bias": False,
        "use_value_norm_no_gamma": True,
        "attn_res_normalize_values": True,
        "n_shared_experts": 0,
        "use_gate_bias": True,
        "loop_config": loop,
        "intermediate_size": 32,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "route_scale": 8.0,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    }
    config_path = source / "config.json"
    config_path.write_text(json.dumps({"text_config": text}))
    mx = {
        "block_size": 32,
        "format": "mxfp8_e4m3",
        "rounding": "nearest",
        "scale_dtype": "ue8m0",
    }
    nv = {
        "block_size": 16,
        "format": "nvfp4_e2m1",
        "rounding": "nearest",
        "scale_dtype": "float8_e4m3fn",
        "amax_type": "per_token",
        "scale_selection_rule": "fouroversix_mae",
    }
    quant = {
        "model": {
            "parameter_groups": [
                {
                    "required_activation_quantization": scheme,
                    "tensors": [
                        k
                        for k in base
                        if k.endswith(("w13_weight", "w2_weight"))
                        and (f"{k}_amax" in base) == fp4
                    ],
                }
                for fp4, scheme in [(True, nv), (False, mx)]
            ]
        }
    }
    quant_path = source / "hf_quant_config.json"
    quant_path.write_text(json.dumps(quant))
    save_file({"mtp.weight": torch.randn(3)}, source / "mtp.safetensors")
    initial_a = projection.get_parameter("lora_a.weight").detach().clone()
    adapter = InklingStateDictAdapter(cfg, str(source))
    recipe = inkling_2_mini_lora(checkpoint=source, text=tmp_path / "train.txt")
    checkpointer_config = recipe.checkpointer
    assert checkpointer_config is not None
    checkpointer_config.keep_latest_k = 0
    optimizers = recipe.optim.optimizer.build(model_parts=[model])
    manager = CheckpointManager(
        config=checkpointer_config,
        model_parts=[model],
        optimizers=optimizers,
        lr_schedulers=recipe.optim.lr_scheduler.build(
            optimizers=optimizers, training_steps=1
        ),
        ema=None,
        extra_states={},
        sd_adapter=adapter,
        base_folder=str(tmp_path),
    )
    assert manager.load()
    assert torch.equal(projection.get_parameter("lora_a.weight"), initial_a)
    assert torch.equal(projection.get_parameter("weight"), base[key])
    output = tmp_path / "merged"
    b = projection.get_parameter("lora_b.weight")
    with torch.no_grad():
        b.normal_()
    trained_b = b.detach().clone()
    assert manager.save(curr_step=1, last_step=True)
    metadata = FileSystemReader(Path(manager.folder) / "step-1").read_metadata()
    saved_model_keys = model.state_dict().keys() & metadata.state_dict_metadata.keys()
    assert saved_model_keys == {
        "layers.0.attn.wq_du.lora_a.weight",
        "layers.0.attn.wq_du.lora_b.weight",
    }
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.zero_()
    assert manager.load()
    manager.close()
    torch.testing.assert_close(b, trained_b, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        if ".lora_" not in name:
            original = base[f"model.llm.{name}"]
            assert torch.equal(
                tensor.view(-1).view(torch.uint8), original.view(-1).view(torch.uint8)
            ), name
    export_merged_checkpoint(model, source, output)
    merged = load_file(output / filename)
    expected = (
        base[key].float() + 3 * (trained_b.float() @ initial_a.float())
    ).bfloat16()
    torch.testing.assert_close(merged[key], expected, atol=0, rtol=0)
    for k in base.keys() - {key}:
        torch.testing.assert_close(
            merged[k].reshape(-1).view(torch.uint8),
            base[k].reshape(-1).view(torch.uint8),
        )
    assert torch.equal(load_file(blob)[key], base[key])
    assert (output / "mtp.safetensors").read_bytes() == (
        source / "mtp.safetensors"
    ).read_bytes()

    # DCP can copy FP32 values into the U8 scale buffer without reporting a mismatch.
    scale_key = "model.llm.layers.3.mlp.experts.w13_weight_scale"
    save_file(base | {scale_key: base[scale_key].float() + 0.5}, blob)
    with pytest.raises(ValueError, match=rf"{re.escape(scale_key)}.*U8.*F32"):
        adapter.get_hf_storage_reader(str(source))
    save_file(base, blob)

    # DCP discovers shards by filename, not from the weight map.
    duplicate_shard = source / "extra.safetensors"
    save_file({scale_key: base[scale_key].float()}, duplicate_shard)
    with pytest.raises(ValueError, match=rf"{re.escape(scale_key)}.*extra.safetensors"):
        adapter.get_hf_storage_reader(str(source))
    duplicate_shard.unlink()

    denominator_key = (
        "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator"
    )
    save_file({k: v for k, v in base.items() if k != denominator_key}, blob)
    with pytest.raises(ValueError, match=rf"Missing.*{re.escape(denominator_key)}"):
        adapter.get_hf_storage_reader(str(source))
    save_file(base, blob)

    # A different visit count has identical tensor shapes but different logits.
    loop["num_passes"] = 3
    config_path.write_text(json.dumps({"text_config": text}))
    with pytest.raises(ValueError, match="loop_config"):
        adapter.get_hf_storage_reader(str(source))
    loop["num_passes"] = 2
    config_path.write_text(json.dumps({"text_config": text}))
    mx["rounding"] = "floor"
    quant_path.write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="unsupported expert quantization"):
        adapter.get_hf_storage_reader(str(source))

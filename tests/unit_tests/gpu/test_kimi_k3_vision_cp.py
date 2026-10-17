# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.multimodal import build_vision_bank_indices
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.vision_cp import build_cp_subgroups
from torchtitan.models.kimi_k3.vision_cp.plan import plan_dynamic_cp

# [t, h, w] grids; each keeps a 144-patch image whole so every rank attends at least 128 queries.
_MIN_PATCHES = 200
_CASES = {
    "one image over the whole CP group": [[1, 12, 12], [1, 16, 16]],
    "two images over two sub-groups": [[1, 16, 16], [1, 12, 12], [1, 20, 12]],
    "four images, one rank each": [
        [1, 16, 16],
        [1, 16, 14],
        [1, 12, 12],
        [1, 20, 12],
        [1, 18, 16],
    ],
    "a video whose last rank holds only padding": [[1, 12, 12], [2, 10, 16]],
}


# max-autotune benchmarks fp32 kernels for every new shape, longer than the multi-GPU CI job allows.
def _compile_flex_without_autotune() -> None:
    FlexInnerAttention.inductor_configs["max_autotune"] = False
    FlexInnerAttention.inductor_configs["coordinate_descent_tuning"] = False
    FlexInnerAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexInnerAttention.inductor_configs
    )


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestKimiK3VisionDynamicCP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def device_type(self) -> str:
        return "cuda"

    @with_comms
    def test_split_images_match_the_whole_tower(self) -> None:
        _compile_flex_without_autotune()
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=self.world_size,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        parallelism_context.build_mesh()
        config = replace(
            build_model_config(
                "debugmodel", parallelism=ParallelismConfig()
            ).vision_encoder,
            dynamic_cp_min_patches=_MIN_PATCHES,
        )
        torch.manual_seed(0)
        tower = config.build().to(device=self.device_type)
        tower.init_states()
        subgroups = build_cp_subgroups(parallelism_context.get_mesh("cp"))
        kh, kw = tower.merge_kernel_size
        patch_dim = tower.patch_embed.in_features

        for name, grids in _CASES.items():
            plan = plan_dynamic_cp(
                grids, cp_size=self.world_size, kh=kh, min_patches=_MIN_PATCHES
            )
            self.assertIsNotNone(plan, name)
            generator = torch.Generator(device=self.device_type).manual_seed(1)
            num_patches = sum(t * h * w for t, h, w in grids)
            num_merged = sum((h // kh) * (w // kw) for _, h, w in grids)
            pixels = torch.randn(
                num_patches, patch_dim, device=self.device_type, generator=generator
            )
            upstream = torch.randn(
                num_merged,
                tower.projector.linear_2.out_features,
                device=self.device_type,
                generator=generator,
            )
            grid_thw = torch.tensor(grids, device=self.device_type)

            results = []
            for split in (False, True):
                tower.zero_grad(set_to_none=True)
                tower.set_cp_subgroups(subgroups if split else {})
                with parallelism_context.activate_spmd():
                    out = tower(pixels, grid_thw=grid_thw)
                (out * upstream).sum().backward()
                grads = {n: p.grad.clone() for n, p in tower.named_parameters()}
                if split:
                    # Every rank back-propagated the whole output, so each image counts once per rank.
                    for grad in grads.values():
                        dist.all_reduce(grad)
                        grad.div_(self.world_size)
                results.append((out.detach(), grads))

            (ref, ref_grads), (out, grads) = results
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5, msg=name)
            for param_name, grad in grads.items():
                ref_grad = ref_grads[param_name]
                rel = (grad - ref_grad).norm() / ref_grad.norm().clamp_min(1e-30)
                self.assertLess(rel.item(), 1e-4, f"{name}: {param_name}")

    @with_comms
    def test_data_parallel_groups_with_different_images_stay_matched(self) -> None:
        _compile_flex_without_autotune()
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        parallelism_context.build_mesh()
        config = replace(
            build_model_config(
                "debugmodel", parallelism=ParallelismConfig()
            ).vision_encoder,
            dynamic_cp_min_patches=_MIN_PATCHES,
        )
        torch.manual_seed(0)
        reference = config.build().to(device=self.device_type)
        reference.init_states()
        tower = copy.deepcopy(reference)
        fully_shard(
            tower,
            mesh=init_device_mesh(self.device_type, (self.world_size,)),
            reshard_after_forward=True,
        )
        tower.set_cp_subgroups(build_cp_subgroups(parallelism_context.get_mesh("cp")))
        dp_rank = parallelism_context.get_mesh("dp_shard").get_local_rank()
        grids = (
            [[1, 12, 12], [1, 16, 16], [1, 20, 12]] if dp_rank == 0 else [[1, 12, 12]]
        )
        generator = torch.Generator(device=self.device_type).manual_seed(dp_rank)
        pixels = torch.randn(
            sum(t * h * w for t, h, w in grids),
            tower.patch_embed.in_features,
            device=self.device_type,
            generator=generator,
        )
        grid_thw = torch.tensor(grids, device=self.device_type)
        with parallelism_context.activate_spmd():
            out = tower(pixels, grid_thw=grid_thw)
        out.sum().backward()
        with torch.no_grad():
            ref = reference(pixels, grid_thw=grid_thw)
        torch.testing.assert_close(out.detach(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestKimiK3VisionDynamicCPGradients(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    def _tower_grads(
        self,
        parallelism_context: ParallelismContext,
        dynamic_cp_min_patches: int | None,
    ) -> dict[str, torch.Tensor]:
        parallelism = ParallelismConfig(context_parallel_degree=self.world_size)
        config = build_model_config(
            "debugmodel", parallelism=parallelism, attn_backend="flex"
        )
        config = replace(
            config,
            vision_encoder=replace(
                config.vision_encoder, dynamic_cp_min_patches=dynamic_cp_min_patches
            ),
        )
        torch.manual_seed(0)
        with torch.device(self.device_type):
            model = config.build()
            model.init_states()
        model.parallelize(
            parallelism_context=parallelism_context,
            training=TrainingConfig(
                mixed_precision_param="float32", mixed_precision_reduce="float32"
            ),
            parallelism=parallelism,
            local_compile_regions=[],
            activation_remat=None,
            dump_folder="",
        )
        tower = model.vision_encoder
        assert tower is not None
        grids = [[1, 16, 16], [1, 12, 20]]
        kh, kw = tower.merge_kernel_size
        image_id = 7
        tokens = torch.cat(
            [
                torch.tensor([3, 4]),
                torch.full(((16 // kh) * (16 // kw),), image_id),
                torch.tensor([5]),
                torch.full(((12 // kh) * (20 // kw),), image_id),
                torch.tensor([6]),
            ]
        ).to(self.device_type)
        bank_indices = build_vision_bank_indices(tokens, placeholder_id=image_id)
        generator = torch.Generator(device=self.device_type).manual_seed(1)
        pixels = torch.randn(
            sum(t * h * w for t, h, w in grids),
            tower.patch_embed.in_features,
            device=self.device_type,
            generator=generator,
        )
        upstream = torch.randn(
            tokens.numel(), config.dim, device=self.device_type, generator=generator
        )
        local = tokens.numel() // self.world_size
        rows = slice(self.rank * local, (self.rank + 1) * local)
        with parallelism_context.activate_spmd():
            embeds = model._prepare_multimodal_embeds(
                tokens[rows],
                pixel_values=pixels,
                grid_thw=torch.tensor(grids, device=self.device_type),
                special_tokens={"image_id": image_id},
                vision_bank_indices_T=bank_indices[rows],
            )
        (embeds * upstream[rows]).sum().backward()
        return {
            name: param.grad.full_tensor()
            for name, param in tower.named_parameters()
            if param.grad is not None
        }

    @with_comms
    def test_split_tower_gradients_match_the_cp_invariant_tower(self) -> None:
        _compile_flex_without_autotune()
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=self.world_size,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        parallelism_context.build_mesh()
        reference = self._tower_grads(parallelism_context, None)
        split = self._tower_grads(parallelism_context, _MIN_PATCHES)
        self.assertEqual(split.keys(), reference.keys())
        for name, ref_grad in reference.items():
            rel = (split[name] - ref_grad).norm() / ref_grad.norm().clamp_min(1e-30)
            self.assertLess(rel.item(), 1e-5, name)


if __name__ == "__main__":
    unittest.main()

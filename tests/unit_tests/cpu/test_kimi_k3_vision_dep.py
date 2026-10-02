# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed import ParallelismContext
from torchtitan.models.common import Linear
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
from torchtitan.models.kimi_k3.pipeline_parallel.layout import infer_block_layout_tables
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep import (
    build_vision_replica,
    install_vision_dep,
)
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.runtime import VisionDep
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.stage import (
    VisionDepPipelineStage,
)

NUM_STAGES, MICROBATCHES, STEPS = 8, 8, 2
TOKENS, DIM, PATCH = 4, 4, 3
# (t, h, w) per image; micro-batches 2 and 5 are text only.
GRIDS = {
    0: [(1, 1, 2)],
    1: [(1, 2, 2)],
    3: [(1, 1, 1), (1, 1, 2)],
    4: [(1, 2, 2)],
    6: [(1, 1, 2)],
    7: [(1, 2, 1)],
}


class _Tower(nn.Module):
    merge_kernel_size = (1, 1)

    def __init__(self, gelu: bool) -> None:
        super().__init__()
        self.gelu = gelu
        weight = torch.arange(PATCH * DIM, dtype=torch.float32).view(PATCH, DIM)
        self.proj = nn.Parameter(weight % 3 - 1)

    def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor):
        features = pixel_values @ self.proj
        return F.gelu(features) if self.gelu else features


class _Stage(nn.Module):
    """One pipeline stage of a model whose first stage splices the tower's output in."""

    def __init__(self, index: int, gelu: bool) -> None:
        super().__init__()
        self.index = index
        if index == 0:
            self.embed = nn.Parameter(torch.arange(DIM, dtype=torch.float32) % 2 + 1)
            self.vision_encoder = _Tower(gelu)
        elif index < NUM_STAGES - 1:
            gain = torch.arange(DIM) == index % DIM
            self.scale = nn.Parameter((gain & (index % 3 == 0)).float())

    def forward(
        self,
        hidden: torch.Tensor,
        stack: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
    ):
        if self.index == 0:
            hidden = hidden * self.embed
            if pixel_values is not None:
                if vision_embeds is None:
                    vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
                n = vision_embeds.shape[0]
                hidden = torch.cat((hidden[:n] + vision_embeds, hidden[n:]))
            return hidden, hidden.unsqueeze(1)
        assert stack is not None
        if self.index == NUM_STAGES - 1:
            return (hidden + stack[:, 0]).sum(-1)
        return hidden + (hidden + stack[:, 0]) * self.scale, stack


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output * target).sum()


def _microbatches(device: torch.device):
    inputs, targets, kwargs = [], [], []
    for mb in range(MICROBATCHES):
        inputs.append(torch.full((TOKENS, 1), mb + 1.0, device=device))
        targets.append(torch.full((TOKENS,), mb % 3 + 1.0, device=device))
        if mb not in GRIDS:
            kwargs.append({})
            continue
        patches = sum(t * h * w for t, h, w in GRIDS[mb])
        pixels = torch.arange(patches * PATCH, dtype=torch.float32, device=device)
        kwargs.append(
            {
                "pixel_values": (pixels.view(patches, PATCH) + mb) % 3 - 1,
                "grid_thw": torch.tensor(GRIDS[mb], device=device),
            }
        )
    return inputs, targets, kwargs


def _modules(frozen_tower: bool, device: torch.device, gelu: bool) -> list[_Stage]:
    modules = [_Stage(s, gelu).to(device) for s in range(NUM_STAGES)]
    modules[0].vision_encoder.proj.requires_grad_(not frozen_tower)
    return modules


def _grads(modules) -> dict[str, torch.Tensor | None]:
    return {
        f"{module.index}.{name}": None if p.grad is None else p.grad.clone()
        for module in modules
        for name, p in module.named_parameters()
    }


def _train(modules, step_fn, lr: float):
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr) if params else None
    history = []
    for _ in range(STEPS):
        losses = step_fn()
        history.append((_grads(modules), losses))
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
    return history


def _run_single_device(frozen_tower: bool, device: torch.device, gelu: bool, lr: float):
    modules = _modules(frozen_tower, device, gelu)
    inputs, targets, kwargs = _microbatches(device)

    def step():
        losses = []
        for x, y, kw in zip(inputs, targets, kwargs, strict=True):
            out = modules[0](x, **kw)
            for module in modules[1:]:
                out = module(*out)
            loss = _loss(out, y)
            loss.backward()
            losses.append(loss.detach())
        return losses

    return _train(modules, step, lr)


class _VisionDepChecks:
    """The decoupled encoder process on four pipeline ranks against one device."""

    gelu = False
    exact = True
    lr = 1.0

    def _device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self.rank)

    def _run_pipeline(
        self, *, bubble: bool, frozen_tower: bool, evals: list, cost_ratio: float
    ):
        device = self._device()
        mine = list(range(self.rank, NUM_STAGES, self.world_size))
        every = _modules(frozen_tower, device, self.gelu)
        modules = [every[s] for s in mine]
        stages = [
            VisionDepPipelineStage(module, s, NUM_STAGES, device)
            for module, s in zip(modules, mine, strict=True)
        ]
        schedule_stages: list[_PipelineStageBase] = list(stages)
        schedule = ScheduleInterleaved1F1B(
            schedule_stages,
            n_microbatches=MICROBATCHES,
            loss_fn=_loss,
            scale_grads=False,
        )
        layout = infer_block_layout_tables(
            stage_to_rank=dict(stages[0].stage_index_to_group_rank),
            n_layers=NUM_STAGES,
            layers_per_block=NUM_STAGES,
            layer_to_stage={s: s for s in range(NUM_STAGES)},
            cache=True,
        )
        store = PPRankLocalCache()
        for stage in stages:
            stage.set_routing(layout, store)
        replica = copy.deepcopy(every[0].vision_encoder)
        with torch.no_grad():
            replica.proj.zero_()
        dep_schedule = install_vision_dep(
            schedule,
            stages,
            replica=replica,
            pp_groups=[list(range(self.world_size))],
            dp_group=None,
            tp_group=None,
            hidden_dim=DIM,
            compute_dtype=torch.float32,
            bubble=bubble,
            cost_ratio=cost_ratio,
        )
        inputs, targets, kwargs = _microbatches(device)
        first, last = 0 in mine, NUM_STAGES - 1 in mine
        arg_mbs = [(x,) for x in inputs] if first else None
        target_mbs = targets if last else None

        def step():
            losses: list[torch.Tensor] = []
            dep_schedule.step(
                arg_mbs=arg_mbs,
                kwarg_mbs=kwargs,
                target_mbs=target_mbs,
                losses=losses if last else None,
                return_outputs=False,
            )
            evaluated: list[torch.Tensor] = []
            with torch.no_grad():
                dep_schedule.eval(
                    arg_mbs=arg_mbs,
                    kwarg_mbs=kwargs,
                    target_mbs=target_mbs,
                    losses=evaluated if last else None,
                )
            evals.append([loss.detach() for loss in evaluated])
            return [loss.detach() for loss in losses]

        history = _train(modules, step, self.lr)
        return history, dep_schedule._dep.plan

    def _check(
        self,
        *,
        bubble: bool,
        frozen_tower: bool,
        cost_ratio: float = 0.5,
        backward_on: tuple[int, int] | None = None,
    ) -> None:
        reference = _run_single_device(frozen_tower, self._device(), self.gelu, self.lr)
        evals: list = []
        history, plan = self._run_pipeline(
            bubble=bubble, frozen_tower=frozen_tower, evals=evals, cost_ratio=cost_ratio
        )
        exact = {"rtol": 0, "atol": 0} if self.exact else {}
        failures: list[str] = []

        def expect(actual, expected, **tolerance) -> None:
            try:
                torch.testing.assert_close(actual, expected, **tolerance)
            except AssertionError as error:
                failures.append(str(error))

        for step in range(STEPS):
            grads, losses = history[step]
            ref_grads, ref_losses = reference[step]
            for name, grad in grads.items():
                expected = ref_grads[name]
                if expected is None or grad is None:
                    if (expected is None) != (grad is None):
                        failures.append(f"step {step} {name}: {grad} vs {expected}")
                elif step > 0 and not self.exact:
                    # After the first step every gradient inherits the tower's summation order.
                    expect(grad, expected, rtol=1e-5, atol=1e-4)
                elif name.endswith("proj") and (step > 0 or not self.exact):
                    # The tower's gradient is summed over ranks in another order.
                    expect(grad, expected, rtol=1e-5, atol=1e-5)
                else:
                    expect(grad, expected, **exact)
            if losses:
                expect(losses, ref_losses, **exact)
                expect(evals[step], ref_losses, **exact)
        placed = {kind for kind, _ in plan.placed}
        wanted = {"encode"} if frozen_tower else {"encode", "backward"}
        if placed != (wanted if bubble else set()):
            failures.append(f"placed {placed} with bubble={bubble}")
        if backward_on is not None:
            mb, rank = backward_on
            if ("backward", mb) not in plan.placed or plan.backward_rank[mb] != rank:
                failures.append(f"backward of {mb} not in an idle slot of rank {rank}")
        # Every rank reaches this reduction, so none waits at teardown for a failed one.
        failed = torch.tensor([len(failures)], device=self._device())
        dist.all_reduce(failed)
        if failed.item():
            self.fail("\n\n".join(failures) or "a check failed on another rank")


class TestKimiK3VisionDep(_VisionDepChecks, DTensorTestBase):
    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_encodes_before_and_backwards_after_the_schedule_match_one_device(self):
        self._check(bubble=False, frozen_tower=False)

    @with_comms
    def test_encodes_and_backwards_in_idle_slots_match_one_device(self):
        self._check(bubble=True, frozen_tower=False)

    @with_comms
    def test_a_backward_waits_in_its_idle_run_for_a_gradient_ready_later(self):
        self._check(
            bubble=True, frozen_tower=False, cost_ratio=0.25, backward_on=(6, 3)
        )

    @with_comms
    def test_a_frozen_tower_gets_no_gradient(self):
        self._check(bubble=True, frozen_tower=True)

    @with_comms
    def test_tensor_parallel_copies_sync_their_shards_and_reduce_to_the_tower(self):
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "tp"))
        pp_rank, tp_rank = mesh.get_coordinate()
        weight = torch.arange(24, dtype=torch.float32).view(8, 3)
        bias = torch.arange(3, dtype=torch.float32) + 1
        tower = None
        if pp_rank == 0:
            tower = nn.Module()
            tower.weight = nn.Parameter(
                distribute_tensor(weight, mesh["tp"], [Shard(0)])
            )
            tower.bias = nn.Parameter(
                distribute_tensor(bias, mesh["tp"], [Replicate()])
            )
        replica = nn.Module()
        replica.weight = nn.Parameter(torch.zeros(4, 3))
        replica.bias = nn.Parameter(torch.zeros(3))
        replica.merge_kernel_size = (1, 1)
        pp_groups = [[0, 2], [1, 3]]
        group, _ = dist.new_subgroups_by_enumeration(pp_groups)
        dep = VisionDep(
            replica,
            tower=tower,
            pp_ranks=next(g for g in pp_groups if self.rank in g),
            stage0_rank=0,
            group=group,
            dp_group=None,
            tp_group=mesh.get_group("tp"),
            hidden_dim=3,
            compute_dtype=torch.float32,
            pipeline_order=None,
            cost_ratio=1.0,
        )
        dep._sync_weights()
        torch.testing.assert_close(replica.weight, weight.chunk(2)[tp_rank])
        torch.testing.assert_close(replica.bias, bias)
        assert dep._accum is not None
        dep._accum.copy_(
            (pp_rank + 1) * torch.cat((replica.weight.flatten(), replica.bias)).detach()
        )
        dep._reduce_gradients()
        if tower is not None:
            torch.testing.assert_close(tower.weight.grad.full_tensor(), 3 * weight)
            torch.testing.assert_close(tower.bias.grad.full_tensor(), 3 * bias)


class TestVisionReplicaSeed(DTensorTestBase):
    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 1

    @with_comms
    def test_building_the_tower_copy_leaves_the_seeded_stream_alone(self):
        with torch.device("meta"):
            model = nn.Module()
            model.vision_encoder = Linear.Config(
                in_features=4, out_features=3, bias=True
            ).build()
        context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        torch.manual_seed(0)
        expected = torch.rand(8)
        torch.manual_seed(0)
        build_vision_replica(
            model,
            parallelism_context=context,
            training=SimpleNamespace(
                enable_cpu_offload=False, mixed_precision_param="float32"
            ),
            ac_config=None,
            dump_folder="",
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(torch.rand(8), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

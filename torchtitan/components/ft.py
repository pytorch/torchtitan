# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Optional, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.tensor import DTensor
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    if TYPE_CHECKING:
        from torchft import local_sgd

    has_torchft = True
else:
    has_torchft = False


class FTManager:
    def __init__(
        self,
        manager: Optional["ft.Manager"],
        group_size: int = 1,
        replica_id: int = 0,
    ) -> None:
        self._manager = manager
        self.group_size = group_size
        self.replica_id = replica_id
        if has_torchft and manager is not None:
            self.replicate_pg = ft.process_group.ManagedProcessGroup(self._manager)
            self.replicate_pg.register("dp_replicate")

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    @property
    def manager(self) -> "ft.Manager":
        assert self._manager is not None
        return self._manager

    def get_dp_info(self, dp_degree: int, dp_rank: int) -> tuple[int, int]:
        return dp_degree * self.group_size, dp_degree * self.replica_id + dp_rank

    def set_all_reduce_hook(self, model_parts: list[torch.nn.Module]) -> None:
        def all_reduce_hook(output):
            dist.all_reduce(output, group=self.replicate_pg, op=ReduceOp.AVG)

        def apply_set_all_reduce_hook(m):
            if isinstance(m, FSDPModule):
                m.set_all_reduce_hook(all_reduce_hook)

        for part in model_parts:
            part.apply(apply_set_all_reduce_hook)


def init_ft_manager(job: JobConfig) -> FTManager:
    """Initialize the FT manager if TorchFT is enabled.

    Args:
        job (JobConfig): The job configuration.

    Returns:
        FTManager: A wrapper around TorchFT.Manager
    """
    if not job.fault_tolerance.enable:
        return FTManager(None)

    if not has_torchft:
        raise ImportError("torchft is not installed. Please install it.")

    if job.fault_tolerance.min_replica_size < 1:
        raise ValueError("At least one FT replica is required.")

    pg = ft.ProcessGroupNCCL()

    # If the training method is specific, then the quorum should be synchronous
    use_async_quorum = job.fault_tolerance.semi_sync_method is None

    return FTManager(
        ft.Manager(
            pg=pg,
            min_replica_size=job.fault_tolerance.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=use_async_quorum,
            replica_id=f"torchtitan_ft_{job.fault_tolerance.replica_id}",
        ),
        group_size=job.fault_tolerance.group_size,
        replica_id=job.fault_tolerance.replica_id,
    )


@dataclass
class FTParallelDims(ParallelDims):
    ft_manager: FTManager

    def build_mesh(self, device_type: str) -> DeviceMesh:
        def func(
            device_type: str, mesh_shape: list[int], mesh_dim_names: list[str]
        ) -> DeviceMesh:
            from torchft.process_group import ft_init_device_mesh

            return ft_init_device_mesh(
                device_type=device_type,
                mesh_shape=mesh_shape,
                mesh_dim_names=mesh_dim_names,
                replicate_dim=mesh_dim_names.index("dp_replicate"),
                manager=self.ft_manager.manager,
            )

        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1 or name == "dp_replicate":
                dims.append(d)
                names.append(name)

        return self._build_mesh(device_type, dims, names, func)

    @property
    def dp_replicate_enabled(self):
        return True


def ft_dist_reduce(
    x: torch.Tensor, reduceOp: str, mesh: DeviceMesh
) -> tuple[torch.Tensor, str, DeviceMesh]:
    if has_torchft and isinstance(mesh, ft.device_mesh._FlattenDeviceMesh):
        x = funcol.all_reduce(
            x, reduceOp=reduceOp, group=mesh.managed_mesh.replicate_pg
        )
        return x, reduceOp, mesh.managed_mesh.mesh
    return x, reduceOp, mesh


def ft_clip_grad_norm_util(total_norm: DTensor) -> torch.Tensor:
    if has_torchft:
        mesh = total_norm._spec.mesh
        if isinstance(mesh, ft.device_mesh.ManagedDeviceMesh):
            # The gradients along the replicated dim has already been reduced.
            # So we don't need another reducution beforing removing the
            # replicate dimension
            local_tensor = total_norm.to_local()
            placements = list(copy.copy(total_norm._spec.placements))
            placements.pop(mesh.replicate_dim)
            return DTensor.from_local(local_tensor, mesh.mesh, placements)

    return total_norm


def maybe_semi_sync_training(
    config: JobConfig,
    ft_manager: FTManager,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sync_every: int,
) -> ContextManager[Union["local_sgd.DiLoCo", "local_sgd.LocalSGD", None]]:
    """
    If TorchFT is enabled and the config is set, use semi_sync_method
    """
    semi_sync_method = config.fault_tolerance.semi_sync_method
    torchft_enabled = config.fault_tolerance.enable
    if torchft_enabled and semi_sync_method is not None:
        from torchft import local_sgd

        assert (
            ft_manager._manager is not None
        ), "FTManager must be enabled to use semi-sync training."
        if semi_sync_method.lower() == "diloco":
            # Create the outer optimizer based on the inner optimizer parameters.
            params = [group["params"] for group in optimizer.param_groups]
            params = [param for sublist in params for param in sublist]
            outer_optimizer = torch.optim.SGD(
                params, lr=0.7, momentum=0.9, nesterov=True
            )

            return local_sgd.DiLoCo(
                manager=ft_manager._manager,
                model=model,
                inner_optimizer=optimizer,
                outer_optimizer=outer_optimizer,
                sync_every=sync_every,
            )
        elif semi_sync_method.lower() == "local_sgd":
            return local_sgd.LocalSGD(
                manager=ft_manager._manager,
                model=model,
                optimizer=optimizer,
                sync_every=sync_every,
            )
        else:
            raise ValueError(
                f"Unknown training method: {semi_sync_method}, only 'diloco' and 'local_sgd' are supported."
            )
    return nullcontext()

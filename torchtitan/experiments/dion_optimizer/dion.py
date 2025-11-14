# This code is from:
# https://github.com/microsoft/dion/blob/main/optimizers/dion.py

# @article{ahn2025dion,
#  title={Dion: Distributed Orthonormalized Updates},
#  author={Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John},
#  journal={arXiv preprint: 2504.05295},
#  year={2025}
# }

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Placement,
    randn as dtensor_randn,
    Replicate,
    Shard,
)
from torch.optim.optimizer import Optimizer, ParamsT

from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    dtensor_from_local,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach, lion_update_foreach

try:
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError:
    _StridedShard = None


@dataclass
class DionParamConfig:
    """
    Per-parameter configuration for Dion optimizer.
    """

    # Dimensions of the tensor that is sharded
    outer_shard_tensor_dim: Optional[int] = None
    inner_shard_tensor_dim: Optional[int] = None

    # Dimensions of the device mesh that the tensor is sharded over
    outer_shard_mesh_dim: Optional[int] = None
    inner_shard_mesh_dim: Optional[int] = None

    # Use transposed version of the algorithm
    is_transposed: bool = False

    # Whether to all-reduce compressed P and R instead of full gradient
    # This should always be False for 1D tensors
    compressed_all_reduce = False

    # Sharding configurations for the Q matrix
    Q_sharded_placements: Optional[Tuple[Placement]] = None
    Q_inner_unsharded_placements: Optional[Tuple[Placement]] = None


@dataclass
class DionMixedPrecisionConfig:
    """
    Configuration for mixed precision in Dion optimizer.
    None means that optimizer states will use the same dtype as each parameter.
    """

    # Momentum state for all algorithms
    momentum_dtype: Optional[torch.dtype] = None
    # Dion Q matrix
    Q_dtype: Optional[torch.dtype] = None
    # Adam variance state
    variance_dtype: Optional[torch.dtype] = None
    # TODO look into separate dtypes for communication operations


class Dion(Optimizer):
    """
    Distributed Dion Optimizer.
    https://arxiv.org/abs/2504.05295

    Args:
        params: Parameters for the optimizer.
        replicate_mesh: DeviceMesh or ProcessGroup for replicated data parallelism.
            Use DeviceMesh for hybrid sharded FSDP and ProcessGroup for DistributedDataParallel.
        outer_shard_mesh: Parameter sharding DeviceMesh, replicated during orthogonalization.
            This is the FS dimension in the paper.
        inner_shard_mesh: Parameter sharding DeviceMesh, sharded during orthogonalization.
            This is the TP dimension in the paper.
        replicate_mesh_grad_sync: If True, optimizer handles data-parallel gradient sync.
            If False, the optimizer expects gradients to be already synchronized.
        rank_fraction: r/d fraction for low-rank approximation. Used to compute the low-rank dimension.
            This may be specified per param-group to have different rank fractions.
        rank_multiple_of: Round up the low-rank dimension to a multiple of this number.
            This may be useful to ensure even sharding.
        lr: Base learning rate. For Dion, this will be scaled based on the matrix dimensions.
            For non-Dion algorithms, this is the actual learning rate and no additional scaling is done.

    Note: We assume parameters are all DTensor or all regular Tensors. All sharded tensors are assumed
    to be uniformly sharded - that is, each device along the sharding axis has identical size shards.
    The only distributed scenarios supported are:
        - DTensor + DeviceMesh: sharding with FSDP2 fully_shard() and/or TP parallelize_module().
        - regular Tensor + ProcessGroup: No sharding allowed. DDP may be used.
    FSDP1 (FullyShardedDataParallel wrapper class) is not supported.
    """

    def __init__(
        self,
        params: ParamsT,
        replicate_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        outer_shard_mesh: Optional[DeviceMesh] = None,
        inner_shard_mesh: Optional[DeviceMesh] = None,
        replicate_mesh_grad_sync: bool = True,
        rank_fraction: float = 1.0,
        rank_multiple_of: int = 1,
        lr: float = 0.01,
        mu: float = 0.95,  # Momentum for Dion
        betas: Tuple[float, float] = (0.9, 0.95),  # Betas for AdamW and Lion
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        power_iters: int = 1,  # Number of power iterations for low-rank approximation
        qr_method: str = "rcqr",  # Method for computing QR decomposition
        cqr_warmup_steps: int = 150,  # (ignored)
        rcqr_oversample: float = 1.25,  # Random sketch matrix oversampling for RCQR
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank_fraction <= 0 or rank_fraction > 1:
            raise ValueError(f"Invalid rank fraction: {rank_fraction}")
        if rank_multiple_of <= 0:
            raise ValueError(f"Invalid rank multiple of: {rank_multiple_of}")
        if power_iters != 1:
            raise ValueError("Async Dion only supports power_iters=1")
        if qr_method != "rcqr":
            raise ValueError("Async Dion only supports qr_method='rcqr'")

        # Check device mesh
        if replicate_mesh is not None:
            if not isinstance(replicate_mesh, (DeviceMesh, ProcessGroup)):
                raise TypeError(
                    f"Replicate mesh must be a DeviceMesh or ProcessGroup, but got {type(replicate_mesh)}."
                )
        if outer_shard_mesh is not None:
            if not isinstance(outer_shard_mesh, DeviceMesh):
                raise TypeError(
                    f"Outer shard mesh must be a DeviceMesh, but got {type(outer_shard_mesh)}."
                )
            if outer_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Outer shard mesh must be 1D, but got {outer_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if outer_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Outer shard mesh must be different from replicate mesh."
                )
        if inner_shard_mesh is not None:
            if not isinstance(inner_shard_mesh, DeviceMesh):
                raise TypeError(
                    f"Inner shard mesh must be a DeviceMesh, but got {type(inner_shard_mesh)}."
                )
            if inner_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Inner shard mesh must be 1D, but got {inner_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if inner_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Inner shard mesh must be different from replicate mesh."
                )
            if inner_shard_mesh == outer_shard_mesh:
                raise ValueError("Outer and inner shard meshes must be different.")

        # Default arguments for each param group
        defaults = dict(
            rank_fraction=rank_fraction,
            rank_multiple_of=rank_multiple_of,
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            oversample=rcqr_oversample,
            algorithm="dion",
            step=0,
        )
        super().__init__(params, defaults)

        # This is intentionally not in self.state so it doesn't get checkpointed
        # State here may change upon resharding a checkpoint, so we recompute it
        self._param_config: Dict[Tensor, DionParamConfig] = {}

        self._replicate_mesh = replicate_mesh
        self._outer_shard_mesh = outer_shard_mesh
        self._inner_shard_mesh = inner_shard_mesh
        self._replicate_mesh_grad_sync = replicate_mesh_grad_sync

        # Get world size for the replicate mesh
        if isinstance(replicate_mesh, DeviceMesh):
            self._replicate_world_size = replicate_mesh.size()
        elif isinstance(replicate_mesh, ProcessGroup):
            self._replicate_world_size = dist.get_world_size(replicate_mesh)
        elif replicate_mesh is None:
            self._replicate_world_size = 1
        else:
            raise TypeError(f"Invalid replicate mesh type: {type(replicate_mesh)}.")

        # Get global ranks for outer and inner shard meshes
        if self._outer_shard_mesh is not None:
            self._outer_shard_ranks = dist.get_process_group_ranks(
                self._outer_shard_mesh.get_group()
            )
        else:
            self._outer_shard_ranks = None
        if self._inner_shard_mesh is not None:
            self._inner_shard_ranks = dist.get_process_group_ranks(
                self._inner_shard_mesh.get_group()
            )
        else:
            self._inner_shard_ranks = None

        # Mixed precision
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config
        # TODO check what happens when loading state dict with different precision

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dion_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "dion":
                dion_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        dion_tasks = self._create_dion_tasks(dion_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(dion_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    @torch.no_grad()
    def synchronize_for_checkpoint(self):
        """
        Synchronize the internal optimizer states across the replicated mesh.

        Dion uses compressed gradient synchronization with decoupled momentum, which
        results in optimizer states diverging across the replicated data-parallel mesh.
        To ensure consistency of distributed checkpoints, we must manually synchronize
        the optimizer states before saving a checkpoint. If replicate_mesh is None or
        replicate_mesh_grad_sync is False, this function is a no-op.
        """

        if self._replicate_mesh is None or not self._replicate_mesh_grad_sync:
            # Nothing to do
            return

        # Get all tensors in optimizer states
        state_tensors: List[Tensor] = []
        for state in self.state.values():
            assert isinstance(state, dict)
            for val in state.values():
                if isinstance(val, Tensor):
                    state_tensors.append(val)

        # Heuristic to determine a reasonable batch size
        if self._inner_shard_mesh is not None:
            batch_size = self._inner_shard_mesh.size()
        elif self._outer_shard_mesh is not None:
            batch_size = self._outer_shard_mesh.size()
        else:
            batch_size = self._replicate_world_size

        # Batching allows for coalesced all-reduce
        for batch in create_param_batches(state_tensors, batch_size):
            batch = to_local(batch)
            reduced_batch = all_reduce_replicate_mesh(
                batch, self._replicate_mesh, return_dtensor=False
            )
            torch._foreach_copy_(batch, reduced_batch)

    def _create_dion_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Dion matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            oversample = torch.tensor(group["oversample"])

            # Split parameters in this param group by sharding
            split_param_dict = self._split_params_by_sharding(group_params)
            for sharding_type, split_params in split_param_dict.items():
                if not split_params:
                    continue

                # Pick the appropriate update function and batch size based on sharding
                # Batch size should equal the number of devices in the mesh
                if sharding_type == "inner_sharded":
                    dion_update_func = dion_update_fsdp_tp
                    batch_size = self._inner_shard_mesh.size()
                    use_dtensor = True
                elif sharding_type == "outer_sharded":
                    dion_update_func = dion_update_fsdp
                    batch_size = self._outer_shard_mesh.size()
                    use_dtensor = True
                elif sharding_type == "non_sharded":
                    dion_update_func = dion_update_ddp
                    batch_size = self._replicate_world_size
                    use_dtensor = False
                else:
                    raise RuntimeError("Unknown sharding type")

                # Create batches of parameters
                for params in create_param_batches(split_params, batch_size):
                    gradients = [p.grad for p in params]
                    states = [self._get_or_initialize_state(p, group) for p in params]
                    momentums = [s["momentum"] for s in states]
                    Qs = [s["Q"] for s in states]
                    param_config = self._get_dion_param_config(params[0])

                    if not use_dtensor:
                        params = to_local(params)
                        gradients = to_local(gradients)
                        momentums = to_local(momentums)
                        Qs = to_local(Qs)

                    yield AsyncTask(
                        dion_update_func(
                            X=pad_batch(params, batch_size),
                            G=pad_batch(gradients, batch_size),
                            M=pad_batch(momentums, batch_size),
                            Q=pad_batch(Qs, batch_size),
                            lr=lr,
                            mu=mu,
                            weight_decay=weight_decay,
                            epsilon=epsilon,
                            param_config=param_config,
                            replicate_mesh=self._replicate_mesh,
                            replicate_mesh_grad_sync=self._replicate_mesh_grad_sync,
                            oversample=oversample,
                        )
                    )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, group) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            # If replicate_mesh_grad_sync is True, the optimizer needs to all-reduce gradients
            # If replicate_mesh_grad_sync is False, gradients are already synchronized
            all_reduce_mesh = (
                self._replicate_mesh if self._replicate_mesh_grad_sync else None
            )

            yield AsyncTask(
                lion_update_allreduce_grad(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    replicate_mesh=all_reduce_mesh,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, group) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            # If replicate_mesh_grad_sync is True, the optimizer needs to all-reduce gradients
            # If replicate_mesh_grad_sync is False, gradients are already synchronized
            all_reduce_mesh = (
                self._replicate_mesh if self._replicate_mesh_grad_sync else None
            )

            yield AsyncTask(
                adamw_update_allreduce_grad(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                    replicate_mesh=all_reduce_mesh,
                )
            )

    def _get_or_initialize_state(self, param: Tensor, group: dict) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        algo = group["algorithm"]
        if not state:
            if algo == "dion":
                self._init_opt_state_dion(
                    param,
                    state,
                    rank_fraction=group["rank_fraction"],
                    rank_multiple_of=group["rank_multiple_of"],
                )
            elif algo == "adamw":
                self._init_opt_state_adam(param, state)
            elif algo == "lion" or algo == "clion":
                self._init_opt_state_momentum(param, state)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
        return state

    def _get_dion_param_config(self, x: Tensor) -> DionParamConfig:
        """
        Get the Dion-specific parameter configuration for a given tensor.
        If the configuration is not already initialized, it will be created.
        Lazy initialization is necessary because PyTorch allows new parameters
        to be added to the optimizer after it has been created.
        """
        if x in self._param_config:
            return self._param_config[x]

        if x.ndim > 2:
            raise NotImplementedError(
                f"Tensors with more than 2 dimensions are not supported. Got {x.ndim}D tensor."
            )

        # Check for allowed DeviceMesh and DTensor combinations
        # We only allow DTensor + DeviceMesh or regular Tensor + ProcessGroup
        using_device_mesh = (
            isinstance(self._replicate_mesh, DeviceMesh)
            or isinstance(self._outer_shard_mesh, DeviceMesh)
            or isinstance(self._inner_shard_mesh, DeviceMesh)
        )
        using_process_group = isinstance(self._replicate_mesh, ProcessGroup)
        if using_device_mesh and not isinstance(x, DTensor):
            raise TypeError("When using DeviceMesh, all parameters must be DTensor.")
        if using_process_group and isinstance(x, DTensor):
            raise TypeError(
                "When using DTensor parameters, the data parallel group must be specified by a DeviceMesh instead of ProcessGroup."
            )

        # State is initialized for both matrix and scalar parameters
        config = DionParamConfig()

        # By default, we transpose matrices so that dim0 >= dim1
        # This can change depending on sharding
        if x.ndim == 2:
            m, n = x.shape
            config.is_transposed = m < n

        # Detect sharding dimensions for DTensor
        if isinstance(x, DTensor) and x.ndim == 2:
            device_mesh = x.device_mesh
            placements = x.placements
            assert len(placements) == device_mesh.ndim

            dim_map = [None for _ in range(x.ndim)]

            for mesh_dim, placement in enumerate(placements):
                # StridedShard not allowed
                if _StridedShard is not None and isinstance(placement, _StridedShard):
                    raise NotImplementedError(
                        f"StridedShard is not supported. Ensure that FSDP and TP shard different dimensions of each matrix."
                    )

                # Skip non-sharded device mesh dimensions
                if not placement.is_shard():
                    continue
                tensor_dim = placement.dim

                # Check for double sharding on same tensor dimension
                if dim_map[tensor_dim] is not None:
                    raise RuntimeError(
                        f"Got double-sharded DTensor for tensor dimension {placement.dim}."
                    )
                dim_map[tensor_dim] = mesh_dim

                # Get global ranks corresponding to this mesh dimension
                mesh_dim_ranks = dist.get_process_group_ranks(
                    device_mesh.get_group(mesh_dim)
                )

                # Check if it matches the outer or inner shard ranks
                outer_sharded, inner_sharded = False, False
                if mesh_dim_ranks == self._outer_shard_ranks:
                    config.outer_shard_tensor_dim = tensor_dim
                    config.outer_shard_mesh_dim = mesh_dim
                    outer_sharded = True
                if mesh_dim_ranks == self._inner_shard_ranks:
                    config.inner_shard_tensor_dim = tensor_dim
                    config.inner_shard_mesh_dim = mesh_dim
                    inner_sharded = True

                # Check for double sharding on same mesh dimension
                if outer_sharded and inner_sharded:
                    raise RuntimeError(
                        "Cannot have outer and inner sharding over the same process group."
                    )

                # Check for sharding on unrecognized mesh dimension
                # Ignore edge case for single GPU "sharding" = Replicate()
                # Make sure to check that size(mesh_dim) > 1
                if (
                    device_mesh.size(mesh_dim) > 1
                    and not outer_sharded
                    and not inner_sharded
                ):
                    raise RuntimeError(
                        f"Got DTensor sharded on unrecognized {mesh_dim=}, which does not match outer_shard_mesh or inner_shard_mesh."
                    )

            # Set transpose so that orthogonalization happens over the inner sharding dimension
            # Standard Dion orthogonalizes over tensor dimension 0
            if config.inner_shard_tensor_dim == 0 or config.outer_shard_tensor_dim == 1:
                config.is_transposed = False
            # Transposed Dion orthogonalizes over tensor dimension 1
            if config.outer_shard_tensor_dim == 0 or config.inner_shard_tensor_dim == 1:
                config.is_transposed = True

        self._param_config[x] = config
        return config

    def _split_params_by_sharding(
        self, params: List[Tensor]
    ) -> Dict[str, List[Tensor]]:
        """
        Sort parameters into inner-sharded, outer-sharded, and non-sharded lists.
        This determines the parallelization strategy used to compute the update.
        The "inner sharding" dimension needs to use distributed orthogonalization.
        """
        inner_sharded = []
        outer_sharded = []
        non_sharded = []

        for p in params:
            config = self._get_dion_param_config(p)
            if config.inner_shard_mesh_dim is not None:
                inner_sharded.append(p)
            elif config.outer_shard_mesh_dim is not None:
                outer_sharded.append(p)
            else:
                non_sharded.append(p)

        return {
            "inner_sharded": inner_sharded,
            "outer_sharded": outer_sharded,
            "non_sharded": non_sharded,
        }

    def _init_opt_state_momentum(self, param: Tensor, state: Dict[str, Any]):
        # Create the momentum buffer
        # If param is DTensor, this will also be a DTensor
        state["momentum"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.momentum_dtype
        )

    def _init_opt_state_adam(self, param: Tensor, state: Dict[str, Any]):
        self._init_opt_state_momentum(param, state)
        state["variance"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.variance_dtype
        )

    def _init_opt_state_dion(
        self,
        param: Tensor,
        state: Dict[str, Any],
        rank_fraction: float,
        rank_multiple_of: int,
    ):
        """
        Initialize the optimizer state for Dion.
        This includes the momentum buffer and the Q matrix.

        The low-rank factor `r` is computed as `rank_fraction` * min(m, n),
        and rounded up to the next multiple of `rank_multiple_of`.
        """
        if param.ndim != 2:
            raise ValueError(
                f"Expected Dion parameters to be 2D matrix, but got {param.ndim}D. "
                f"For scalar parameters, set 'algorithm' to 'lion' or 'adamw' when creating param group."
            )

        param_config = self._get_dion_param_config(param)
        self._init_opt_state_momentum(param, state)

        # Compute the low-rank factor r
        m, n = param.shape
        r = rank_fraction * min(m, n)
        r = rank_multiple_of * math.ceil(r / rank_multiple_of)
        r = min(r, m, n)
        Q_shape = (m, r) if param_config.is_transposed else (n, r)

        # Set compressed_all_reduce based on if it saves communication cost
        # Otherwise we will all-reduce the gradient matrix instead
        if rank_fraction < 1 and (m + n) * r < m * n:
            param_config.compressed_all_reduce = True

        # Get dtype for Q
        if self._mixed_precision_config.Q_dtype is not None:
            Q_dtype = self._mixed_precision_config.Q_dtype
        else:
            Q_dtype = param.dtype

        if isinstance(param, DTensor):
            # Directly construct Q as DTensor
            # Shard(0) on outer sharding mesh and Shard(1) on inner sharding mesh
            placements = [Replicate() for _ in range(param.device_mesh.ndim)]
            if param_config.outer_shard_mesh_dim is not None:
                placements[param_config.outer_shard_mesh_dim] = Shard(0)
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Shard(1)
            param_config.Q_sharded_placements = tuple(placements)

            # Q is unsharded along the inner sharding dimension only
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Replicate()
                param_config.Q_inner_unsharded_placements = tuple(placements)
            else:
                # No inner sharding, so placements are the same as Q_sharded_placements
                param_config.Q_inner_unsharded_placements = None

            # DTensor RNG should automatically produce identical results across DP replicas
            Q = dtensor_randn(
                Q_shape,
                device_mesh=param.device_mesh,
                dtype=Q_dtype,
                placements=param_config.Q_sharded_placements,
            )

        else:
            # Make sure all DP ranks have the same Q
            Q = torch.randn(Q_shape, device=param.device, dtype=Q_dtype)
            self._replicate_mesh_broadcast(Q)

        state["Q"] = Q

    def _replicate_mesh_broadcast(self, tensor: Tensor):
        """
        Broadcast a tensor from rank 0 over the replicated data-parallel world.
        Tensor is modified in place.
        """
        if self._replicate_mesh is None:
            # No data parallelism used, do nothing
            pass
        elif isinstance(self._replicate_mesh, DeviceMesh):
            for group in self._replicate_mesh.get_all_groups():
                dist.broadcast(tensor, group=group, group_src=0)
        elif isinstance(self._replicate_mesh, ProcessGroup):
            dist.broadcast(tensor, group=self._replicate_mesh, group_src=0)
        else:
            raise TypeError(
                "Data parallel mesh must be either a DeviceMesh or ProcessGroup."
            )


def dion_update_ddp(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    Q: List[Tensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: DionParamConfig,  # shared for all params in batch
    replicate_mesh: Union[DeviceMesh, ProcessGroup, None] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for DDP.
    This function does not support sharded matrices.

    Batch size should equal the DDP world size. Each device will
    orthogonalize one full matrix in the batch.
    """
    assert param_config.outer_shard_mesh_dim is None
    assert param_config.outer_shard_tensor_dim is None
    assert param_config.inner_shard_mesh_dim is None
    assert param_config.inner_shard_tensor_dim is None

    # Get rank and world size
    if isinstance(replicate_mesh, DeviceMesh):
        world_size = replicate_mesh.size()
        device_rank = replicate_mesh.get_rank()
    elif isinstance(replicate_mesh, ProcessGroup):
        world_size = dist.get_world_size(replicate_mesh)
        device_rank = dist.get_rank(replicate_mesh)
    else:
        world_size = 1
        device_rank = 0
    assert (
        len(X) == world_size
    ), f"Batch size {len(X)} must match DDP world size {world_size}."

    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G = all_reduce_replicate_mesh(G, replicate_mesh, return_dtensor=False)
        yield

    # Add new gradient to momentum
    torch._foreach_add_(M, G)

    # Compute low-rank approximation of M = P @ Q^T
    # M_batch shape is (batch_size, m, n)
    # P_batch shape is (batch_size, m, r)
    # Q_batch shape is (batch_size, n, r)
    M_batch, Q_batch = tensor_list_to_batch(M, Q, param_config.is_transposed)
    P_batch = M_batch @ Q_batch

    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    if compressed_all_reduce and replicate_mesh is not None:
        # Synchronize P across all DDP ranks by reduce-scatter
        # Each rank will orthogonalize one full matrix in the batch
        P_single = funcol.reduce_scatter_tensor(
            P_batch,
            reduceOp="avg",
            scatter_dim=0,  # dim 0 = batch
            group=replicate_mesh,
        )
        yield
    else:
        # Gradients are already synchronized, P_batch is identical across DDP world
        # We can just take one matrix of the batch
        P_single = P_batch[device_rank : device_rank + 1]

    # Orthogonalize one matrix in the batch
    P_single = orthogonalize(P_single, oversample=oversample)

    # All gather orthogonal P_batch from the per-device single matrices
    if replicate_mesh is not None:
        P_batch = funcol.all_gather_tensor(
            P_single,
            gather_dim=0,  # dim 0 = batch
            group=replicate_mesh,
        )
        yield
    else:
        assert world_size == 1
        P_batch = P_single  # batch size is 1

    # M_batch shape is (batch_size, m, n)
    # P_batch shape is (batch_size, m, r)
    # R_batch shape is (batch_size, n, r)
    R_batch = M_batch.mT @ P_batch

    # Synchronize R across all DDP ranks by all-reduce
    if compressed_all_reduce and replicate_mesh is not None:
        R_batch = all_reduce_replicate_mesh(R_batch, replicate_mesh)
        yield

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    Q_batch = column_normalize(R_batch, epsilon=epsilon)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight decay and weight update
    # X = (1 - lr * weight_decay) * X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X,
        P_batch,
        Q_batch,
        alpha=-scaled_lr,
        beta=1 - lr * weight_decay,
        transpose=param_config.is_transposed,
    )

    # Update Q in place
    update_Q_matrix_(Q, Q_batch)


def dion_update_fsdp(
    X: List[DTensor],  # Model weights (modified in place)
    G: List[DTensor],  # Gradient
    M: List[DTensor],  # Momentum buffer (modified in place)
    Q: List[DTensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: DionParamConfig,  # shared for all params in batch
    replicate_mesh: Optional[DeviceMesh] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for FSDP2 sharding.
    This function only supports sharding over the outer shard mesh.

    Batch size should equal the outer shard mesh size. Each device along the
    outer shard mesh dimension will orthogonalize one full matrix in the batch.
    """
    assert param_config.inner_shard_mesh_dim is None
    assert param_config.inner_shard_tensor_dim is None

    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G = all_reduce_replicate_mesh(G, replicate_mesh)
        yield

    # Add new gradient to momentum
    torch._foreach_add_(M, G)

    # Compute low-rank approximation of M = P @ Q^T
    # M_batch shape is (batch_size, m, n/outer)
    # P_batch shape is (batch_size, m, r)
    # Q_batch shape is (batch_size, n/outer, r)
    M_batch, Q_batch = tensor_list_to_batch(M, Q, param_config.is_transposed)
    P_batch: DTensor = M_batch @ Q_batch

    # Get a single full matrix of the batch
    # Shard(0) = shard on batch dimension
    P_single = P_batch.redistribute(
        placements=[Shard(0) if p.is_partial() else p for p in P_batch.placements],
        async_op=True,
    )
    yield

    # If compressed_all_reduce is True, also average over replicate mesh
    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    if compressed_all_reduce and replicate_mesh is not None:
        P_single = all_reduce_replicate_mesh(P_single, replicate_mesh)
        yield

    # Orthogonalize one matrix in the batch
    P_single = orthogonalize(P_single, oversample=oversample)

    # All gather orthogonal P_batch from the per-device single matrices
    P_batch = P_single.redistribute(
        placements=[Replicate() for _ in P_single.placements], async_op=True
    )
    yield

    # M_batch shape is (batch_size, m, n/outer)
    # P_batch shape is (batch_size, m, r)
    # R_batch shape is (batch_size, n/outer, r)
    R_batch: DTensor = M_batch.mT @ P_batch

    # The contracting dimension of R = M.mT @ P should not be sharded
    # There should not be any Partial() placements, so no need to redistribute
    assert not any(p.is_partial() for p in R_batch.placements)

    # If compressed_all_reduce is True, also average over replicate mesh
    if compressed_all_reduce and replicate_mesh is not None:
        R_batch = all_reduce_replicate_mesh(R_batch, replicate_mesh)
        yield

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    if param_config.outer_shard_mesh_dim is not None:
        assert isinstance(R_batch, DTensor)
        assert R_batch.placements[param_config.outer_shard_mesh_dim].is_shard()

        # Compute per-shard squared sum and sum across shards
        R_sum_sq = local_column_sum_sq(R_batch)
        R_sum_sq = funcol.all_reduce(
            R_sum_sq,
            reduceOp="sum",
            group=(R_batch.device_mesh, param_config.outer_shard_mesh_dim),
        )
        yield

        Q_batch = column_normalize(
            R_batch,
            full_column_sum_sq=R_sum_sq,
            epsilon=epsilon,
        )

    else:
        # The sum dimension is not sharded, so we can normalize directly
        Q_batch = column_normalize(R_batch, epsilon=epsilon)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight decay and weight update
    # X = (1 - lr * weight_decay) * X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X,
        P_batch,
        Q_batch,
        alpha=-scaled_lr,
        beta=1 - lr * weight_decay,
        transpose=param_config.is_transposed,
    )

    # Update Q in place
    update_Q_matrix_(Q, Q_batch)


def dion_update_fsdp_tp(
    X: List[DTensor],  # Model weights (modified in place)
    G: List[DTensor],  # Gradient
    M: List[DTensor],  # Momentum buffer (modified in place)
    Q: List[DTensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: DionParamConfig,  # shared for all params in batch
    replicate_mesh: Optional[DeviceMesh] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for combined FSDP2 + TP.
    This function supports sharding over both outer and inner shard meshes.

    Batch size should equal the inner shard mesh size. The full matrix will not be
    unsharded for orthogonalization. Each device along the inner shard mesh dimension
    will compute low-rank QR and Cholesky decompositions for one matrix in the batch.
    """
    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G = all_reduce_replicate_mesh(G, replicate_mesh)
        yield

    # Add new gradient to momentum
    torch._foreach_add_(M, G)

    # Unshard Q along the inner sharding dimension
    if param_config.Q_inner_unsharded_placements is not None:
        # Sharded Q has shape (n/outer, r/inner)
        # Unsharded Q has shape (n/outer, r)
        Q_unshard = [
            q.redistribute(
                placements=param_config.Q_inner_unsharded_placements,
                async_op=True,
            )
            for q in Q
        ]
        yield
    else:
        # Q is not sharded along inner sharding dimension
        Q_unshard = Q

    # Compute low-rank approximation of M = P @ Q^T
    # M_batch shape is (batch_size, m/inner, n/outer)
    # P_batch shape is (batch_size, m/inner, r)
    # Q_batch shape is (batch_size, n/outer, r)
    M_batch, Q_batch = tensor_list_to_batch(M, Q_unshard, param_config.is_transposed)
    Q_unshard = None  # No longer needed, free memory
    P_batch: DTensor = M_batch @ Q_batch

    # All reduce P to get the sharded matrix multiplication result
    P_batch = P_batch.redistribute(
        placements=[Replicate() if p.is_partial() else p for p in P_batch.placements],
        async_op=True,
    )
    yield

    # If compressed_all_reduce is True, also average over replicate mesh
    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    if compressed_all_reduce and replicate_mesh is not None:
        P_batch = all_reduce_replicate_mesh(P_batch, replicate_mesh)
        yield

    # Orthogonalize P_batch
    P_batch = distributed_orthogonalize(
        P_batch, oversample, shard_mesh_dim=param_config.inner_shard_mesh_dim
    )

    # M_batch shape is (batch_size, m/inner, n/outer)
    # P_batch shape is (batch_size, m/inner, r)
    # R_batch shape is (batch_size, n/outer, r)
    R_batch: DTensor = M_batch.mT @ P_batch

    # All reduce R to get the sharded matrix multiplication result
    R_batch = R_batch.redistribute(
        placements=[Replicate() if p.is_partial() else p for p in R_batch.placements],
        async_op=True,
    )
    yield

    # If compressed_all_reduce is True, also average over replicate mesh
    if compressed_all_reduce and replicate_mesh is not None:
        R_batch = all_reduce_replicate_mesh(R_batch, replicate_mesh)
        yield

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    if param_config.outer_shard_mesh_dim is not None:
        assert isinstance(R_batch, DTensor)
        assert R_batch.placements[param_config.outer_shard_mesh_dim].is_shard()

        # Compute per-shard squared sum and sum across shards
        R_sum_sq = local_column_sum_sq(R_batch)
        R_sum_sq = funcol.all_reduce(
            R_sum_sq,
            reduceOp="sum",
            group=(R_batch.device_mesh, param_config.outer_shard_mesh_dim),
        )
        yield

        Q_batch = column_normalize(
            R_batch,
            full_column_sum_sq=R_sum_sq,
            epsilon=epsilon,
        )

    else:
        # The sum dimension is not sharded, so we can normalize directly
        Q_batch = column_normalize(R_batch, epsilon=epsilon)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight decay and weight update
    # X = (1 - lr * weight_decay) * X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X,
        P_batch,
        Q_batch,
        alpha=-scaled_lr,
        beta=1 - lr * weight_decay,
        transpose=param_config.is_transposed,
    )

    # Re-shard and update Q in place
    update_Q_matrix_(Q, Q_batch, param_config.Q_sharded_placements)


def all_reduce_replicate_mesh(
    G: Union[Tensor, List[Tensor]],
    replicate_mesh: Optional[DeviceMesh] = None,
    return_dtensor: bool = True,
    reduce_op: str = "avg",
) -> Union[Tensor, List[Tensor]]:
    """
    All-reduce a tensor or list of tensors across replicated data-parallel ranks.
    """
    if replicate_mesh is None:
        # No data parallelism, return original tensors unmodified
        return G

    if isinstance(G, Tensor):
        # Single tensor
        result_local = funcol.all_reduce(
            to_local(G),
            reduceOp=reduce_op,
            group=replicate_mesh,
        )
    else:
        # List of tensors, use coalesced all-reduce
        result_local = funcol.all_reduce_coalesced(
            to_local(G),
            reduceOp=reduce_op,
            group=replicate_mesh,
        )

    if return_dtensor:
        ref = G if isinstance(G, Tensor) else G[0]
        return dtensor_from_local(result_local, ref=ref)
    else:
        return result_local


def tensor_list_to_batch(
    M: List[Tensor],
    Q: List[Tensor],
    is_transposed: bool,
) -> Tuple[Tensor, Tensor]:
    """
    Convert a list of tensors M and Q into 3D batched tensors.
    Outputs M_batch and Q_batch will have dtype matching M.
    """
    # Transpose the momentum matrices if needed
    if is_transposed:
        M_batch = torch.stack([m.mT for m in M])
    else:
        M_batch = torch.stack(M)

    # Match dtype of Q and M
    Q_batch = torch.stack(Q).to(M_batch.dtype)

    return M_batch, Q_batch


def generate_random_sketch_matrix(
    P: Tensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
) -> Tensor:
    """
    Generate a random sketch matrix S for low-rank approximation.
    P is the input tensor with shape (batch_size, m, r).
    The sketch matrix S will have shape (batch_size, k, m),
    where k = round(oversample * r) to the next multiple of 128.
    """
    assert P.ndim >= 3, "P must have batch dimension"

    batch_size = P.shape[:-2]
    m = P.size(-2)
    r = P.size(-1)
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)

    if isinstance(P, DTensor):
        S_placements = list(P.placements)
        if shard_mesh_dim is not None:
            # Shard along tensor dimension -1 (size m dimension)
            S_placements[shard_mesh_dim] = Shard(P.ndim - 1)

        S = dtensor_randn(
            (*batch_size, k, m),
            device_mesh=P.device_mesh,
            dtype=P.dtype,
            placements=S_placements,
        )
        S = S * std

    else:
        # Regular tensor case
        if shard_mesh_dim is not None:
            raise TypeError("Must use DTensor parameters for sharded random sketch.")

        S = torch.empty((*batch_size, k, m), device=P.device, dtype=P.dtype).normal_(
            std=std
        )

    return S


# Graph break in torch.compile due to DTensor RNG
# @torch.compile()
def orthogonalize(P: Tensor, oversample: float = 1.25) -> Tensor:
    """
    Orthogonalize a batch of matrices.
    The input cannot be sharded along the matrix dimensions.
    If input is DTensor, the output will also be DTensor.
    """
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    P_local = to_local(P)

    if isinstance(P, DTensor):
        # Matrix dimensions (-2, -1) cannot be sharded
        assert not any(p.is_shard(P.ndim - 2) for p in P.placements)
        assert not any(p.is_shard(P.ndim - 1) for p in P.placements)

    # Standard QR is faster if matrix is square or wide
    if P.size(-2) <= P.size(-1):
        P_local, _ = torch.linalg.qr(P_local.to(dtype=torch.float32))

    # Randomized Cholesky QR
    else:
        # Must generate random sketch as DTensor for synchronized RNG
        S = generate_random_sketch_matrix(P, oversample)
        S_local = to_local(S)

        # Orthogonalize P using random sketch QR
        SP = S_local @ P_local
        _, R = torch.linalg.qr(SP.to(dtype=torch.float32), mode="r")
        P_local = torch.linalg.solve_triangular(
            R, P_local.to(dtype=torch.float32), upper=True, left=False
        )

        # Apply Cholesky QR to better orthogonalize
        PP = P_local.mT @ P_local  # always do float32 matrix multiply
        R, _ = torch.linalg.cholesky_ex(PP, upper=True)
        P_local = torch.linalg.solve_triangular(R, P_local, upper=True, left=False)

    return dtensor_from_local(
        P_local.to(original_dtype).contiguous(),
        ref=P,
    )


# Graph break in torch.compile due to DTensor RNG
# @torch.compile()
def distributed_orthogonalize(
    P: DTensor, oversample: float = 1.25, shard_mesh_dim: Optional[int] = None
) -> DTensor:
    """
    Orthogonalize a batch of sharded matrices.
    The input cannot be sharded along the last dimension.
    """
    assert isinstance(P, DTensor)
    assert not any(p.is_partial() for p in P.placements)
    assert not any(p.is_shard(P.ndim - 1) for p in P.placements)
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    original_placements = P.placements

    # Batch-sharded placement shards dimension 0 = batch
    # Each device gets one full matrix in the batch
    fully_replicated_placements = [Replicate() for _ in P.placements]
    batch_sharded_placements = fully_replicated_placements.copy()
    if shard_mesh_dim is not None:
        batch_sharded_placements[shard_mesh_dim] = Shard(0)

    # Standard QR is faster if matrix is square or wide
    if P.size(-2) <= P.size(-1):
        # Get one full matrix in the batch
        P_single = P.redistribute(
            placements=batch_sharded_placements,
        )  # this should do all-to-all

        # Compute Q matrix of QR decomposition
        Q_local, _ = torch.linalg.qr(
            P_single.to_local().to(dtype=torch.float32), mode="reduced"
        )

        # Convert back to DTensor and redistribute to original sharding
        P = dtensor_from_local(
            Q_local.to(original_dtype).contiguous(),
            ref=P_single,
        ).redistribute(
            placements=original_placements,
        )  # this should do all-to-all

    # Randomized Cholesky QR
    else:
        # Compute the random sketch matrix
        S = generate_random_sketch_matrix(P, oversample, shard_mesh_dim=shard_mesh_dim)
        SP: DTensor = S @ P

        # Get one full matrix in the batch
        SP_single = SP.redistribute(
            placements=batch_sharded_placements,
        )  # this should do reduce-scatter

        # Compute R matrix using QR decomposition
        _, R_local = torch.linalg.qr(
            SP_single.to_local().to(dtype=torch.float32), mode="r"
        )

        # Convert back to DTensor and get entire batch of R matrices
        R = dtensor_from_local(R_local, ref=SP_single).redistribute(
            placements=fully_replicated_placements,
        )  # this should do all-gather

        # Solve for orthogonalized batch of P matrix shards
        P_local = torch.linalg.solve_triangular(
            R.to_local(),  # already float32
            P.to_local().to(dtype=torch.float32),
            upper=True,
            left=False,
        )
        P = dtensor_from_local(P_local, ref=P)

        # Apply Cholesky QR to better orthogonalize P
        PP: DTensor = P.mT @ P

        # Get one full matrix in the batch
        PP_single = PP.redistribute(
            placements=batch_sharded_placements,
        )  # this should do reduce-scatter

        # Compute R matrix using Cholesky decomposition
        R_local, _ = torch.linalg.cholesky_ex(PP_single.to_local(), upper=True)

        # Convert back to DTensor and get entire batch of R matrices
        R = dtensor_from_local(R_local, ref=PP_single).redistribute(
            placements=fully_replicated_placements,
        )  # this should do all-gather

        # Solve for orthogonalized batch of P matrix shards
        P_local = torch.linalg.solve_triangular(
            R.to_local(),
            P.to_local(),
            upper=True,
            left=False,
        )

        # Convert back to DTensor and restore original dtype
        P = dtensor_from_local(P_local.to(original_dtype).contiguous(), ref=P)

    assert P.dtype == original_dtype, "Output dtype mismatch"
    assert P.placements == original_placements, "Output placements mismatch"
    return P


# @torch.compile(fullgraph=True)
def fix_all_zero_or_nan(
    P: Tensor,  # Output of power iteration
    R: Tensor,  # Output of power iteration
    Q_init: Tensor,  # Initial Q matrix
    B: Tensor,  # Buffer to check if all zeros
) -> Tuple[Tensor, Tensor]:
    """
    If input is all zero, P and R will be nan or all zero.
    We want to return the conditional expressions:

        if is_all_zero:
            P = torch.zeros_like(P)
            R = Q_init
        else:
            P = P
            R = R

    Here this is implemented without data-dependent control flow.
    To avoid additional communication, we handle sharded tensors independently.
    """
    B_local = to_local(B)
    is_all_zero = (B_local == 0).all(dim=(-2, -1), keepdim=True)
    not_all_zero = ~is_all_zero
    P_local = to_local(P).nan_to_num() * not_all_zero
    R_local = to_local(R).nan_to_num() * not_all_zero + to_local(Q_init) * is_all_zero
    P = dtensor_from_local(P_local, ref=P)
    R = dtensor_from_local(R_local, ref=R)
    return P, R


# @torch.compile(fullgraph=True)
def local_column_sum_sq(X: Tensor) -> Tensor:
    """
    Compute the per-column sum of squares of a tensor, or local shard of a DTensor.
    If the input has shape (m, n), the output will have shape (1, n).
    Regardless of input, the output will be a local tensor with float32 dtype.
    """
    X = to_local(X).to(dtype=torch.float32)
    # Sum over all rows to get one value per column
    X = X.square().sum(dim=-2, keepdim=True)
    return X


# @torch.compile(fullgraph=True)
def column_normalize(
    X: Tensor,
    full_column_sum_sq: Optional[Tensor] = None,
    epsilon: float = 1e-8,
) -> Tensor:
    """
    Normalize the columns of a tensor or local shard of a DTensor.
    If the input is a row-sharded DTensor, full_column_sum_sq must be provided.
    The computation is performed internally in float32 for numerical stability.
    """
    if isinstance(X, DTensor) and full_column_sum_sq is None:
        # To compute the per-column norm, we need to sum across all rows
        # If X is row sharded, we require full_column_sum_sq to be provided
        if any(p.is_shard(X.ndim - 2) for p in X.placements):
            raise RuntimeError("Cannot normalize row-sharded DTensor.")

    original_dtype = X.dtype
    X_local = to_local(X).to(dtype=torch.float32)

    # Compute per-column sum of squares if not provided
    if full_column_sum_sq is None:
        full_column_sum_sq = X_local.square().sum(dim=-2, keepdim=True)
    else:
        full_column_sum_sq = to_local(full_column_sum_sq).to(dtype=torch.float32)

    # Normalize each column
    full_column_norm = torch.sqrt(full_column_sum_sq)
    X_local = X_local / (full_column_norm + epsilon)

    X = dtensor_from_local(X_local.to(original_dtype), ref=X)
    return X


# @torch.compile(fullgraph=True)
def foreach_baddbmm_(
    X: List[Tensor],  # List of 2D matrices (modified in place)
    A: Tensor,  # 3D batch of matrices
    B: Tensor,  # 3D batch of matrices
    alpha: float = 1.0,
    beta: float = 1.0,
    transpose: bool = False,
):
    """
    Perform batch matrix multiplication and in-place addition.
    This is basically a foreach version of torch.baddbmm().

    If transpose is False, we compute X[i] = beta * X[i] + alpha * (A @ B.mT)[i].
    If transpose is True, we compute X[i] = beta * X[i] + alpha * (B @ A.mT)[i].
    """
    assert A.size(0) == B.size(0), "A and B must have the same batch size"
    assert len(X) == A.size(0), "len(X) must equal the batch dimension of A and B"

    if not transpose:
        update = A @ B.mT
    else:
        update = B @ A.mT

    # Convert DTensor to local tensor for foreach operations
    if isinstance(update, DTensor):
        if any(p.is_partial() for p in update.placements):
            raise NotImplementedError(
                "This function does not support DTensor matrix multiplication resulting in Partial() placements."
            )
        update = update.to_local()
        X = to_local(X)

    update = update.unbind(dim=0)  # Split batch into list of tensors
    update = torch._foreach_mul(update, alpha)  # Scale update by alpha
    torch._foreach_mul_(X, beta)  # Scale existing X by beta
    torch._foreach_add_(X, update)


# @torch.compile(fullgraph=True)
def update_Q_matrix_(
    Q: List[Tensor],  # Q matrix for power iteration (modified in place)
    Q_batch: Tensor,  # New Q matrix from orthogonalization
    Q_sharded_placements: Optional[Tuple[Placement]] = None,
):
    """
    Update the list of Q matrices in place with the new 3D stacked Q_batch.
    """
    if Q_sharded_placements is not None:
        # Increment all Shard() dimensions by 1 because of the batch dimension
        assert isinstance(Q_batch, DTensor)
        Q_batch_sharded_placements = list(Q_sharded_placements)

        for i in range(len(Q_batch_sharded_placements)):
            if Q_batch_sharded_placements[i].is_shard():
                Q_batch_sharded_placements[i] = Shard(
                    Q_batch_sharded_placements[i].dim + 1
                )

        # Redistribute Q_batch to sharded placements
        Q_batch = Q_batch.redistribute(
            placements=Q_batch_sharded_placements,
        )

    # Match dtype and convert to local tensor
    Q_init_dtype = Q[0].dtype
    Q_batch = to_local(Q_batch).to(Q_init_dtype)
    torch._foreach_copy_(to_local(Q), Q_batch.unbind(dim=0))


def adamw_update_allreduce_grad(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    replicate_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    AdamW optimizer algorithm with gradient all-reduce.
    """
    if replicate_mesh is not None:
        G = all_reduce_replicate_mesh(G, replicate_mesh, return_dtensor=False)
        yield
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)


def lion_update_allreduce_grad(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    replicate_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    Lion optimizer algorithm with gradient all-reduce.
    """
    if replicate_mesh is not None:
        G = all_reduce_replicate_mesh(G, replicate_mesh, return_dtensor=False)
        yield
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)

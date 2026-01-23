# source link:
# https://github.com/microsoft/dion/blob/main/dion/muon.py

import math

from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT

from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach, lion_update_foreach


class Muon(Optimizer):
    """
    Distributed Muon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."

            )

        # Cache for process groups by world_size to ensure consistency across all ranks
        # When different DTensors have different meshes but same world_size, we use the
        # FIRST process group we encounter to avoid NCCL communicator mismatches
        self._pg_cache: Dict[int, ProcessGroup] = {}
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        lion_groups = []
        adamw_groups = []

        # Debug logging for expert verification
        expert_param_count = 0
        muon_expert_count = 0
        adamw_expert_count = 0

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]

            # Check for expert parameters in this group
            for param in group["params"]:
                if hasattr(param, "_param_name"):
                    param_name = param._param_name
                    if self._is_expert_param_name(param_name):
                        expert_param_count += 1
                        if algo == "muon":
                            muon_expert_count += 1
                        elif algo == "adamw":
                            adamw_expert_count += 1

            if algo == "muon":
                muon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        # IMPORTANT: Process tasks grouped by process group to avoid deadlocks
        # with FSDP+EP where different params use different process groups
        muon_task_lists = self._create_muon_tasks_by_process_group(muon_groups)
        lion_tasks = list(self._create_lion_tasks(lion_groups))
        adamw_tasks = list(self._create_adamw_tasks(adamw_groups))

        # Run muon tasks grouped by process group to ensure all ranks in a
        # process group execute the same collectives at the same time.
        #
        # With FSDP+EP, different params use different process groups:
        # - Expert params use dp_mod_ep process group (local, no collectives)
        # - Non-expert params use dp_shard_cp process group (all_to_all)
        #
    
        pg_ids = list(muon_task_lists.keys())
        for pg_id in pg_ids:
            generators = muon_task_lists[pg_id]

            # Barrier BEFORE each group that uses collectives (not "local")
            # This ensures all ranks are synchronized before anyone starts this group
            if pg_id != "local":
                world_pg = dist.group.WORLD
                dist.barrier(group=world_pg)

            # Run generators sequentially
            for gen in generators:
                yield_count = 0
                for _ in gen:
                    yield_count += 1

        # Final barrier after all muon tasks to ensure all ranks are done
        dist.barrier(group=dist.group.WORLD)

        # Run lion and adamw tasks (no collectives needed)
        other_tasks = chain(lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(other_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _is_expert_param_name(self, name: str) -> bool:
        """Check if parameter name indicates it's an expert parameter."""
        expert_patterns = [
            "experts.",
            ".expert.",
            "expert_",
            "moe.expert",
            "shared_experts",
            "routed_experts",
            ".experts[",
            ".w1.",
            ".w2.",
            ".w3.",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in expert_patterns)

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _get_process_group_id(
        self, process_group: Optional[ProcessGroup], is_local: bool
    ) -> str:
        """
        Get a unique identifier for a process group.

        This is used to group tasks that use the same process group together,
        ensuring all ranks in that group execute the same collectives at the same time.

        Args:
            process_group: The process group (or None for single GPU)
            is_local: If True, this is a local-only operation with no collectives

        Returns:
            A string identifier that is consistent across all ranks in the same group
        """
        if is_local:
            # Local operations don't need synchronization, use a special ID
            return "local"

        if process_group is None:
            # Single GPU case
            return "single_gpu"

        # Use the process group's name or create an ID from its properties
        # The key insight: ranks in the same PG will have the same world_size
        # but different ranks. We use world_size to identify the PG type.
        pg_world_size = dist.get_world_size(process_group)
        # Also include a hash of the group to distinguish groups with same size
        # but different membership (e.g., multiple dp_mod_ep groups)
        try:
            pg_name = process_group.group_name if hasattr(process_group, 'group_name') else ""
        except Exception:
            pg_name = ""

        return f"pg_ws{pg_world_size}_{pg_name}"

    def _create_muon_tasks_by_process_group(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Dict[str, List[Generator]]:
        """
        Create Muon task generators grouped by process group to avoid deadlocks.

        With FSDP+EP, different parameters may use different process groups:
        - Expert params use dp_mod_ep process group
        - Non-expert params use dp_shard_cp process group

        To avoid deadlocks, we must ensure all ranks in a process group
        execute the same collectives at the same time. This method groups
        task generators by their process group so they can be executed together.

        IMPORTANT: We return generators, NOT AsyncTasks! AsyncTask.__init__ calls
        run() immediately, which starts executing the task before we're ready.
        We defer AsyncTask creation until execution time to ensure all ranks are
        synchronized when they start each collective operation.

        Returns:
            Dict mapping process group identifier to list of task generators
        """
        from collections import defaultdict

        # Group task GENERATORS by their process group (not AsyncTasks!)
        # This is critical: AsyncTask.__init__ runs the generator immediately,
        # which would start all-to-all before all ranks are ready.
        generators_by_pg: Dict[str, List[Generator]] = defaultdict(list)

        for generator, pg_id in self._create_muon_generators_with_pg_id(param_groups, algo_name):
            generators_by_pg[pg_id].append(generator)

        # Sort by pg_id to ensure consistent ordering across all ranks
        return dict(sorted(generators_by_pg.items()))

    def _create_muon_generators_with_pg_id(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Generator[Tuple[Generator, str], None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        (generator, process_group_id) tuples for grouping by process group.

        IMPORTANT: Yields generators, NOT AsyncTasks, to defer execution.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            nesterov = group["nesterov"]
            flatten = group["flatten"]
            adjust_lr = group["adjust_lr"]

            # Create batches of parameters of size self._world_size
            batch_idx = 0
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                sharded_mesh_dim = None
                sharded_tensor_dim = None
                batch_process_group = self._process_group
                batch_world_size = self._world_size
                batch_device_rank = self._device_rank
                is_batch_dim_sharded = False  # True for EP-style sharding on batch dimension

                if isinstance(params[0], DTensor):
                    param_mesh = params[0].device_mesh
                    param_ndim = params[0].ndim
                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and param_mesh.size(i) > 1
                    ]

                    # Separate batch dimension sharding from matrix dimension sharding
                    # For 3D+ tensors, dims 0..ndim-3 are batch dims, ndim-2..ndim-1 are matrix dims
                    # For 2D tensors, both dims are matrix dims
                    batch_dim_shards = []  # e.g., EP expert dimension
                    matrix_dim_shards = []  # e.g., FSDP or TP on matrix dims
                    for mesh_dim, placement in shard_placements:
                        tensor_dim = placement.dim
                        if param_ndim >= 3 and tensor_dim < param_ndim - 2:
                            batch_dim_shards.append((mesh_dim, placement))
                        else:
                            matrix_dim_shards.append((mesh_dim, placement))

                    # Handle batch dimension sharding (EP style)
                    if batch_dim_shards and not matrix_dim_shards:
                        # Only batch dimension sharding - process locally, no communication
                        is_batch_dim_sharded = True
                        #print(f"[Muon DEBUG]   -> batch_dim_sharded path (no communication)", flush=True)
                    elif matrix_dim_shards:
                        # Has matrix dimension sharding - needs all-to-all communication
                        if len(matrix_dim_shards) > 1:
                            raise NotImplementedError(
                                "Muon does not support parameters with multiple matrix-dimension shards."
                            )
                        sharded_mesh_dim = matrix_dim_shards[0][0]
                        sharded_tensor_dim = matrix_dim_shards[0][1].dim

                        raw_process_group = param_mesh.get_group(sharded_mesh_dim)
                        batch_world_size = dist.get_world_size(raw_process_group)
                        batch_process_group = raw_process_group

                        batch_device_rank = dist.get_rank(batch_process_group)

                        # Check for uneven sharding - if the tensor size isn't divisible by world_size,
                        # we can't use collective communication (all_gather requires equal-sized chunks)
                        global_size_on_shard_dim = params[0].size(sharded_tensor_dim)
                        if global_size_on_shard_dim % batch_world_size != 0:
                            # Treat as batch_dim_sharded - process locally without collectives
                            is_batch_dim_sharded = True
                            sharded_mesh_dim = None
                            sharded_tensor_dim = None
                            # Reset process group to default (won't be used for collectives)
                            batch_process_group = self._process_group
                            batch_world_size = self._world_size
                            batch_device_rank = self._device_rank

                # Create process group identifier for grouping tasks
                # Tasks with the same pg_id must be executed together to avoid deadlocks
                pg_id = self._get_process_group_id(batch_process_group, is_batch_dim_sharded)

                if is_batch_dim_sharded:
                    yield (
                        muon_update_batch_dim_sharded_async(
                            X=to_local(params),
                            G=to_local(gradients),
                            M=to_local(momentums),
                            lr=lr,
                            momentum=mu,
                            weight_decay=weight_decay,
                            epsilon=epsilon,
                            nesterov=nesterov,
                            flatten=flatten,
                            adjust_lr=adjust_lr,
                            newton_schulz_func=self._newton_schulz_func,
                        ),
                        pg_id,
                    )
                elif batch_world_size != self._world_size:
                    for i in range(0, len(params), batch_world_size):
                        sub_params = params[i : i + batch_world_size]
                        sub_gradients = gradients[i : i + batch_world_size]
                        sub_momentums = momentums[i : i + batch_world_size]

                        yield (
                            muon_update_batch_async(
                                X=pad_batch(sub_params, batch_world_size),
                                G=pad_batch(sub_gradients, batch_world_size),
                                M=pad_batch(sub_momentums, batch_world_size),
                                lr=lr,
                                momentum=mu,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                adjust_lr=adjust_lr,
                                device_rank=batch_device_rank,
                                world_size=batch_world_size,
                                shard_dim=sharded_tensor_dim,
                                process_group=batch_process_group,
                                newton_schulz_func=self._newton_schulz_func,
                            ),
                            pg_id,
                        )
                else:
                    # Standard case: matrix dimension sharding with matching world size, or non-sharded
                    yield (
                        muon_update_batch_async(
                            X=pad_batch(params, batch_world_size),
                            G=pad_batch(gradients, batch_world_size),
                            M=pad_batch(momentums, batch_world_size),
                            lr=lr,
                            momentum=mu,
                            weight_decay=weight_decay,
                            epsilon=epsilon,
                            nesterov=nesterov,
                            flatten=flatten,
                            adjust_lr=adjust_lr,
                            device_rank=batch_device_rank,
                            world_size=batch_world_size,
                            shard_dim=sharded_tensor_dim,
                            process_group=batch_process_group,
                            newton_schulz_func=self._newton_schulz_func,
                        ),
                        pg_id,
                    )
                batch_idx += 1

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
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
                )
            )


def muon_update_batch_dim_sharded_async(
    X: List[Tensor],  # Model weights (modified in place) - local tensors
    G: List[Tensor],  # Gradient - local tensors
    M: List[Tensor],  # Momentum buffer (modified in place) - local tensors
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    Muon update for batch-dimension sharded tensors (e.g., EP expert weights).

    When tensors are sharded on a batch dimension (like the expert dimension in MoE),
    each GPU owns different data (different experts) rather than shards of the same data.
    In this case:
    - Newton-Schulz orthogonalization treats the batch dimension independently
    - Each GPU processes ALL its local params without any communication
    - This is mathematically equivalent to orthogonalizing each expert's weights independently

    This function processes all params locally without all-to-all or all-gather.

    Optimized for CPU offloading with:
    - Double-buffered CUDA streams to overlap transfer and compute
    - Batched Newton-Schulz for fewer kernel launches
    - Single sync point at end (no intermediate cuda.synchronize())
    """
    # Check if we need CPU offloading (tensors are on CPU)
    original_device = G[0].device
    needs_gpu_transfer = original_device.type != "cuda"

    # Compute scaled learning rate upfront
    # Use the first tensor's shape (they should all be the same shape within a batch)
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    if needs_gpu_transfer:
        # PIPELINED MODE: Double-buffered streams for maximum overlap
        # Timeline: transfer[i+1] overlaps with compute[i] overlaps with writeback[i-1]
        cuda_device = torch.device("cuda")
        dtype = M[0].dtype
        n_tensors = len(X)

        # Mini-batch size for batched Newton-Schulz (fewer kernel launches)
        BATCH_SIZE = 4

        # Create streams: one for H2D transfers, one for compute, one for D2H transfers
        h2d_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()

        # Double buffer: prefetch next batch while computing current
        prefetch_data = None  # Will hold (g_batch, m_batch, x_batch, indices) for next iteration

        def prefetch_batch(start_idx):
            """Prefetch a batch of tensors to GPU (non-blocking)."""
            end_idx = min(start_idx + BATCH_SIZE, n_tensors)
            indices = list(range(start_idx, end_idx))
            with torch.cuda.stream(h2d_stream):
                g_batch = [G[i].to(dtype=dtype).to(cuda_device, non_blocking=True) for i in indices]
                m_batch = [M[i].to(cuda_device, non_blocking=True) for i in indices]
                x_batch = [X[i].to(cuda_device, non_blocking=True) for i in indices]
            return (g_batch, m_batch, x_batch, indices)

        def compute_batch(g_batch, m_batch, x_batch, indices):
            """Compute momentum update and Newton-Schulz on GPU."""
            with torch.cuda.stream(compute_stream):
                # Wait for H2D transfer to complete (lightweight stream sync)
                compute_stream.wait_stream(h2d_stream)

                u_batch = []
                for j in range(len(indices)):
                    g_gpu, m_gpu = g_batch[j], m_batch[j]
                    # Update momentum: M = mu * M + G
                    m_gpu.mul_(momentum)
                    m_gpu.add_(g_gpu)
                    # Compute U
                    if nesterov:
                        u_gpu = m_gpu * momentum + g_gpu
                    else:
                        u_gpu = m_gpu.clone()
                    u_batch.append(u_gpu.to(dtype=torch.bfloat16))

                # Batched Newton-Schulz: stack same-shape tensors for single kernel
                if len(u_batch) > 1 and all(u.shape == u_batch[0].shape for u in u_batch):
                    u_stacked = torch.stack(u_batch, dim=0)
                    u_stacked = muon_update_newton_schulz(u_stacked, newton_schulz_func, flatten, epsilon)
                    u_batch = list(u_stacked.unbind(0))
                else:
                    u_batch = [muon_update_newton_schulz(u, newton_schulz_func, flatten, epsilon) for u in u_batch]

                # Apply weight decay and update
                for j in range(len(indices)):
                    x_batch[j].mul_(1 - lr * weight_decay)
                    x_batch[j].sub_(u_batch[j] * adjusted_lr)

            return m_batch, x_batch

        def writeback_batch(m_batch, x_batch, indices):
            """Write results back to CPU (non-blocking)."""
            with torch.cuda.stream(d2h_stream):
                # Wait for compute to complete
                d2h_stream.wait_stream(compute_stream)
                for j, i in enumerate(indices):
                    M[i].copy_(m_batch[j], non_blocking=True)
                    X[i].copy_(x_batch[j], non_blocking=True)

        # Pipeline: prefetch first batch
        if n_tensors > 0:
            prefetch_data = prefetch_batch(0)

        # Main loop with double buffering
        for batch_start in range(0, n_tensors, BATCH_SIZE):
            # Get current batch (already prefetched)
            g_batch, m_batch, x_batch, indices = prefetch_data

            # Start prefetching NEXT batch (overlaps with current compute)
            next_start = batch_start + BATCH_SIZE
            if next_start < n_tensors:
                prefetch_data = prefetch_batch(next_start)

            # Compute current batch
            m_batch, x_batch = compute_batch(g_batch, m_batch, x_batch, indices)

            # Writeback current batch (overlaps with next iteration's prefetch/compute)
            writeback_batch(m_batch, x_batch, indices)

        # Single sync at end to ensure all D2H transfers complete
        torch.cuda.synchronize()

        yield  # Single yield to make this a generator
    else:
        # STANDARD GPU MODE: Process all tensors together (original behavior)
        U = muon_update_pre_orthogonalize(
            G=G,
            M=M,
            momentum=momentum,
            nesterov=nesterov,
        )

        # Orthogonalize each tensor locally
        # Newton-Schulz treats dim 0 as batch, processing each slice independently
        U = [
            muon_update_newton_schulz(
                u,
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )
            for u in U
        ]

        # Update model parameters with orthogonalized output
        muon_update_post_orthogonalize(
            X=X,
            U=U,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
        )

        yield  # Single yield to make this a generator


def muon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.

    Memory-optimized for CPU offloading: when tensors are on CPU, moves ALL computation
    to GPU (momentum update, all_to_all, Newton-Schulz, weight update) then copies back.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)
    assert len(X) == world_size

    # Check early if we're in CPU offloading mode
    G_local = to_local(G)
    M_local = to_local(M)
    X_local = to_local(X)
    original_device = M_local[0].device
    needs_gpu_transfer = original_device.type != "cuda"

    if needs_gpu_transfer:
        # ====== CPU OFFLOADING PATH: Do ALL computation on GPU ======
        # This avoids slow CPU foreach operations for momentum and weight updates
        cuda_device = torch.device("cuda")
        dtype = M_local[0].dtype

        # Transfer G, M to GPU for momentum update
        G_gpu = [g.to(dtype=dtype).to(cuda_device, non_blocking=True) for g in G_local]
        M_gpu = [m.to(cuda_device, non_blocking=True) for m in M_local]
        torch.cuda.synchronize()

        # Momentum update on GPU (equivalent to muon_update_pre_orthogonalize)
        torch._foreach_mul_(M_gpu, momentum)
        torch._foreach_add_(M_gpu, G_gpu)

        if nesterov:
            U_gpu = torch._foreach_mul(M_gpu, momentum)
            torch._foreach_add_(U_gpu, G_gpu)
        else:
            # U shares memory with M when not using nesterov
            U_gpu = M_gpu

        # Free G_gpu - no longer needed
        del G_gpu

        # Convert to bfloat16 for communication
        U_gpu = [u.to(dtype=torch.bfloat16) for u in U_gpu]

        # Get one whole matrix for each device to orthogonalize
        if shard_dim is not None:
            # Use all-to-all to transform from a batch of shards to a single whole matrix
            assert process_group is not None, "process_group must be provided for sharded DTensors"
            assert isinstance(X[0], DTensor), "X should contain DTensors"

            # Validation
            x0 = X[0]
            x0_mesh = x0.device_mesh
            x0_mesh_sizes = {name: x0_mesh.size(i) for i, name in enumerate(x0_mesh.mesh_dim_names)}
            assert (
                X[0].size(shard_dim) % world_size == 0
            ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}. " \
               f"Tensor info: global_shape={tuple(X[0].shape)}, local_shape={X[0].to_local().shape}, " \
               f"mesh={X[0].device_mesh.mesh_dim_names}, mesh_sizes={x0_mesh_sizes}, placements={X[0].placements}"

            # Make contiguous for all_to_all
            U_gpu = [u.contiguous() for u in U_gpu]

            # First all_to_all: batch of shards -> single whole matrix
            single_matrix_shards = [torch.empty_like(U_gpu[0]) for _ in range(world_size)]
            dist.all_to_all(single_matrix_shards, U_gpu, group=process_group)
            del U_gpu

            yield

            # Concatenate shards to form whole matrix
            single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
            del single_matrix_shards

            # Newton-Schulz orthogonalization (on GPU)
            single_matrix = muon_update_newton_schulz(
                single_matrix,
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )

            # Split result back into shards
            orth_shards = [
                x.contiguous()
                for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
            ]
            del single_matrix

            # Second all_to_all to redistribute orthogonalized shards
            U_orth_gpu = [torch.empty_like(orth_shards[0]) for _ in range(world_size)]
            dist.all_to_all(U_orth_gpu, orth_shards, group=process_group)
            del orth_shards

            yield

        else:
            # Matrices are not sharded, orthogonalize directly
            single_matrix = U_gpu[device_rank]

            single_matrix = muon_update_newton_schulz(
                single_matrix,
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )

            if process_group is not None and process_group.size() > 1:
                U_orth_gpu = [torch.empty_like(single_matrix) for _ in range(world_size)]
                work = dist.all_gather(
                    U_orth_gpu, single_matrix.contiguous(), group=process_group, async_op=True
                )
                yield
                work.wait()
                del single_matrix
            else:
                assert world_size == 1
                U_orth_gpu = [single_matrix]

        # Compute scaled learning rate (use full tensor shape from X[0])
        if adjust_lr is None:
            adjusted_lr = lr
        elif adjust_lr == "spectral_norm":
            adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape)
        elif adjust_lr == "rms_norm":
            adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape)
        else:
            raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

        # Transfer X to GPU for weight update
        X_gpu = [x.to(cuda_device, non_blocking=True) for x in X_local]
        torch.cuda.synchronize()

        # Weight update on GPU (equivalent to muon_update_post_orthogonalize)
        torch._foreach_mul_(X_gpu, 1 - lr * weight_decay)
        U_scaled = torch._foreach_mul(U_orth_gpu, adjusted_lr)
        torch._foreach_sub_(X_gpu, U_scaled)
        del U_scaled, U_orth_gpu

        # Copy M and X back to CPU
        for i in range(world_size):
            M_local[i].copy_(M_gpu[i], non_blocking=True)
            X_local[i].copy_(X_gpu[i], non_blocking=True)

        torch.cuda.synchronize()
        del M_gpu, X_gpu

    else:
        # ====== STANDARD GPU PATH ======
        # Update momentum and compute the inputs for orthogonalization
        U = muon_update_pre_orthogonalize(
            G=G_local,
            M=M_local,
            momentum=momentum,
            nesterov=nesterov,
        )

        # Get one whole matrix for each device to orthogonalize
        # JQ: This is the N sequential gather version
        # if shard_dim is not None:
        #     # Use all-to-all to transform from a batch of shards to a single whole matrix
        #     # https://www.essential.ai/blog/infra
        #     assert (
        #         process_group is not None
        #     ), "process_group must be provided for sharded DTensors"
        #     assert isinstance(X[0], DTensor), "X should contain DTensors"
        #     assert not isinstance(U[0], DTensor), "U should contain local shards"

        #     # Debug: print full tensor info before the divisibility check
        #     x0 = X[0]
        #     x0_mesh = x0.device_mesh
        #     x0_mesh_sizes = {name: x0_mesh.size(i) for i, name in enumerate(x0_mesh.mesh_dim_names)}

        #     assert (
        #         X[0].size(shard_dim) % world_size == 0
        #     ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}. " \
        #     f"Tensor info: global_shape={tuple(X[0].shape)}, local_shape={X[0].to_local().shape}, " \
        #     f"mesh={X[0].device_mesh.mesh_dim_names}, mesh_sizes={x0_mesh_sizes}, placements={X[0].placements}"

        #     # Allocate buffers to receive shards of one whole matrix from other devices
        #     single_matrix_shards = [torch.empty_like(u) for u in U]

        #     # Redistribute the shards to form one unique full tensor on each device
        #     # Sync CUDA before collective to ensure all prior GPU ops are complete
        #     # This can prevent NCCL hangs due to async GPU operations
        #     torch.cuda.synchronize()

        #     # N sequential all_gathers - only keep result for our assigned param
        #     single_matrix_shards = None
        #     for param_idx in range(world_size):
        #         # Allocate output buffer for this all_gather
        #         gathered = [torch.empty_like(U[param_idx]) for _ in range(world_size)]

        #         # All ranks send their shard of param_idx
        #         dist.all_gather(gathered, U[param_idx].contiguous(), group=process_group)

        #         # Only keep if this is our assigned parameter
        #         if param_idx == device_rank:
        #             single_matrix_shards = gathered
        #         # Otherwise 'gathered' goes out of scope and memory can be freed

        #     yield

        #     # Concatentate shards to form a whole matrix to orthogonalize
        #     single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        #     single_matrix = muon_update_newton_schulz(
        #         single_matrix,
        #         newton_schulz_func=newton_schulz_func,
        #         flatten=flatten,
        #         epsilon=epsilon,
        #     )

        #     # Split result back into shards
        #     # Contiguous is needed for communication to work correctly
        #     orth_shards = [
        #         x.contiguous()
        #         for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        #     ]

        #     # N sequential all_gathers - collect results as we go
        #     for shard_idx in range(world_size):
        #         # Allocate output buffer for this all_gather
        #         gathered = [torch.empty_like(orth_shards[shard_idx]) for _ in range(world_size)]

        #         # All ranks send their shard at index shard_idx
        #         dist.all_gather(gathered, orth_shards[shard_idx].contiguous(), group=process_group)

        #         # gathered[r] = rank r's orth_shards[shard_idx] = O^r_{shard_idx}
        #         # We need U[r] = O^r_{device_rank}
        #         # So when shard_idx == device_rank: U[r] = gathered[r] for all r
        #         if shard_idx == device_rank:
        #             for r in range(world_size):
        #                 U[r].copy_(gathered[r])

        #     yield

        # Get one whole matrix for each device to orthogonalize
        if shard_dim is not None:
            assert process_group is not None, "process_group must be provided for sharded DTensors"
            assert isinstance(X[0], DTensor), "X should contain DTensors"
            assert not isinstance(U[0], DTensor), "U should contain local shards"

            x0 = X[0]
            x0_mesh = x0.device_mesh
            x0_mesh_sizes = {name: x0_mesh.size(i) for i, name in enumerate(x0_mesh.mesh_dim_names)}
            assert (
                X[0].size(shard_dim) % world_size == 0
            ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}. " \
               f"Tensor info: global_shape={tuple(X[0].shape)}, local_shape={X[0].to_local().shape}, " \
               f"mesh={X[0].device_mesh.mesh_dim_names}, mesh_sizes={x0_mesh_sizes}, placements={X[0].placements}"

            # Sync CUDA before collective to prevent NCCL hangs from async GPU ops
            torch.cuda.synchronize()

            single_matrix_shards = [torch.empty_like(U[0]) for _ in range(world_size)]
            dist.all_to_all(single_matrix_shards, [u.contiguous() for u in U], group=process_group)

            yield

            single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
            del single_matrix_shards

            single_matrix = muon_update_newton_schulz(
                single_matrix,
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )

            orth_shards = [
                x.contiguous()
                for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
            ]
            del single_matrix

            output_shards = [torch.empty_like(orth_shards[0]) for _ in range(world_size)]
            dist.all_to_all(output_shards, orth_shards, group=process_group)
            del orth_shards

            for i in range(world_size):
                U[i].copy_(output_shards[i])
            del output_shards

            yield

        else:
            single_matrix = U[device_rank]
            assert not isinstance(single_matrix, DTensor)

            single_matrix = muon_update_newton_schulz(
                single_matrix,
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )

            if process_group is not None and process_group.size() > 1:
                U_gathered = [torch.empty_like(single_matrix) for _ in range(world_size)]
                work = dist.all_gather(
                    U_gathered, single_matrix.contiguous(), group=process_group, async_op=True
                )
                yield
                work.wait()
                del single_matrix
                U = U_gathered
            else:
                assert world_size == 1
                U = [single_matrix]

        # Compute scaled learning rate
        if adjust_lr is None:
            adjusted_lr = lr
        elif adjust_lr == "spectral_norm":
            adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape)
        elif adjust_lr == "rms_norm":
            adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape)
        else:
            raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

        # Update model parameters with orthogonalized output
        muon_update_post_orthogonalize(
            X=X_local,
            U=U,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
        )


def adamw_update_foreach_async(
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
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach AdamW update.
    """
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def lion_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach Lion update.
    """
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
    yield


# @torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


# @torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    Always normalizes to 3D before calling newton_schulz_func to avoid torch.compile recompilations.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    # Always ensure 3D input to newton_schulz_func to avoid torch.compile recompilations
    # due to rank mismatch (2D vs 3D tensors triggering separate traces)
    added_batch_dim = False
    if X.ndim == 2:
        X = X.unsqueeze(0)  # Add batch dimension: [M, N] -> [1, M, N]
        added_batch_dim = True

    result = newton_schulz_func(X, epsilon=epsilon)

    if added_batch_dim:
        result = result.squeeze(0)  # Remove batch dimension: [1, M, N] -> [M, N]

    return result.reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape):
    # Adjust learning rate for constant element-wise RMS norm
    # https://arxiv.org/abs/2502.16982
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    fan_out, fan_in = param_shape[:2]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


# @torch.compile(fullgraph=True)
def _is_expert_param_name_helper(name: str) -> bool:
    """Helper function to check if parameter name indicates it's an expert parameter."""
    expert_patterns = [
        "experts.",
        ".expert.",
        "expert_",
        "moe.expert",
        "shared_experts",
        "routed_experts",
        ".experts[",
        ".w1.",
        ".w2.",
        ".w3.",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in expert_patterns)


def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

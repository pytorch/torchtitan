from dataclasses import dataclass, field

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from src.logging import logger
from src.utils import device_type

__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    etp: int
    world_size: int

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None

    # ? ? a validation logic after the initializtion
    def __post_init__(self):
        self._validate()

    # ? ? actual validation logic
    def _validate(self):
        """Validate parallelism degrees and infer ``dp_shard`` if unset.

        Checks that every parallelism degree is ``>= 1`` (``dp_shard`` may be
        ``-1`` as a sentinel for auto-inference), then ensures the product of
        the rank-consuming dimensions equals ``world_size``:

            dp_replicate * dp_shard * cp * tp * pp == world_size

        Note that ``ep`` is not part of this product — expert parallelism
        reuses ranks from other dimensions rather than claiming its own
        slice of the world.

        If ``dp_shard == -1``, it is inferred as::

            dp_shard = world_size // (dp_replicate * cp * tp * pp)

        so that the product equation above holds. Both ``self.dp_shard`` and
        the local ``dp_shard`` are updated so subsequent assertions in this
        method see the resolved value.

        Raises:
            AssertionError: If any degree is invalid, or if the product of
                degrees does not equal ``world_size``.
        """
        dp_replicate, dp_shard, cp, tp, pp, ep, etp = (
            self.dp_replicate,  # ? * classic ddp
            self.dp_shard,  # ? * fsdp sharding
            self.cp,  # ? * context parallelism
            self.tp,  # ? * tensor parallelism
            self.pp,  # ? * pipeline parallelism
            self.ep,  # ? * expert parallelism
            self.etp,  # ? * expert tensor parallelism
        )
        # ? ? assert that all parallelism degreee needs to be >= 1
        for d in (dp_replicate, cp, tp, pp, ep, etp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (
                dp_replicate * cp * tp * pp
            )  # ? ? infer dp_shard from world size and other parallelism degrees
        assert dp_shard >= 1

        # ?? notice the ep does not account in the rank consumption because it reuses the same ranks as other parallelism dimensions, instead of claiming its own slice of the world
        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        # ?! not wired in
        if ep > 1:
            assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"

    def _mesh_exist(self, name: str, degree: int) -> bool:
        if name == "efsdp":
            # ? We always keep the efsdp if EP is larger than 1 because we need
            # ? FSDP wrapping to help the MoE layers do mixed precision training.
            return True if self.ep > 1 else False
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        """
        Build the device mesh with the required mesh dimensions.

        The following mesh dimensions will be created:

            pp:      Pipeline Parallelism (PP).
            batch:   Used by data loading to determine the global batch size and which
                        part of the data each rank should read. This dimension includes both
                        ``dp_replicate`` and ``dp_shard``. The backend is set to ``fake`` for
                        this dimension to avoid unnecessary process group creation.
            loss:    Used by all-reduce when computing the loss. Includes ``dp_replicate``,
                        ``dp_shard``, and ``cp`` degrees, as all of them parallelize the data,
                        essentially require the weight gradients reduction.
            dp_replicate: For DDP or HSDP replicate dimension.
            fsdp:    For FSDP dimension. This includes ``dp_shard`` and ``cp``. Note that
                        we always assume that when ``cp`` is used, FSDP is also applied to
                        utilize its weight all-gather and gradients reduce_scatter even if
                        there may be no data parallelism (e.g., global batch size is 1).
            cp:      Context Parallelism (CP).
            tp:      Tensor Parallelism (TP).
            ep:      Expert Parallelism (EP).
            efsdp:   FSDP in the EP region.
            etp:     TP in the EP region.

        Note: Most dimensions above are created by unflattening the world mesh, except for loss,
        which is created by flattening the batch and cp dimensions.
        This API performs the following unflatten operations from the world mesh:

            ["pp", "batch", "cp", "tp"]  #? dataloading_mesh
            ["pp", "dp_replicate", "fsdp", "tp"]  #? dense_mesh
            ["pp", "dp_replicate", "efsdp", "ep", "etp"]  #? sparse_mesh

        Note: DeviceMesh currently recreates the process group for each dimension.
        It should share the process group for the same dim group to avoid unnecessary
        process group creation. We can also use Fake to achieve a similar goal.
        However, using Fake to avoid redundancy messing up the code. We only use Fake
        when it is necessary. For now, we just let DeviceMesh create redundant process
        group and wait for DeviceMesh to fix the issue.
        """

        def unflatten_mesh(
            world_mesh: DeviceMesh,
            dim_names: tuple[str, ...],
            dim_degrees: tuple[int, ...],
        ):
            """Reshape the flat world mesh into a named N-D mesh, skipping process
            group creation for axes that don't need real communication.

            Takes the 1-D ``world_mesh`` and unflattens it into an N-D view with
            the given named axes. The product of ``dim_degrees`` must equal
            ``world_size``.

            To avoid wasting startup time and GPU memory on unused NCCL process
            groups, each axis is marked with the ``"fake"`` backend when it does
            not need real collectives. An axis is faked when either:

            - ``_mesh_exist`` returns ``False`` for it (degree == 1 for normal
                dims, with ``efsdp`` as a special case that is always real when
                ``ep > 1`` to support MoE mixed-precision FSDP wrapping), or
            - the axis is ``"batch"``, which is only used by the dataloader to
                derive per-rank data slices and never runs a collective.

            Faked axes still appear in the returned mesh and can be indexed by
            name, but attempting a collective along them will error. This lets
            downstream code use a uniform mesh structure (e.g. always writing
            ``mesh["ep"]``) without paying the real-group cost for parallelism
            dimensions that are disabled in the current run.

            Args:
                world_mesh: The flat 1-D mesh containing all ranks, with a single
                    axis named ``"world"``.
                dim_names: Names for the new axes, in order.
                dim_degrees: Sizes for the new axes, in the same order as
                    ``dim_names``. Must multiply to ``world_size``.

            Returns:
                An N-D ``DeviceMesh`` shaped as ``dim_degrees`` with axes named
                ``dim_names``. Real NCCL process groups exist only along axes
                that were not marked fake.
            """
            backend_override = {}
            for name, degree in zip(dim_names, dim_degrees, strict=True):
                if (not self._mesh_exist(name, degree)) or name == "batch":
                    backend_override[name] = "fake"

            # ? ? the actual call to flatten the world mesh
            # ? ? the first dimension needs to be 0 as the world mesh is 1D
            return world_mesh._unflatten(
                0, dim_degrees, dim_names, backend_override=backend_override
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}, etp={self.etp}"
        )

        batch = (
            self.dp_replicate * self.dp_shard
        )  # ? ? all ranks that see different data samples
        fsdp = (
            self.dp_shard * self.cp
        )  # ? ? from the weight communication perspective, the fsdp and cp are the same, because we always use the all-gather to get the model weight and reduce-scatter to get the gradient
        efsdp = fsdp * self.tp // (self.etp * self.ep)

        # ? ? first a flat device mesh for all ranks
        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )

        # ?? Dataloading view of the world: (pp, batch, cp, tp).
        # ??
        # ?? This mesh is a pure coordinate system — no collectives run on it. Each
        # ?? rank reads its own coordinates to decide which slice of the global
        # ?? batch to load from disk. Axis roles from the dataloader's perspective:
        # ??
        # ??   pp    - same data across pipeline stages (stage 0 reads; later
        # ??           stages receive activations from their predecessor).
        # ??   batch - different samples. Folds dp_replicate and dp_shard together,
        # ??           since the loader doesn't care whether a rank replicates or
        # ??           shards weights — only that it needs a distinct data slice.
        # ??   cp    - different sequence chunks of the same samples.
        # ??   tp    - same data across TP peers (TP shards weights inside a layer,
        # ??           not inputs). Included to satisfy the shape = world_size
        # ??           constraint; unflatten would fail without it when tp > 1.
        # ??
        # ?? EP/ETP do not appear here because they don't consume their own rank
        # ?? budget — EP ranks are TP ranks relabeled for the MoE region, and the
        # ?? dataloader would load identical data along an EP axis anyway.
        # ??
        # ?? Every axis ends up with the "fake" backend (batch is always faked,
        # ?? the rest are size-1 or faked on demand), so this mesh costs nothing
        # ?? in process groups — it exists purely to hand each rank its
        # ?? (batch, cp) coordinate pair.
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "batch", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )

        # ? Loss-reduction view: flatten (batch, cp) per (pp, tp) position.
        # ?
        # ? Example layout with pp=2, batch=2, cp=2, tp=2 (16 ranks total).
        # ? dataloading_mesh groups ranks as (pp, batch, cp, tp); we slice out
        # ? the (batch, cp) plane for each (pp, tp) coord and flatten it:
        # ?
        # ?   pp=0, tp=0:             pp=0, tp=1:
        # ?          cp=0  cp=1              cp=0  cp=1
        # ?   b=0:    r0    r2         b=0:   r1    r3
        # ?   b=1:    r4    r6         b=1:   r5    r7
        # ?          └──── flatten ────────── flatten ────┐
        # ?            group: [r0,r2,r4,r6]    group: [r1,r3,r5,r7]
        # ?
        # ?   pp=1, tp=0:             pp=1, tp=1:
        # ?          cp=0  cp=1              cp=0  cp=1
        # ?   b=0:    r8    r10        b=0:   r9    r11
        # ?   b=1:    r12   r14        b=1:   r13   r15
        # ?            group:[r8,r10,r12,r14]  group:[r9,r11,r13,r15]
        # ?
        # ? Result: 4 independent loss_mesh groups of size 4. Each rank reduces
        # ? only with the 3 others sharing its (pp, tp) coords — the ones that
        # ? hold the same parameters but processed different (batch, cp) data.
        # ?
        # ? These are the axes where ranks share parameters but processed
        # ? different data, so their partial gradients must be summed. pp and
        # ? tp are excluded — ranks there hold disjoint parameters (different
        # ? layers, different weight shards), so there's nothing to sum across.
        # ? Derived from dataloading_mesh to reuse its rank grouping; _flatten
        # ? creates the real NCCL group (batch alone is faked in dataloading_mesh).
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")

        # ? Dense parameter view: (pp, dp_replicate, fsdp, tp). Used by all
        # ? non-MoE layers (attention, dense FFN, embeddings, output head) for
        # ? parameter storage and training collectives:
        # ?
        # ?  pp            - pipeline send/recv between layer stages.
        # ?   dp_replicate  - DDP/HSDP all-reduce on gradients (when > 1).
        # ?   fsdp          - FSDP all-gather on weights (forward) and
        # ?                   reduce-scatter on gradients (backward).
        # ?                   = dp_shard * cp, since CP ranks also use FSDP's
        # ?                   weight-gather + grad-reduce pattern.
        # ?   tp            - tensor-parallel all-reduces inside sharded
        # ?                   matmuls within a layer.
        # ?
        # ? Splits dp_replicate and fsdp into separate axes (rather than
        # ? combining them like dataloading_mesh's `batch`) because their
        # ? collectives are different — replicate does all-reduce on grads,
        # ? fsdp does all-gather/reduce-scatter on weights and grads.
        dense_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "fsdp", "tp"),
            (self.pp, self.dp_replicate, fsdp, self.tp),
        )

        # ? Sparse (MoE) parameter view: (pp, dp_replicate, efsdp, ep, etp).
        # ? Used by MoE expert layers for parameter storage and training
        # ? collectives. Holds the same ranks as dense_mesh, re-organized:
        # ? where dense_mesh has (fsdp, tp), sparse_mesh has (efsdp, ep, etp),
        # ? carving experts out of the same rank pool via:
        # ?
        # ?     efsdp = fsdp * tp // (etp * ep)
        # ?
        # ? This is why EP doesn't appear in the world_size = dp_rep * dp_shard
        # ? * cp * tp * pp equation — EP and ETP reuse ranks from the fsdp*tp
        # ? pool, shrinking the expert-FSDP shard size rather than consuming
        # ? new ranks.
        # ?
        # ? Axis roles:
        # ?   pp            - pipeline send/recv (shared with dense_mesh).
        # ?   dp_replicate  - DDP/HSDP all-reduce on expert gradients.
        # ?   efsdp         - FSDP all-gather / reduce-scatter on a single
        # ?                   expert's weights.
        # ?   ep            - expert parallelism: different ranks hold
        # ?                   different experts. All-to-all in forward/backward
        # ?                   routes tokens to the right rank.
        # ?   etp           - tensor parallelism inside a single expert
        # ?                   (expected to be tp or 1; enforced in _validate).
        sparse_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "efsdp", "ep", "etp"),
            (self.pp, self.dp_replicate, efsdp, self.ep, self.etp),
        )

        # ? the final product: mesh look up tables
        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": dense_mesh,
            "sparse": sparse_mesh,
        }
        self._meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh,
            "dp_replicate": dense_mesh["dp_replicate"],
            "fsdp": dense_mesh["fsdp"],
            "cp": dataloading_mesh["cp"],
            "tp": dataloading_mesh["tp"],
            "ep": sparse_mesh["ep"],
            "efsdp": sparse_mesh["efsdp"],
            "etp": sparse_mesh["etp"],
        }

        # ? Validate mesh sizes
        self._validate_meshes()

        logger.info(
            f"Successfully created meshes with active dimensions: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        """Assert each registered 1-D sub-mesh has the size implied by the config.

        After ``build_mesh`` populates ``self._meshes``, each axis should
        have a specific degree derived purely from the parallelism config
        (``self.pp``, ``self.dp_shard``, etc.). This method recomputes the
        expected degree for every axis and asserts the actual mesh size
        matches.

        The check catches bugs in:
        - the derived-degree arithmetic inside ``build_mesh`` (e.g. the
            ``efsdp = fsdp * tp // (etp * ep)`` formula),
        - which global mesh a given axis was pulled from when populating
            ``_meshes``,
        - and anything else that would produce a mesh whose shape
            doesn't match the config.

        Catching the mismatch here is much cheaper than debugging the
        resulting NCCL hangs or mis-sized collectives at training time.

        Raises:
            AssertionError: If any axis's actual size differs from the
                config-derived expected size.
        """
        expected_sizes = {
            "pp": self.pp,
            "batch": self.dp_replicate * self.dp_shard,
            "loss": self.dp_replicate * self.dp_shard * self.cp,
            "dp_replicate": self.dp_replicate,
            "fsdp": self.dp_shard * self.cp,
            "cp": self.cp,
            "tp": self.tp,
            "ep": self.ep,
            "efsdp": self.dp_shard * self.cp * self.tp // (self.etp * self.ep),
            "etp": self.etp,
        }

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._meshes[mesh_name].size()
            assert actual_size == expected_size, (
                f"Mesh '{mesh_name}' has unexpected size: "
                f"expected {expected_size}, got {actual_size}"
            )

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Get a device mesh by dimension name(s), returning None if not enabled.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s), or None if:
            - The dimension size is 1 (parallelism not enabled)
            - The dimension doesn't exist (except efsdp which can exist even if size is 1 when ep > 1)

        Raises:
            ValueError: If the requested dimension name(s) is not valid.
        """

        # ? lazy initialization
        if not self._meshes:
            self.build_mesh()

        # ? unifiy the input format
        if isinstance(dims, str):
            dims = [dims]

        # ? validate the input dimension names
        for mesh_name in dims:
            if mesh_name not in self._meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. "
                    f"Valid dimensions are: {list(self._meshes.keys())}"
                )

        # ? check if any of the requested dimensions is not enabled (size = 1 or efsdp when ep=1), return None
        if any(not self._mesh_exist(dim, self._meshes[dim].size()) for dim in dims):
            return None

        # ? if only one dimension is requested, return the dimension directly
        if len(dims) == 1:
            return self._meshes[dims[0]]
        else:
            # ? here, if we are requesting a multi-dimensional mesh, we need to make sure we do not cross reference between different global meshes.
            # ? once we find the first global mesh that contains all the requested dimensions, we can return the sub-mesh from it. If no global mesh contains all the requested dimensions, we raise an error.
            for global_mesh in self._global_meshes.values():
                assert global_mesh.mesh_dim_names is not None
                if not set(dims).issubset(set(global_mesh.mesh_dim_names)):
                    continue
                return global_mesh[tuple(dims)]
            raise ValueError(f"Invalid mesh name combinations {dims}.")

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                    'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                    'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s).

        Raises:
            ValueError: If the mesh is not available (dimension size = 1 or not enabled),
                or if the requested dimension name(s) is not valid.
        """
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            enabled_str = (
                "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            )
            raise ValueError(
                f"Mesh '{dims}' is not available. "
                f"Ensure the corresponding parallelism dimension is {enabled_str}."
            )
        return mesh

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        """Get all enabled one-dimensional device meshes.

        Returns a dictionary of enabled one-dimensional device meshes, allowing you to
        access their process groups.

        Note:
            Device meshes created with the Fake backend are still included in the results.

        Returns:
            dict[str, DeviceMesh]: A dictionary mapping mesh dimension names to their
                corresponding DeviceMesh objects. Only includes meshes where:
                - ndim == 1 (one-dimensional)
                - parallelism is enabled (size > 1)

        Example:
            >>> parallel_dims = ParallelDims(
            ...     dp_replicate=2, dp_shard=2, cp=1, tp=2, pp=1, ep=1, etp=1, world_size=8
            ... )
            >>> meshes = parallel_dims.get_all_one_dimensional_meshes()
            >>> print(meshes.keys())
            dict_keys(['dp_replicate', 'fsdp', 'tp', 'batch', 'loss', 'efsdp'])
        """
        # ? lazy initialization
        if not self._meshes:
            self.build_mesh()

        # ? filter the meshes to only include those that are 1-dimensional and have size > 1 (parallelism enabled)
        return {k: v for k, v in self._meshes.items() if v.ndim == 1 and v.size() > 1}

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @property
    def etp_enabled(self):
        return self.etp > 1

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        return self.dp_replicate * self.dp_shard * self.cp

    @property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # https://github.com/pytorch/torchtitan/pull/640#discussion_r1849481001

        # Context Parallel requires that seq_len be divisible by 2 * CP degree,
        # when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.tp * (self.cp * 2)

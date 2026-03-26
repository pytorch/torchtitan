# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    JointWithDescriptors,
)
from torch._guards import tracing, TracingContext

from torchtitan.config import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    create_extra_fsdp_pg,
    end_with_pass,
    get_extra_fsdp_pg_name,
)
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


def _dump_gm(dump_folder: str | None, gm: torch.fx.GraphModule, name: str) -> None:
    # TODO: make the dump rank configurable
    if not dump_folder or torch.distributed.get_rank() != 0:
        return

    output_path = Path(dump_folder) / "compiler" / f"{name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        gm.print_readable(print_output=False, include_stride=True, include_device=True)
    )


def export_joint(
    model, args, kwargs=None, dump_folder: str | None = None
) -> tuple[JointWithDescriptors, TracingContext]:
    """
    Export joint forward-backward graph with AOT Autograd.

    Args:
        model: The model to export
        args: Tuple of input arguments
        kwargs: Dict of keyword arguments for the model
        dump_folder: Optional folder to dump the graph to
    """
    if kwargs is None:
        kwargs = {}
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    with (
        # TODO Investigate error on MOE model with use_grouped_mm=False.
        # For repro, see: https://gist.github.com/zhxchen17/d794ff58236243d9faddf713b9fc6a61
        torch._dynamo.config.patch(fake_tensor_cache_enabled=False),
        torch.fx.traceback.preserve_node_meta(),
    ):
        gm = dynamo_graph_capture_for_export(model)(*args, **kwargs)
        _dump_gm(dump_folder, gm, "dynamo_gm")

        tracing_context = gm.meta["tracing_context"]

    with tracing(tracing_context):
        return (
            aot_export_joint_with_descriptors_alone(gm, args, kwargs),
            tracing_context,
        )


def aot_export_joint_with_descriptors_alone(model, args, kwargs=None):
    """
    Export joint forward-backward graph with AOT Autograd.

    Args:
        model: The model to export
        args: Tuple of input arguments
        kwargs: Dict of keyword arguments for the model
    """
    if kwargs is None:
        kwargs = {}
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            args,
            kwargs,
        )

        return joint_with_descriptors


def joint_graph_builder(
    model: torch.nn.Module,
    model_args: tuple,
    model_kwargs: dict,
    fw_compiler: Callable | None = None,
    bw_compiler: Callable | None = None,
    joint_custom_passes: list[Callable] | None = None,
    dump_folder: str | None = None,
    compile_config: CompileConfig | None = None,
):
    """
    Build a joint forward-backward graph for the model with optional custom compilers.

    Args:
        model: The model to compile
        model_args: Tuple of model input arguments
        model_kwargs: Dict of model input keyword arguments
        fw_compiler: Optional custom forward compiler function
        bw_compiler: Optional custom backward compiler function
        joint_custom_passes: list of custom passes to run on the joint graph
        dump_folder: Optional folder to dump the graph to
        job_config: Job configuration
    """
    assert isinstance(model_args, tuple)

    # get joint graph
    (joint_with_descriptors, tracing_context,) = export_joint(
        model,
        model_args,
        model_kwargs,
        dump_folder=dump_folder,
    )

    # Check if inductor_decomposition is configured and create the pass with proper context
    if compile_config is not None:
        joint_pass_names = getattr(compile_config, "joint_passes", [])
        if "inductor_decomposition" in joint_pass_names:
            from torchtitan.experiments.graph_trainer.passes import (
                inductor_decomposition_pass,
            )

            # Create the decomposition pass with context
            decomp_pass = functools.partial(
                inductor_decomposition_pass,
                joint_with_descriptors=joint_with_descriptors,
            )

            # Prepend to joint_custom_passes
            if joint_custom_passes is None:
                joint_custom_passes = []
            joint_custom_passes = [decomp_pass] + joint_custom_passes

    # run custom passes on joint-graph before partitioner
    if joint_custom_passes is not None:
        for joint_custom_pass in joint_custom_passes:
            joint_with_descriptors.graph_module = joint_custom_pass(
                joint_with_descriptors.graph_module
            )

    with tracing(tracing_context):
        fn = aot_compile_joint_with_descriptors(
            joint_with_descriptors, fw_compiler=fw_compiler, bw_compiler=bw_compiler
        )

    def wrapper_fn(args, kwargs):
        inputs = [
            *model.parameters(),
            *model.buffers(),
            *args,
        ]
        return fn(*inputs, **kwargs)

    return wrapper_fn


class CompiledModule(Module):
    def __init__(
        self,
        inner: torch.nn.Module,
        parallel_dims: ParallelDims,
        joint_graph_builder: Callable,
        parallelize_inputs: Callable,
        **overrides,
    ) -> None:
        super().__init__()
        self.inner = inner  # register as submodule
        self.parallel_dims = parallel_dims

        self.joint_graph_builder = joint_graph_builder
        self.joint_graph_module = None

        self.parallelize_inputs = parallelize_inputs

        self._overrides = overrides  # for custom hooks

    def __getattr__(self, name: str):
        # check overrides
        if "_overrides" in self.__dict__ and name in self._overrides:
            return self._overrides[name]
        try:
            # let nn.Module handle registered stuff
            return super().__getattr__(name)
        except AttributeError:
            # fallback to inner model
            return getattr(self.inner, name)

    def __setattr__(self, name: str, value) -> None:
        if "_overrides" in self.__dict__ and name in self._overrides:
            self._overrides[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if "_overrides" in self.__dict__ and name in self._overrides:
            del self._overrides[name]
        else:
            super().__delattr__(name)

    def init_weights(self, **kwargs) -> None:
        # Explicitly delegate to inner model. Without this override,
        # Module.init_weights (a no-op) would be found via MRO before
        # the overwritten __getattr__ is triggered, silently skipping
        # weight initialization.
        # This is similar to state_dict, load_state_dict, ...
        self.inner.init_weights(**kwargs)

    def state_dict(self, *args, **kwargs) -> Any:
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs) -> Any:
        return self.inner.load_state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs) -> Any:
        return self.inner.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Any:
        return self.inner.parameters(*args, **kwargs)

    def forward(self, *args, **kwargs):
        assert "forward" not in self._overrides, "forward cannot be overridden"

        dt_args, dt_kwargs = self.parallelize_inputs(self.parallel_dims, args, kwargs)

        if self.joint_graph_module is None:
            self.joint_graph_module = self.joint_graph_builder(
                self.inner, dt_args, dt_kwargs
            )

        # calling the line below returns control to torchtitan's runner
        # letting it call the backward, and optimizer.
        return self.joint_graph_module(dt_args, dt_kwargs)


# Default compiler pass configuration - no passes by default
DEFAULT_COMPILER_PASSES = []


def compiler(
    name: str,
    gm: torch.fx.GraphModule,
    example_inputs,
    passes: list[Callable] = None,
    dump_folder: str | None = None,
    is_forward: bool = True,
):
    """
    Compile a graph module by applying a sequence of compiler passes.

    Args:
        name: Name for logging purposes
        gm: The graph module to compile
        example_inputs: Example inputs for the graph module
        passes: List of compiler pass functions to apply. Each function should take
                (gm, example_inputs) and return a transformed gm. If None, uses
                DEFAULT_COMPILER_PASSES.
        dump_folder: Optional folder to dump the graph to
    """
    if passes is None:
        passes = DEFAULT_COMPILER_PASSES

    _dump_gm(dump_folder, gm, f"{name}_before_compiler")

    if end_with_pass(passes, ["cudagraph_pass"]):
        # cudagraph pass is always the last pass if it is applied
        cg_pass = passes[-1]

        # to identify static input indices, cudagraph passes behaves differently for
        # forward and backward pass. so we explicitly pass the info.
        _cg_pass = functools.partial(cg_pass, is_forward=is_forward)

        # keep the function name for debug log
        passes[-1] = functools.wraps(cg_pass)(_cg_pass)

    for pass_fn in passes:
        pass_name = (
            pass_fn.func.__name__
            if isinstance(pass_fn, functools.partial)
            else pass_fn.__name__
        )
        logger.info(f"Applying pass: {pass_name}")
        gm = pass_fn(gm, example_inputs)

    # Only try to print/dump if gm is still a GraphModule
    # (compile_fx_inner returns a CompiledFxGraph which doesn't have print_readable)
    if hasattr(gm, "print_readable"):
        _dump_gm(dump_folder, gm, f"{name}_after_compiler")

        # Log the final transformed graph to tlparse.
        from torchtitan.experiments.graph_trainer.passes import tlparse_log_graph_pass

        graph_name = (
            "aot_forward_graph_transformed"
            if is_forward
            else "aot_backward_graph_transformed"
        )
        tlparse_log_graph_pass(gm, example_inputs, graph_name=graph_name)

    return gm


def make_compiler_with_passes(
    passes: list[Callable] = None,
    dump_folder: str | None = None,
):
    """
    Create forward and backward compilers with specified passes.

    Args:
        passes: List of compiler pass functions to apply. If None, uses DEFAULT_COMPILER_PASSES.
        dump_folder: Optional folder to dump graphs

    Returns:
        Tuple of (fw_compiler, bw_compiler) functions
    """

    def fw_compiler(gm: torch.fx.GraphModule, example_inputs):
        return compiler(
            "fwd_gm",
            gm,
            example_inputs,
            passes=passes,
            dump_folder=dump_folder,
            is_forward=True,
        )

    def bw_compiler(gm: torch.fx.GraphModule, example_inputs):
        return compiler(
            "bwd_gm",
            gm,
            example_inputs,
            passes=passes,
            dump_folder=dump_folder,
            is_forward=False,
        )

    return fw_compiler, bw_compiler


def validate_pass_names(pass_names: list[str], joint_pass_names: list[str]) -> None:
    """
    Validate compiler and joint pass names and their dependencies.

    Args:
        pass_names: List of compiler pass names
        joint_pass_names: List of joint custom pass names

    Raises:
        ValueError: If pass configuration is invalid
    """
    if "cudagraph" in pass_names and pass_names[-1] != "cudagraph":
        raise ValueError("cudagraph has to be the last pass to apply")

    if "auto_bucketing" in pass_names and "transformer_block_bucketing" in pass_names:
        raise ValueError(
            "Cannot apply auto_bucketing and transformer_block_bucketing at the same time!"
        )

    # full_inductor_compilation returns a CompiledFxGraph (not a GraphModule),
    # so no subsequent pass can inspect/modify the FX graph. It must be the
    # last pass, or second-to-last if cudagraph is last.
    if "full_inductor_compilation" in pass_names:
        if "inductor_decomposition" not in joint_pass_names:
            raise ValueError(
                "full_inductor_compilation pass requires inductor_decomposition to be "
                "specified in joint_passes. Please add --compile.joint_passes inductor_decomposition"
            )
        full_inductor_idx = pass_names.index("full_inductor_compilation")
        expected_idx = (
            len(pass_names) - 2
            if pass_names[-1] == "cudagraph"
            else len(pass_names) - 1
        )
        if full_inductor_idx != expected_idx:
            raise ValueError(
                "full_inductor_compilation must be the last pass "
                "(or second-to-last if cudagraph is last)."
            )


def get_compiler_passes_from_config(
    model: torch.nn.Module,
    compile_config: CompileConfig,
    parallel_dims: ParallelDims,
):
    """
    Extract and validate compiler passes from job config.

    Args:
        model: The model being compiled
        job_config: Job configuration containing compile.passes and compile.joint_passes
        parallel_dims: Parallelism dimensions (required for separate_ag_pg pass)

    Returns:
        List of compiler pass functions
    """
    from torchtitan.experiments.graph_trainer.common_utils import (
        get_transformer_block_buckets,
    )
    from torchtitan.experiments.graph_trainer.passes import AVAILABLE_COMPILER_PASSES

    pass_names = getattr(compile_config, "passes", [])
    joint_pass_names = getattr(compile_config, "joint_passes", [])

    validate_pass_names(pass_names, joint_pass_names)
    compiler_passes = []

    # Warn if full Inductor compilation is enabled
    if "full_inductor_compilation" in pass_names:
        logger.warning(
            "Full Inductor compilation is enabled. Note that Inductor may change numerics "
            "and does not guarantee bitwise equivalent results compared to eager mode."
        )

    for pass_name in pass_names:
        if pass_name not in AVAILABLE_COMPILER_PASSES:
            raise ValueError(
                f"Unknown compiler pass: {pass_name}. "
                f"Available compiler passes: {list(AVAILABLE_COMPILER_PASSES.keys())}"
            )
        if pass_name == "transformer_block_bucketing":
            from torchtitan.experiments.graph_trainer.passes import reassign_to_pg_pass

            create_extra_fsdp_pg(parallel_dims)
            fsdp_mesh = parallel_dims.get_mesh("fsdp")
            fsdp_pg = fsdp_mesh.get_group()
            fsdp_pg_name = fsdp_pg.group_name
            extra_pg_name = get_extra_fsdp_pg_name(fsdp_pg_name)
            compiler_passes.append(
                functools.partial(
                    reassign_to_pg_pass,
                    source_pg_name=fsdp_pg_name,
                    target_pg_name=extra_pg_name,
                )
            )
            compiler_passes.append(
                functools.partial(
                    AVAILABLE_COMPILER_PASSES[pass_name],
                    fsdp_manual_buckets=get_transformer_block_buckets(model),
                )
            )
        else:
            compiler_passes.append(AVAILABLE_COMPILER_PASSES[pass_name])

    if pass_names:
        logger.info(f"Using compiler passes from config: {pass_names}")

    return compiler_passes


def get_joint_custom_passes_from_config(
    parallel_dims: ParallelDims,
    compile_config: CompileConfig,
    fsdp_reshard_after_forward: bool,
):
    """
    Extract and validate joint custom passes from job config.

    Note: The inductor_decomposition pass is handled separately in joint_graph_builder
    because it requires context (model, joint_with_descriptors, etc.) that's only
    available at graph capture time.

    Args:
        parallel_dims: Parallelism dimensions
        compile_config: Compile configuration containing joint_passes
        fsdp_reshard_after_forward: Whether to reshard after forward (already resolved)

    Returns:
        List of joint custom pass functions
    """
    from torchtitan.experiments.graph_trainer.passes import (
        AVAILABLE_JOINT_PASSES,
        fsdp_reshard_after_fwd_pass,
        validate_flex_attn_annotation_pass,
    )

    joint_custom_passes = []

    # Skip flex_attention annotation validation when full_inductor_compilation
    # is used, since it compiles everything through Inductor regardless of
    # annotations. The validation is only relevant for regional_inductor.
    pass_names = getattr(compile_config, "passes", [])
    if "full_inductor_compilation" not in pass_names:
        joint_custom_passes.append(validate_flex_attn_annotation_pass)

    # Handle joint passes from config (excluding inductor_decomposition)
    joint_pass_names = getattr(compile_config, "joint_passes", [])
    for pass_name in joint_pass_names:
        if pass_name not in AVAILABLE_JOINT_PASSES:
            raise ValueError(
                f"Unknown joint pass: {pass_name}. "
                f"Available joint passes: {list(AVAILABLE_JOINT_PASSES.keys())}"
            )

        # Skip inductor_decomposition - it's handled in joint_graph_builder
        if pass_name == "inductor_decomposition":
            continue

        joint_custom_passes.append(AVAILABLE_JOINT_PASSES[pass_name])

    if joint_pass_names:
        logger.info(f"Using joint passes from config: {joint_pass_names}")

    joint_custom_passes.append(
        functools.partial(
            fsdp_reshard_after_fwd_pass,
            reshard_after_forward=fsdp_reshard_after_forward,
        )
    )

    return joint_custom_passes


def pp_joint_graph_builder(
    model: torch.nn.Module,
    model_args: tuple,
    model_kwargs: dict,
    fw_compiler: Callable | None = None,
    bw_compiler: Callable | None = None,
    joint_custom_passes: list[Callable] | None = None,
    dump_folder: str | None = None,
    compile_config: CompileConfig | None = None,
    graph_pp_passes: list[str] | None = None,
    parallel_dims: ParallelDims | None = None,
):
    """
    Build a joint forward-backward graph, partition it, and apply graph PP
    passes (split_fsdp_collectives, split_dI_dW) to produce subgraphs for
    graph-based pipeline parallelism execution via GraphPPRunner.

    Args:
        model: The model to compile
        model_args: Tuple of model input arguments
        model_kwargs: Dict of model input keyword arguments
        fw_compiler: Optional custom forward compiler function
        bw_compiler: Optional custom backward compiler function
        joint_custom_passes: list of custom passes to run on the joint graph
        dump_folder: Optional folder to dump the graph to
        compile_config: Compile configuration
        graph_pp_passes: List of graph PP pass names to apply
        parallel_dims: Parallel dimensions for pipeline configuration
    """
    from autoparallel.graph_passes.graph_partition import (
        partition_joint_with_descriptors,
    )

    assert isinstance(model_args, tuple)
    assert graph_pp_passes, "graph_pp_passes must be non-empty"
    assert parallel_dims is not None and parallel_dims.pp_enabled

    valid_passes = {"split_fsdp_collectives", "split_dI_dW"}
    for p in graph_pp_passes:
        if p not in valid_passes:
            raise ValueError(
                f"Unknown graph PP pass: {p}. Available: {sorted(valid_passes)}"
            )

    # Step 1: Export the joint graph
    joint_with_descriptors, tracing_context = export_joint(
        model, model_args, model_kwargs, dump_folder=dump_folder
    )

    # Step 2: Always apply inductor decompositions for graph PP.
    # The PP subgraphs are compiled with Inductor (compile_fx_inner)
    # which requires ops to be decomposed first.
    from torchtitan.experiments.graph_trainer.passes import inductor_decomposition_pass

    decomp_pass = functools.partial(
        inductor_decomposition_pass,
        joint_with_descriptors=joint_with_descriptors,
    )
    if joint_custom_passes is None:
        joint_custom_passes = []
    joint_custom_passes = [decomp_pass] + joint_custom_passes

    # Step 3: Run custom passes on joint graph before partitioning
    if joint_custom_passes is not None:
        for joint_custom_pass in joint_custom_passes:
            joint_with_descriptors.graph_module = joint_custom_pass(
                joint_with_descriptors.graph_module
            )

    # Step 4: Partition into fw/bw using autoparallel's partitioner
    with tracing(tracing_context):
        (
            fw_module,
            bw_module,
            num_params_buffers,
            num_user_outputs,
            num_mutate_inputs,
            num_fw_outs_saved_for_bw,
            num_symints_saved_for_bw,
            _indices_of_inps_to_detach,
            adjusted_flat_args,
        ) = partition_joint_with_descriptors(joint_with_descriptors)

    num_params = len(list(model.parameters()))
    num_buffers = len(list(model.buffers()))
    assert num_params_buffers == num_params + num_buffers, (
        f"num_params_buffers: {num_params_buffers}, "
        f"num_params: {num_params}, num_buffers: {num_buffers}"
    )
    num_input_grads = (
        len(bw_module.graph.find_nodes(op="output")[0].args[0]) - num_params_buffers
    )

    _dump_gm(dump_folder, fw_module, "pp_fw_module")
    _dump_gm(dump_folder, bw_module, "pp_bw_module")

    logger.info(
        f"Partitioned joint graph: {num_params_buffers=}, {num_user_outputs=}, "
        f"{num_mutate_inputs=}, {num_input_grads=}, "
        f"{num_fw_outs_saved_for_bw=}, {num_symints_saved_for_bw=}"
    )

    # Step 5: Apply split_fsdp_collectives if configured
    unshard_module = None
    reduce_grad_module = None
    if "split_fsdp_collectives" in graph_pp_passes:
        from autoparallel.graph_passes.split_fsdp_collectives import (
            split_fsdp_prefetch,
            split_fsdp_reduce_scatters_epilogue,
        )

        unshard_module, fw_module = split_fsdp_prefetch(fw_module, num_params)
        bw_module, reduce_grad_module = split_fsdp_reduce_scatters_epilogue(
            bw_module, num_params
        )
        _dump_gm(dump_folder, unshard_module, "pp_unshard_module")
        _dump_gm(dump_folder, fw_module, "pp_fw_module_no_fsdp")
        _dump_gm(dump_folder, bw_module, "pp_bw_module_no_fsdp")
        _dump_gm(dump_folder, reduce_grad_module, "pp_reduce_grad_module")
        logger.info("Applied split_fsdp_collectives pass")

    # Step 6: Apply split_dI_dW if configured
    bw_dI_module = None
    bw_dW_module = None
    if "split_dI_dW" in graph_pp_passes:
        from autoparallel.graph_passes.split_di_dw_graph import split_di_dw_graph

        bw_dI_module, bw_dW_module, num_input_grads = split_di_dw_graph(
            bw_module, num_weight_gradients=num_params_buffers
        )
        _dump_gm(dump_folder, bw_dI_module, "pp_bw_dI_module")
        _dump_gm(dump_folder, bw_dW_module, "pp_bw_dW_module")
        logger.info("Applied split_dI_dW pass")

    # Attach shape_env to subgraphs so compile_fx_inner can handle
    # data-dependent ops (_local_scalar_dense) via unbacked symints.
    # Without this, compile_fx_inner creates a FakeTensorMode without
    # allow_scalar_outputs, causing DataDependentOutputException.
    shape_env = getattr(tracing_context, "fake_mode", None)
    shape_env = getattr(shape_env, "shape_env", None)
    for gm in (
        fw_module, bw_module, bw_dI_module, bw_dW_module,
        unshard_module, reduce_grad_module,
    ):
        if gm is not None and shape_env is not None:
            gm.shape_env = shape_env

    # Step 7: Apply compiler passes to each subgraph
    subgraphs = {
        "fw": fw_module,
        "full_bw": bw_module,
        "bw_dI": bw_dI_module,
        "bw_dW": bw_dW_module,
        "unshard": unshard_module,
        "reduce_grad": reduce_grad_module,
    }
    if fw_compiler is not None:
        for name, gm in subgraphs.items():
            if gm is not None:
                is_fw = name in ("fw", "unshard")
                compile_fn = fw_compiler if is_fw else bw_compiler
                example_inputs = [
                    node.meta.get("val")
                    for node in gm.graph.find_nodes(op="placeholder")
                ]
                subgraphs[name] = compile_fn(gm, example_inputs)

    # Step 8: Build GraphCallables and GraphMeta
    from autoparallel.graph_passes.graph_pp_runner import GraphCallables, GraphMeta

    graph_callables = GraphCallables(
        fw=subgraphs["fw"],
        full_bw=subgraphs["full_bw"],
        bw_dI=subgraphs["bw_dI"],
        bw_dW=subgraphs["bw_dW"],
        unshard=subgraphs["unshard"],
        reduce_grad=subgraphs["reduce_grad"],
    )
    graph_meta = GraphMeta(
        num_mutate_inputs=num_mutate_inputs,
        num_user_outputs=num_user_outputs,
        num_symints_saved_for_bw=num_symints_saved_for_bw,
        num_params=num_params,
        num_buffers=num_buffers,
        num_input_grads=num_input_grads,
    )

    return graph_callables, graph_meta


class CompiledPPModule(Module):
    """Wraps a model for graph-based pipeline parallelism execution.

    On first forward, builds the PP subgraphs (fw, bw, unshard, reduce_grad,
    etc.), constructs GraphPipelineStages, and wires them into a
    GraphPPRunner that executes the pipeline schedule.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        parallel_dims: ParallelDims,
        pp_joint_graph_builder_fn: Callable,
        parallelize_inputs: Callable,
        pp_schedule_builder: Callable,
        **overrides,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.parallel_dims = parallel_dims
        self.pp_joint_graph_builder_fn = pp_joint_graph_builder_fn
        self.parallelize_inputs = parallelize_inputs
        self.pp_schedule_builder = pp_schedule_builder
        self._pp_runner = None
        self._overrides = overrides

    def __getattr__(self, name: str):
        if "_overrides" in self.__dict__ and name in self._overrides:
            return self._overrides[name]
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def __setattr__(self, name: str, value) -> None:
        if "_overrides" in self.__dict__ and name in self._overrides:
            self._overrides[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if "_overrides" in self.__dict__ and name in self._overrides:
            del self._overrides[name]
        else:
            super().__delattr__(name)

    def init_weights(self, **kwargs) -> None:
        self.inner.init_weights(**kwargs)

    def state_dict(self, *args, **kwargs) -> Any:
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs) -> Any:
        return self.inner.load_state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs) -> Any:
        return self.inner.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Any:
        return self.inner.parameters(*args, **kwargs)

    def _build_pp_runner(self, dt_args, dt_kwargs):
        """Build graph callables and construct the GraphPPRunner."""
        from autoparallel.graph_passes.graph_pp_runner import (
            GraphPipelineStage,
            GraphPPRunner,
            stage_backward_input,
            stage_backward_weight,
            stage_forward,
            stage_full_backward,
            stage_reduce_grad,
            stage_reshard,
            stage_unshard,
        )
        from torch.distributed.pipelining.schedules import (
            _PipelineScheduleRuntime,
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            FORWARD,
            FULL_BACKWARD,
            REDUCE_GRAD,
            RESHARD,
            UNSHARD,
        )

        graph_callables, graph_meta = self.pp_joint_graph_builder_fn(
            self.inner, dt_args, dt_kwargs
        )

        # Build a single GraphPipelineStage for this rank's stage
        # (single-stage-per-rank for now)
        pp_mesh = self.parallel_dims.get_mesh("pp")
        stage_index = pp_mesh.get_local_rank()
        num_stages = self.parallel_dims.pp

        # Determine input/output args for the pipeline stage's send/recv
        # buffer setup. For graph PP, the graph handles the actual computation;
        # these are only used for P2P communication buffer shapes.
        fw_gm = graph_callables.fw
        fw_placeholders = fw_gm.graph.find_nodes(op="placeholder")
        num_params_buffers = graph_meta.num_params + graph_meta.num_buffers
        # Activation inputs are fw placeholders after params/buffers
        activation_placeholders = fw_placeholders[num_params_buffers:]
        # Get user outputs from the fw graph for output shape inference
        fw_outputs = fw_gm.graph.find_nodes(op="output")
        num_inner_fwd_outputs = (
            graph_meta.num_mutate_inputs + graph_meta.num_user_outputs
        )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        def _meta_from_val(val):
            if isinstance(val, torch.Tensor):
                return torch.empty(val.shape, dtype=val.dtype, device="meta")
            return val

        # Input args: what this stage receives from prev stage
        input_args = tuple(
            _meta_from_val(p.meta["val"])
            for p in activation_placeholders
            if "val" in p.meta and isinstance(p.meta["val"], torch.Tensor)
        )
        if not input_args:
            input_args = (torch.empty(1, dtype=torch.bfloat16, device="meta"),)

        # Output args: what this stage sends to next stage
        if fw_outputs:
            out_args = fw_outputs[0].args[0]
            if isinstance(out_args, tuple):
                user_outs = out_args[
                    graph_meta.num_mutate_inputs : num_inner_fwd_outputs
                ]
                output_args = tuple(
                    _meta_from_val(o.meta["val"])
                    for o in user_outs
                    if hasattr(o, "meta")
                    and "val" in o.meta
                    and isinstance(o.meta["val"], torch.Tensor)
                )
            else:
                output_args = None
        else:
            output_args = None

        stage = GraphPipelineStage(
            submodule=self.inner,
            graph_callables=graph_callables,
            graph_meta=graph_meta,
            stage_index=stage_index,
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
            group=pp_mesh.get_group(),
        )

        # Build the schedule via the provided builder.
        # The builder should return a _PipelineScheduleRuntime which
        # supports custom action functions needed by GraphPPRunner.
        schedule = self.pp_schedule_builder(stages=[stage])
        assert isinstance(schedule, _PipelineScheduleRuntime), (
            f"Graph PP requires _PipelineScheduleRuntime, got {type(schedule).__name__}. "
            f"Set --parallelism.pipeline_parallel_schedule_csv to a schedule CSV file."
        )

        # Register custom action functions
        schedule.register_custom_function(FORWARD, stage_forward)
        schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
        schedule.register_custom_function(REDUCE_GRAD, stage_reduce_grad)
        schedule.register_custom_function(RESHARD, stage_reshard)
        schedule.register_custom_function(UNSHARD, stage_unshard)
        schedule.register_custom_function(BACKWARD_INPUT, stage_backward_input)
        schedule.register_custom_function(BACKWARD_WEIGHT, stage_backward_weight)

        return GraphPPRunner(schedule)

    def forward(
        self,
        *args,
        target=None,
        losses=None,
        return_outputs=False,
        **kwargs,
    ):
        assert "forward" not in self._overrides, "forward cannot be overridden"

        if self._pp_runner is None:
            # Lazily build the PP runner on first call.
            # All ranks need example inputs for graph capture, even
            # non-first stages that don't receive inputs at runtime.
            if args:
                dt_args, dt_kwargs = self.parallelize_inputs(
                    self.parallel_dims, args, kwargs
                )
            else:
                # Non-first stage: generate example inputs for graph capture
                capture_args = self._make_capture_args()
                dt_args, dt_kwargs = self.parallelize_inputs(
                    self.parallel_dims, capture_args, kwargs
                )
            self._pp_runner = self._build_pp_runner(dt_args, dt_kwargs)

        # GraphPPRunner.step() handles both forward and backward
        if args:
            dt_args, _ = self.parallelize_inputs(self.parallel_dims, args, kwargs)
            self._pp_runner.step(
                *dt_args if isinstance(dt_args, tuple) else (dt_args,),
                target=target,
                losses=losses,
                return_outputs=return_outputs,
            )
        else:
            self._pp_runner.step(
                target=target,
                losses=losses,
                return_outputs=return_outputs,
            )

        # Return a dummy loss; actual losses are collected via the losses list
        return torch.tensor(
            [-1.0], device=torch.device(f"cuda:{torch.cuda.current_device()}")
        )

    def _make_capture_args(self) -> tuple:
        """Create example inputs for graph capture on non-first PP stages."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        vocab_size = getattr(self, "_capture_vocab_size", 1024)
        batch_size = getattr(self, "_capture_batch_size", 1)
        seq_len = getattr(self, "_capture_seq_len", 128)
        dummy_tokens = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
        return (dummy_tokens,)

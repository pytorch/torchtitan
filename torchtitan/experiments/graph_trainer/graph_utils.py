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
from torch.utils._pytree import TreeSpec

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
    model,
    args,
    kwargs=None,
    dump_folder: str | None = None,
    precompile: bool = False,
) -> tuple[JointWithDescriptors, TracingContext]:
    """
    Export joint forward-backward graph with AOT Autograd.

    Args:
        model: The model to export
        args: Tuple of input arguments
        kwargs: Dict of keyword arguments for the model
        dump_folder: Optional folder to dump the graph to
        precompile: If True, enable compile-on-one-rank (CooR) so that
            ProcessGroups flow through the graph as opaque inputs instead
            of being baked as string-literal PG names. This makes the
            serialized artifact rank-agnostic.
    """
    if kwargs is None:
        kwargs = {}
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    import torch.distributed.config as dist_config

    coor_ctx = (
        dist_config.patch("compile_on_one_rank", True)
        if precompile
        else contextlib.nullcontext()
    )
    with coor_ctx:
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
    serializable: bool = False,
    on_compile: Callable[[Any, TreeSpec | None], None] | None = None,
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
        compile_config: Compile configuration
        serializable: If True, compile with serialization support
        on_compile: Optional callback invoked after compilation with
            (compiled_fn, out_spec)
    """
    assert isinstance(model_args, tuple)

    # get joint graph
    (joint_with_descriptors, tracing_context,) = export_joint(
        model,
        model_args,
        model_kwargs,
        dump_folder=dump_folder,
        precompile=serializable,
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
            joint_with_descriptors,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            serializable=serializable,
        )

    if on_compile is not None:
        on_compile(fn, joint_with_descriptors.out_spec)

    def wrapper_fn(args, kwargs):
        params = [p for _, p in model.named_parameters(remove_duplicate=False)]
        inputs = [
            *params,
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
        precompiled_fn: Callable | None = None,
        **overrides,
    ) -> None:
        super().__init__()
        self.inner = inner  # register as submodule
        self.parallel_dims = parallel_dims

        self.joint_graph_builder = joint_graph_builder
        self.joint_graph_module = None
        self.precompiled_fn = precompiled_fn

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

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        # Explicitly delegate to inner model. Without this override,
        # Module.init_states would be found via MRO before the overwritten
        # __getattr__ is triggered, silently skipping weight initialization.
        # This is similar to state_dict, load_state_dict, ...
        self.inner.init_states(buffer_device=buffer_device)

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
            if self.precompiled_fn is not None:
                self.joint_graph_module = self.precompiled_fn
            else:
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
        elif pass_name == "regional_inductor" and getattr(
            compile_config, "precompile_artifact_dir", ""
        ):
            # regional_inductor needs an explicit serializable=True at
            # the pass level so it produces serializable RegionalOutputCode.
            # full_inductor_compilation does NOT need a pass-level flag:
            # compile_fx_inner already returns a CompiledFxGraph that is
            # natively serializable, so aot_compile_joint_with_descriptors
            # (called with serializable=True in joint_graph_builder) can
            # bundle it into a BundledAOTAutogradSerializableCallable
            # without any pass-level cooperation.
            compiler_passes.append(
                functools.partial(
                    AVAILABLE_COMPILER_PASSES[pass_name],
                    serializable=True,
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
        annotate_flex_attention_for_regional_inductor_pass,
        AVAILABLE_JOINT_PASSES,
        fsdp_reshard_after_fwd_pass,
    )

    joint_custom_passes = []

    # Skip flex_attention annotation validation when full_inductor_compilation
    # is used, since it compiles everything through Inductor regardless of
    # annotations. The validation is only relevant for regional_inductor.
    pass_names = getattr(compile_config, "passes", [])
    if "full_inductor_compilation" not in pass_names:
        from torchtitan.models.common.attention import FlexAttention

        joint_custom_passes.append(
            functools.partial(
                annotate_flex_attention_for_regional_inductor_pass,
                flex_compile_config=FlexAttention.inductor_configs,
            )
        )

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

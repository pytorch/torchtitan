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
from torch.distributed.tensor import DTensor
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.compiler_toolkit.common_utils import (
    create_extra_fsdp_pg,
    end_with_pass,
    get_extra_fsdp_pg_name,
)
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
        logger.debug("Dynamo gm:")
        logger.debug(
            gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            )
        )
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
        model_args: Tuple of model input arguments (should be DTensors)
        model_kwargs: Dict of model input keyword arguments
        fw_compiler: Optional custom forward compiler function
        bw_compiler: Optional custom backward compiler function
        joint_custom_passes: list of custom passes to run on the joint graph
        dump_folder: Optional folder to dump the graph to
        job_config: Job configuration
    """
    assert isinstance(model_args, tuple)
    for idx, arg in enumerate(model_args):
        assert isinstance(arg, DTensor), f"Argument {idx} is of type {type(arg)}"

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
            from torchtitan.experiments.compiler_toolkit.passes import (
                inductor_decomposition_pass,
            )

            # Create the decomposition pass with context
            decomp_pass = functools.partial(
                inductor_decomposition_pass,
                model=model,
                joint_with_descriptors=joint_with_descriptors,
                forward_inputs=model_args,
                tracing_context=tracing_context,
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


class CompiledModule(torch.nn.Module):
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

    logger.debug(f"{name} before compiler:")
    logger.debug(
        gm.print_readable(print_output=False, include_stride=True, include_device=True)
    )
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
        logger.debug(f"{name} after compiler:")
        logger.debug(
            gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            )
        )
        _dump_gm(dump_folder, gm, f"{name}_after_compiler")

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
    if "cudagraph" in pass_names:
        assert (
            pass_names[-1] == "cudagraph"
        ), "cudagraph has to be the last pass to apply"

    if (
        "autobucketing_reordering" in pass_names
        and "transformer_block_bucketing" in pass_names
    ):
        raise ValueError(
            "Cannot apply autobucketing_reordering and transformer_block_bucketing at the same time!"
        )

    # Validate that full_inductor_compilation requires inductor_decomposition
    if "full_inductor_compilation" in pass_names:
        if "inductor_decomposition" not in joint_pass_names:
            raise ValueError(
                "full_inductor_compilation pass requires inductor_decomposition to be "
                "specified in joint_passes. Please add --compile.joint_passes inductor_decomposition"
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
    from torchtitan.experiments.compiler_toolkit.passes import AVAILABLE_COMPILER_PASSES
    from torchtitan.experiments.simple_fsdp.llama3.parallelize import (
        get_transformer_block_buckets,
    )

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
            from torchtitan.experiments.compiler_toolkit.passes import (
                reassign_to_pg_pass,
            )

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
    parallelism: ParallelismConfig,
):
    """
    Extract and validate joint custom passes from job config.

    Note: The inductor_decomposition pass is handled separately in joint_graph_builder
    because it requires context (model, joint_with_descriptors, etc.) that's only
    available at graph capture time.

    Args:
        parallel_dims: Parallelism dimensions
        job_config: Job configuration containing parallelism.fsdp_reshard_after_forward
                    and compile.joint_passes

    Returns:
        List of joint custom pass functions
    """
    from torchtitan.experiments.compiler_toolkit.passes import (
        AVAILABLE_JOINT_PASSES,
        fsdp_reshard_after_fwd_pass,
        validate_flex_attn_annotation_pass,
    )

    joint_custom_passes = []
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

    # Handle FSDP reshard after forward
    match parallelism.fsdp_reshard_after_forward:
        case "always":
            fsdp_reshard_after_forward = True
        case "never":
            fsdp_reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            fsdp_reshard_after_forward = not parallel_dims.pp_enabled
        case _:
            raise ValueError(
                f"Invalid fsdp_reshard_after_forward_policy: {parallelism.fsdp_reshard_after_forward}."
            )

    joint_custom_passes.append(
        functools.partial(
            fsdp_reshard_after_fwd_pass,
            reshard_after_forward=fsdp_reshard_after_forward,
        )
    )

    return joint_custom_passes

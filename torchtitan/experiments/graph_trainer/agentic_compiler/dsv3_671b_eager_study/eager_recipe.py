# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager DSv3 671B DistMoE MXFP8 recipes in the graph loop's measurement setup.

The base is ``deepseek_v3_671b_dist_moe_mxfp8`` (the eager twin of the graph loop's
``graph_trainer_deepseek_v3_671b_dist_moe_mxfp8``) with the loop's layout changes
(``example_recipe.dsv3_671b_base``): FSDP 256 x EP 64 without PP, one 4096-token microbatch
per step, Dist-MoE scratch capacity 64, deterministic, seed 42.

    RUN=<dir> PYTHONPATH=<this dir>:<agentic_compiler dir>:. NGPU=256 LOCAL_RANK=0 \\
        python -m torchtitan.train --module eager_recipe --config <name> --comm-backend fake

Configs:
- eager_lc: the recipe's local_compile regions (loss, rope, mla_qk, moe_aux_loss, ...).
- eager_nolc: local_compile.regions = [], the aten ops GraphTrainer traces.
- eager_kernels: the graph loop's kernels in eager Functions, selected by GK (see ``_kernels``).
- *_prof: 10 steps, step 10 profiled, accelerator graphs off (as the graph loop measured).
- *_tps: 20 steps, accelerator graphs as the recipe sets them, no profiler.
"""

import atexit
import os

from torchtitan.observability.metrics import TensorBoardLogger

from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b_dist_moe_mxfp8


def _base(*, local_compile: bool):
    config = deepseek_v3_671b_dist_moe_mxfp8()
    config.parallelism.data_parallel_shard_degree = 256
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.fsdp_defer_gradient_reduction = False
    config.training.num_tokens_per_train_step = -1
    config.dist_moe_runtime.scratch_capacity_factor = float(
        config.parallelism.expert_parallel_degree
    )
    if not local_compile:
        config.model.local_compile.regions = []
    if os.environ.get("LC_REGIONS"):
        # Only the listed local_compile regions (e.g. LC_REGIONS=rope).
        config.model.local_compile.regions = os.environ["LC_REGIONS"].split(",")
    config.debug.seed = 42
    config.debug.deterministic = True
    run = os.path.abspath(os.environ["RUN"])
    os.makedirs(run, exist_ok=True)
    config.dump_folder = run
    if not any(isinstance(c, TensorBoardLogger.Config) for c in config.metrics.loggers):
        config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(run, "tb")
    return config


def _prof(config):
    config.training.steps = 10
    config.training.disable_accelerator_graphs = True
    config.profiler.enable_profiling = True
    config.profiler.enable_memory_snapshot = False
    config.profiler.save_traces_folder = os.path.join(config.dump_folder, "profile")
    config.profiler.profile_freq = 10
    config.profiler.profiler_warmup = 3
    config.profiler.profiler_active = 1
    config.profiler.profiler_repeat = 1
    return config


def _tps(config):
    config.training.steps = 20
    config.profiler.enable_profiling = False
    return config


def eager_lc_prof():
    return _prof(_base(local_compile=True))


def eager_nolc_prof():
    return _prof(_base(local_compile=False))


def eager_lc_tps():
    return _tps(_base(local_compile=True))


def eager_nolc_tps():
    return _tps(_base(local_compile=False))


BLOCK_FLAGS = (
    "attention_input",
    "wq_b",
    "mla_kv",
    "attention_output",
    "router",
    "ffn",
    "fsdp_copy_in",
)


def _kernels(config):
    """Graph-loop kernels in eager. GK: comma-separated Functions (BLOCK_FLAGS and
    expert_weight_prep) or ``all``; LOCAL_COMPILE=1 keeps the recipe's local_compile regions."""
    names = [n for n in os.environ.get("GK", "").split(",") if n]
    if names == ["all"]:
        names = [*BLOCK_FLAGS, "expert_weight_prep"]
    # PyTorch's experimental FSDP2 copy-in, compared with fsdp_copy_in; not part of ``all``.
    native = "native_copy_in" in names
    names = [n for n in names if n != "native_copy_in"]
    unknown = set(names) - {*BLOCK_FLAGS, "expert_weight_prep"}
    if unknown:
        raise ValueError(f"unknown GK entries: {sorted(unknown)}")
    flags = {n: n in names for n in BLOCK_FLAGS}
    flags["native_copy_in"] = native
    if any(flags.values()):
        # The block override applies fused_mla and fused_swiglu inside each block itself.
        stock = {
            "torchtitan_recipes.overrides.fused_mla.fused_mla",
            "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
        }
        if set(config.override.imports) != stock:
            raise ValueError(f"unexpected overrides: {config.override.imports}")
        config.override.imports = [("override_dsv3.graph_kernel_block", flags)]
    import graph_kernels

    if "expert_weight_prep" in names:
        graph_kernels.install_expert_weight_prep()
    atexit.register(
        lambda: print(
            f"graph kernel calls: {dict(graph_kernels.CALLS)}; quantized gradient "
            f"lookups: {dict(graph_kernels.GRAD_QUANT_LOOKUPS)}"
        )
    )
    return config


def eager_kernels_prof():
    lc = os.environ.get("LOCAL_COMPILE", "0") == "1"
    return _kernels(_prof(_base(local_compile=lc)))


def eager_kernels_tps():
    lc = os.environ.get("LOCAL_COMPILE", "0") == "1"
    return _kernels(_tps(_base(local_compile=lc)))

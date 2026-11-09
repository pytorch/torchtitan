# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer throughput reference for the eager study: the 30B pack's recipe with and without
the pack, 20 steps, CUDA graphs as the recipe sets them, TensorBoard in RUN.

    RUN=<dir> PYTHONPATH=<this dir>:<pack repo>:. NGPU=8 LOCAL_RANK=0 \\
        python -m torchtitan.train --module graph_recipe --config graph_pack_tps --comm-backend fake
"""

import json
import os
from pathlib import Path

import graph_trainer_kernel_packs.pack as pack
import graph_trainer_kernel_packs.qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103 as pack_module

from graph_trainer_kernel_packs.qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103.recipe import (
    qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8,
)
from graph_trainer_kernel_packs.qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103.regions import (
    PLAN,
)

from torchtitan.experiments.graph_trainer.registry import register_graph_pass
from torchtitan.observability.metrics import TensorBoardLogger

PACK_DIR = Path(pack_module.__file__).parent
# Region -> pattern of its kernel, from the loop README's kernel table.
REGION_PATTERNS = {
    "G-01": "qnfe",
    "G-02": "4dzo",
    "G-06": "s3of",
    "G-07": "53aq",
    "G-90": "bme7",
    "G-17": "bp6b",
    "G-19": "zfby",
    "G-91": "bj3b",
    "G-15": "tl5t",
    "G-33": "gri3",
    "G-42": "adoe",
    "G-41": "kbxm",
    "G-45": "na2k",
    "G-28": "vf5c",
    "G-47": "5ifh",
}


def _tps(config):
    run = os.path.abspath(os.environ["RUN"])
    os.makedirs(run, exist_ok=True)
    config.dump_folder = run
    config.training.steps = 20
    config.profiler.enable_profiling = False
    config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(run, "tb")
    return config


@register_graph_pass
def qwen3_30b_pack_this_host(gm, example_inputs=None):
    """The 30B pack on this host, keyed by region (local measurement only).

    The pack was built on torch 084f866 / CUDA 13.2 / cutlass-dsl 4.8.1; this host has torch
    3c53c6f / CUDA 13.0 / 4.8.0. Every region lifts at its verified count, but neither key
    hits here: l2 comes from AOTAutograd's cache key and l1 hashes custom-op sources (torchao),
    both of which differ between installs. Each of the pack's 15 regions has exactly one
    kernel, so this pass substitutes by region id and keeps the exact call-count check; each
    kernel validates its input shapes itself.
    """
    manifest = json.loads((PACK_DIR / "manifest.json").read_text())
    kernels = manifest["kernels"]
    by_prefix = {key.removeprefix("pat_")[:4]: key for key in kernels}
    by_region = {
        region: by_prefix[prefix] for region, prefix in REGION_PATTERNS.items()
    }
    for region in PLAN:
        gm = pack.lift_matches(gm, region.REGION_ID, region.MATCHERS)
    region_ids = {region.REGION_ID for region in PLAN}
    calls = dict.fromkeys(kernels, 0)
    runs = {}
    for node in list(gm.graph.find_nodes(op="call_module")):
        if node.meta.get("fusion_region") not in region_ids:
            continue
        key = by_region[node.meta["fusion_region"]]
        if key not in runs:
            runs[key] = pack._load_kernel(
                PACK_DIR / "kernels" / key, kernels[key]["sha256"]
            )
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(runs[key], node.args)
        new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        calls[key] += 1
    wrong = {
        k: (n, kernels[k]["calls"])
        for k, n in calls.items()
        if n != kernels[k]["calls"]
    }
    if wrong:
        raise ValueError(f"kernel calls differ: {wrong}")
    gm.graph.eliminate_dead_code()
    gm.delete_all_unused_submodules()
    gm.recompile()
    print(
        f"qwen3_30b_pack_this_host: {sum(calls.values())} calls to {len(kernels)} kernels"
    )
    return gm


def graph_pack_tps():
    config = qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8()
    config.compile.extra_passes = ["qwen3_30b_pack_this_host"]
    return _tps(config)


def graph_base_tps():
    config = qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8()
    config.compile.extra_passes = []
    return _tps(config)


def _prof(config):
    config = _tps(config)
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


def graph_base_prof():
    return _prof(graph_base_tps())


def graph_pack_prof():
    return _prof(graph_pack_tps())

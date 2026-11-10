# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer reference for the eager study: the DSv3 loop's recipe without and with its
plan and kernel cache (``dsv3_671b_distmoe_mxfp8``), in the eager recipes' measurement setup.

    RUN=<dir> PYTHONPATH=<fusion_finder scripts>:<loop dir>:. NGPU=256 LOCAL_RANK=0 \\
        python -m torchtitan.train --module graph_recipe --config graph_pack_prof --comm-backend fake

Configs: graph_base_* (no plan), graph_pack_* (plan_iter9 lifted, kernels substituted from the
loop's kernel_cache); *_prof and *_tps as in eager_recipe.py.
"""

import os

import example_recipe
import fusion_regions

from torchtitan.experiments.graph_trainer.registry import register_graph_pass
from torchtitan.experiments.graph_trainer.substitute_kernels import (
    substitute_with_kernels_pass,
)

from torchtitan.observability.metrics import TensorBoardLogger

LOOP_DIR = os.path.dirname(os.path.abspath(fusion_regions.__file__)).rsplit("/", 1)[0]


@register_graph_pass
def substitute_loop_kernels(gm, example_inputs=None):
    return substitute_with_kernels_pass(
        gm, example_inputs, cache_dir=os.path.join(LOOP_DIR, "kernel_cache")
    )


def _config(*, pack: bool):
    config = example_recipe.dsv3_671b_base()
    run = os.path.abspath(os.environ["RUN"])
    os.makedirs(run, exist_ok=True)
    config.dump_folder = run
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(run, "tb")
    config.compile.extra_passes = (
        [*fusion_regions.DEFAULT_SELECTION, "substitute_loop_kernels"] if pack else []
    )
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


def graph_base_prof():
    return _prof(_config(pack=False))


def graph_pack_prof():
    return _prof(_config(pack=True))


def graph_base_tps():
    return _tps(_config(pack=False))


def graph_pack_tps():
    return _tps(_config(pack=True))

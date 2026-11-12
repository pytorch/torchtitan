# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture recipes for the finder-substrate experiment (see ../PROTOCOL.md).

One eager config per model; the graph config is that config converted to GraphTrainer, so both arms run
the same recipe. No recompute on either side, local_compile off, deterministic, seed 42, one microbatch
per step, CUDA graphs off, step 10 profiled after 3 warmup steps.

    OUT=<dir> PYTHONPATH=<this dir>:<fusion_finder scripts>:. NGPU=<n> LOCAL_RANK=0 \\
        python -m torchtitan.train --module recipes --config <model>_<eager|graph> --comm-backend fake

Eager artifacts: Kineto (CPU + CUDA, shapes, Python stacks) with a "## mod <fqn> [<class>] ##"
record_function around every module call, and an execution trace of the same step.
Graph artifacts: Kineto with the same settings, per-node labels (custom_codegen_pass), and graph dumps
(fusion_finder_passes: pre_inductor and final node records and readable graphs).
"""

import json
import os
import threading
from dataclasses import dataclass

import torch
from torch.profiler import ExecutionTraceObserver

from torchtitan.config import derive
from torchtitan.observability import profiler as tt_profiler
from torchtitan.observability.metrics import TensorBoardLogger
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

PROFILE_STEP = 10
WARMUP = 3


class CaptureProfiler(Profiler):
    """Kineto with Python stacks, plus an execution trace when CAPTURE_ET=1."""

    @dataclass(kw_only=True, slots=True)
    class Config(Profiler.Config):
        pass

    def build_torch_profiler(self, *, global_step, base_folder, leaf_folder):
        cfg = self._config
        trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
        os.makedirs(trace_dir, exist_ok=True)
        rank = torch.distributed.get_rank()
        et = None
        if os.environ.get("CAPTURE_ET") == "1":
            et = ExecutionTraceObserver().register_callback(
                os.path.join(trace_dir, f"rank{rank}_execution_trace.json")
            )

        def trace_handler(prof):
            d = os.path.join(trace_dir, f"iteration_{prof.step_num}", leaf_folder)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, f"rank{rank}_trace.json.gz")
            tt_profiler.logger.info(f"Dumping Kineto trace to {out}")
            prof.export_chrome_trace(out)
            tt_profiler.logger.info("Finished dumping Kineto trace")

        wait = cfg.profile_freq - (cfg.profiler_active + cfg.profiler_warmup)
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait,
                warmup=cfg.profiler_warmup,
                active=cfg.profiler_active,
                repeat=cfg.profiler_repeat or 0,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
            execution_trace_observer=et,
        )
        prof.__enter__()
        prof.step_num = global_step
        return prof


def _capture_profiler(out):
    return CaptureProfiler.Config(
        enable_profiling=True,
        enable_memory_snapshot=False,
        save_traces_folder=os.path.join(out, "traces"),
        profile_freq=PROFILE_STEP,
        profiler_warmup=WARMUP,
        profiler_active=1,
        profiler_repeat=1,
    )


_tls = threading.local()


def _stack():
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


def _install_module_scopes(model: torch.nn.Module) -> int:
    count = 0
    for fqn, mod in model.named_modules():
        label = f"## mod {fqn or '<root>'} [{type(mod).__name__}] ##"

        def pre_hook(m, args, kwargs, _label=label):
            rf = torch.autograd.profiler.record_function(_label)
            rf.__enter__()
            _stack().append(rf)

        def post_hook(m, args, kwargs, output):
            if _stack():
                _stack().pop().__exit__(None, None, None)

        mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
        mod.register_forward_hook(post_hook, with_kwargs=True, always_call=True)
        count += 1
    return count


class CaptureTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        n = sum(_install_module_scopes(m) for m in self.engine.model_parts)
        tt_profiler.logger.info(f"[capture] module scopes on {n} modules")


def _common(config):
    out = os.path.abspath(os.environ["OUT"])
    os.makedirs(out, exist_ok=True)
    config.model.local_compile.regions = []
    config.activation_remat = None
    config.debug.seed = 42
    config.debug.deterministic = True
    config.training.steps = PROFILE_STEP
    config.training.disable_accelerator_graphs = True
    config.dump_folder = out
    if not any(isinstance(c, TensorBoardLogger.Config) for c in config.metrics.loggers):
        config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(out, "tb")
    return config, out


# ---- per-model eager configs (the graph config is converted from these) ----


def _dsv3():
    from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b_dist_moe_mxfp8

    config = deepseek_v3_671b_dist_moe_mxfp8()
    # The DSv3 loop's layout (fusion_finder example_recipe.dsv3_671b_base).
    config.parallelism.data_parallel_shard_degree = 256
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.fsdp_defer_gradient_reduction = False
    config.training.num_tokens_per_train_step = -1
    # Under the fake PG every EP peer aliases this rank; EP x covers the worst-case receive.
    config.dist_moe_runtime.scratch_capacity_factor = float(
        config.parallelism.expert_parallel_degree
    )
    return config


def _qwen3_30b():
    from torchtitan.models.common.attention.attention import VarlenInnerAttention

    from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
        qwen3_30b_a3b_dist_moe_mxfp8_8gpu,
    )

    config = qwen3_30b_a3b_dist_moe_mxfp8_8gpu()
    config.training.num_tokens_per_train_step = -1
    # FA4 varlen attention, as in the Qwen3 loop.
    for layer in config.model.layers:
        inner = layer.attention.inner_attention
        assert isinstance(inner, VarlenInnerAttention.Config), type(inner)
        inner.sdpa_backends = ["FLASH_ATTENTION"]
    return config


def _llama3_8b():
    from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
    from torchtitan.components.data.text_datasets import DATASETS

    from torchtitan_recipes.models.llama3 import llama3_8b

    config = llama3_8b(seq_len=8192)
    # Step cost does not depend on token content: the repository's test tokenizer and C4 sample, as in
    # the benchmark recipes (no Llama assets on this host).
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader = GrainDataLoader.Config(
        max_num_documents=config.dataloader.max_num_documents,
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.checkpointer = None
    return config


def _graph_model_cls(model):
    if model == "dsv3":
        from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
            GraphTrainerDeepSeekV3Model,
        )

        return GraphTrainerDeepSeekV3Model.Config
    if model == "qwen3_30b":
        from torchtitan.experiments.graph_trainer.qwen3.model import (
            GraphTrainerQwen3Model,
        )

        return GraphTrainerQwen3Model.Config
    from torchtitan.experiments.graph_trainer.llama3.model import (
        GraphTrainerLlama3Model,
    )

    return GraphTrainerLlama3Model.Config


_BASES = {"dsv3": _dsv3, "qwen3_30b": _qwen3_30b, "llama3_8b": _llama3_8b}


def _eager(model):
    config, out = _common(_BASES[model]())
    config.profiler = _capture_profiler(out)
    os.environ["CAPTURE_ET"] = "1"
    _write_setup(out, model, "eager", config)
    return derive(config, CaptureTrainer.Config)


def _graph(model):
    from fusion_finder_passes import enable_fusion_finder

    from torchtitan.experiments.graph_trainer.configs import (
        GraphTrainerCompileConfig,
        to_graph_trainer_config,
    )

    config, out = _common(_BASES[model]())
    if model == "dsv3":
        # As graph_trainer_deepseek_v3_671b_dist_moe_mxfp8: functional weight-gradient accumulation.
        from torchtitan.models.common.dist_moe import DistMoeRoutedExperts

        for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config):
            experts.inplace_wgrad_accum = False
    config = to_graph_trainer_config(config, _graph_model_cls(model))
    config.compile = GraphTrainerCompileConfig(memory_policy="none")
    config = enable_fusion_finder(
        config, out, profile_step=PROFILE_STEP, profiler_warmup=WARMUP
    )
    config.profiler = _capture_profiler(out)
    os.environ["CAPTURE_ET"] = "0"
    _write_setup(out, model, "graph", config)
    return config


def _write_setup(out, model, arm, config):
    setup = {
        "model": model,
        "arm": arm,
        "trainer_config": type(config).__qualname__,
        "model_config": type(config.model).__qualname__,
        "parallelism": {
            k: getattr(config.parallelism, k)
            for k in (
                "data_parallel_shard_degree",
                "expert_parallel_degree",
                "tensor_parallel_degree",
                "pipeline_parallel_degree",
            )
        },
        "num_tokens_per_microbatch_per_dp_rank": config.training.num_tokens_per_microbatch_per_dp_rank,
        "num_tokens_per_train_step": config.training.num_tokens_per_train_step,
        "activation_remat": repr(config.activation_remat),
        "memory_policy": getattr(
            getattr(config, "compile", None), "memory_policy", None
        ),
        "local_compile_regions": list(config.model.local_compile.regions),
        "override_imports": list(config.override.imports),
        "deterministic": config.debug.deterministic,
        "profiled_step": PROFILE_STEP,
    }
    with open(os.path.join(out, "capture_setup.json"), "w") as f:
        json.dump(setup, f, indent=1, default=str)


def dsv3_eager():
    return _eager("dsv3")


def dsv3_graph():
    return _graph("dsv3")


def qwen3_30b_eager():
    return _eager("qwen3_30b")


def qwen3_30b_graph():
    return _graph("qwen3_30b")


def llama3_8b_eager():
    return _eager("llama3_8b")


def llama3_8b_graph():
    return _graph("llama3_8b")

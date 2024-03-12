# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import os

from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List

import numpy as np

# torch imports
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.tensor.parallel import loss_parallel

from torchtrain.checkpoint import CheckpointManager, IntervalType
from torchtrain.config_manager import JobConfig

# torchtrain related
from torchtrain.datasets import create_tokenizer, dataloader_fn
from torchtrain.float8_linear import build_fp8_linear
from torchtrain.logging_utils import init_logger, rank0_log
from torchtrain.lr_scheduling import get_lr_scheduler
from torchtrain.meta_init import meta_model_init
from torchtrain.metrics import build_metric_logger, get_num_params, GPUMemoryMonitor

from torchtrain.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtrain.parallelisms import models_parallelize_fns, ParallelDims

from torchtrain.profiling import maybe_run_profiler
from torchtrain.utils import Color, dist_max, dist_mean

from offload_aux import *
from hyperopt import atpe, fmin, hp, tpe, Trials
import gc


import logging

_is_local_logging = True
if "SLURM_JOB_ID" in os.environ:
    _is_local_logging = False

MAXIMUM_LATENCY_ALLOWED = 50
MAXIMUM_MEMORY_ALLOWED = 100


def clear_memory_usage_stats(device) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    # logging.info(f"memory usage reset = curr_max = {get_peak_memory_usage(device)}")


def get_peak_memory_usage(device) -> List[float]:
    MEMORY_METRICS = ["active_bytes.all.peak", "num_alloc_retries", "num_ooms"]
    BYTES_TO_GB = 1024**3
    memory_usages = []
    for metric in MEMORY_METRICS:
        curr_metrics_usage = (
            torch.cuda.memory_stats(device).get(metric, 0) / BYTES_TO_GB
            if torch.cuda.is_available()
            else 0
        )
        memory_usages.append(curr_metrics_usage)
    # logging.info(f"curr_max_mem = {memory_usages[0]}")
    return memory_usages


@dataclass
class TrainState:
    step: int = 0
    current_loss: float = -1
    losses: List[float] = field(default_factory=list)
    iter_times: List[float] = field(default_factory=list)
    data_load_times: List[float] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "current_loss": torch.tensor(self.current_loss, dtype=torch.float32),
            "losses": torch.tensor(self.current_loss, dtype=torch.float32),
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        self.current_loss = state_dict["current_loss"].item()
        self.losses = state_dict["losses"].tolist()


def build_optimizer(model, job_config: JobConfig):
    # build optimizer
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    if name == "Adam":
        # TODO: make the optimizer options configurable by toml/cmd args
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
    else:
        raise NotImplementedError(f"optimizer {name} not added")

    return optimizer


def build_grad_scaler(model):
    # apply gradient scaling if mixed precision training is enabled with fp16 param dtype
    if model.mixed_precision.param_dtype == torch.float16:
        enable_grad_scaling = True
        rank0_log("Enabling gradient scaling for mixed precision training.")
    else:
        enable_grad_scaling = False
        rank0_log("Gradient scaling not enabled.")

    return ShardedGradScaler(enabled=enable_grad_scaling)


def main(job_config: JobConfig):
    init_logger()
    # init world mesh
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        sp=job_config.training.sequence_parallel_degree,
        pp=job_config.training.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    rank0_log(f"Starting job: {job_config.job.description}")
    model_name = job_config.model.name
    rank0_log(f"Building {model_name}")
    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    build_dataloader_fn = dataloader_fn[job_config.training.dataset]
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    data_loader = build_dataloader_fn(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )
    rank0_log(
        f"{Color.green}Built Dataloader for '{job_config.training.dataset}' dataset.{Color.reset}"
    )

    # build model
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    model_config.vocab_size = tokenizer.n_words

    # build model using meta init
    with meta_model_init():
        model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(model, job_config)

    # log model size
    model_param_count = get_num_params(model)

    if _is_local_logging:
        rank0_log(
            f"{Color.blue}Model {model_name} {job_config.model.flavor} {Color.red}size: {model_param_count:,}"
            f" total parameters{Color.reset}"
        )
    else:
        rank0_log(
            f"{model_name} {job_config.model.flavor} size: {model_param_count:,} total parameters"
        )

    gpu_metrics = GPUMemoryMonitor("cuda")
    rank0_log(f"GPU memory usage: {gpu_metrics}")

    # apply PTD parallelisms + AC
    model = models_parallelize_fns[model_name](
        model, world_mesh, parallel_dims, job_config
    )

    # to use FSDP-customized gradient scaler and gradient clipping solutions
    assert isinstance(model, FSDP)

    # build optimizer after apply parallelisms to the model
    optimizer = build_optimizer(model, job_config)
    scheduler = get_lr_scheduler(optimizer, job_config)

    scaler = build_grad_scaler(model)

    metric_logger = build_metric_logger(job_config)

    # torch.compile model for improved performance
    if job_config.training.compile:
        if job_config.training.enable_selective_ac:
            torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = (
                True
            )
        rank0_log(f"Compiling model {model_name} with torch.compile...")
        model = torch.compile(
            model,
        )

    train_state = TrainState()

    # train loop
    model.train()

    checkpoint = CheckpointManager(
        model=model,
        optimizer=optimizer,
        states={"train_state": train_state},
        folder=job_config.training.checkpoint_folder,
        interval_type=(
            IntervalType.SECONDS
            if job_config.training.checkpoint_interval_type == "seconds"
            else IntervalType.STEPS
        ),
        interval=job_config.training.checkpoint_interval,
    )
    checkpoint.load()

    data_iterator = iter(data_loader)

    offload_candidates = []
    offload_candidate_modules = {}
    #### INITIALIZE COPILOT OFFLOAD ####
    for name, module in model.named_modules():
        if "TransformerBlock" in str(type(module)):
            offload_candidates.append(name)
            offload_candidate_modules[name] = module

        # globally, perform a set interaction to filter out modules not n all candidates
    allgather_handle = [None] * torch.distributed.get_world_size()

    torch.distributed.all_gather_object(allgather_handle, offload_candidates)

    # intersect
    allgather_handle_sets = [set(x) for x in allgather_handle]

    final_offload_candidates = set.intersection(*allgather_handle_sets)

    final_offload_candidates_ordered = sorted(final_offload_candidates)
    _legokit_original_fwd = {}
    name_to_module = {}
    module_sequencer = {}
    module_offloaded = {}
    _device = torch.device(f"cuda:{torch.distributed.get_rank() % 8}") # okay for both HSDP and FSDP
    MAXIMUM_ALLOWED_MEMORY_USAGE = (
        torch.cuda.get_device_properties(_device).total_memory / 1024**3
    ) * 0.85

    for name in final_offload_candidates_ordered:
        module = offload_candidate_modules[name]
        logging.info(
            f"INIT:: registering {type(module)}'s forward as {digest_forward_name(module.forward)}"
        )
        _legokit_original_fwd[module] = module.forward
        name_to_module[name] = module
        module_sequencer[name] = len(module_sequencer)
        module_offloaded[name] = 1

        toggle_offload_simple(module, _legokit_original_fwd[module], True, False)

        if torch.distributed.get_rank() == 0:
            logging.info(f"{name} identified as a candidate for offloading")

    logging.info(f"registered {len(module_sequencer)} modules")

    launcher_job_name = os.environ.get("launcher_job_name", "LIKELY_LOCAL_RUN")

    segs = launcher_job_name.split("-")

    if len(segs) > 1:
        launcher_job_name = "-".join(segs[:-1])  # the last nounce is omitted

    CURRENT_JOB_IDENTIFIER = launcher_job_name
    search_engine = (
        "knapsack" if "knapsack" in launcher_job_name.lower() else "hyperopt"
    )

    MODEL_TYPE = (
        PerformanceModelTrainerXGB
        if "xgb" in launcher_job_name.lower()
        else PerformanceModelTrainer
    )

    logging.info(
        f"******** INFERRED PARAMETERS id = {CURRENT_JOB_IDENTIFIER}, engine = {search_engine}, model = {MODEL_TYPE}"
    )

    target_manifold_filename_qps = (
        f"{CURRENT_JOB_IDENTIFIER}-{torch.distributed.get_rank()}_qps.pickle"
    )

    should_collapse = (
        torch.distributed.get_world_size() * len(module_sequencer) > 4096
        and MODEL_TYPE is not PerformanceModelTrainerXGB
    )

    qps_model = get_trainer(
        model_type=MODEL_TYPE,
        target_manifold_filename=target_manifold_filename_qps,
        target_manifold_rootname=CURRENT_JOB_IDENTIFIER,
        name="latency estimator",
        num_embeddings=(
            len(_legokit_original_fwd)
            if should_collapse
            else len(_legokit_original_fwd) * torch.distributed.get_world_size()
        ),
        normalization_value=1 / MAXIMUM_LATENCY_ALLOWED,
        rank_loss_margin=1e-4,  # 1ms
        use_dual_embedding_table=True,
        normalization_type="relu",
        collapse_inputs_by_world_size=should_collapse,
        device=_device,
    )

    target_manifold_filename_mem = (
        f"{CURRENT_JOB_IDENTIFIER}-{torch.distributed.get_rank()}_mem.pickle"
    )

    memory_model = get_trainer(
        model_type=MODEL_TYPE,
        target_manifold_filename=target_manifold_filename_mem,
        target_manifold_rootname=CURRENT_JOB_IDENTIFIER,
        name="peak memory estimator",
        num_embeddings=len(_legokit_original_fwd),
        normalization_value=1 / MAXIMUM_MEMORY_ALLOWED,
        rank_loss_margin=1e-5,  # 1MB
        use_dual_embedding_table=True,
        normalization_type="relu",
        collapse_inputs_by_world_size=False,
        device=_device,
    )

    cycle = 10
    cycle_increment_multiplier = 1.01
    cycle_increment = 10
    QPS_WINDOW_SIZE = 65536
    gaussian_qps_counter = GaussianMeter(QPS_WINDOW_SIZE, "qps_reading")
    gaussian_mem_counter = GaussianMeter(QPS_WINDOW_SIZE, "peak_memory")
    # self.gaussian_idle_memory_counter = GaussianMeter(
    #     self.QPS_WINDOW_SIZE, "idle_memory"
    # )
    best_suggest = None
    decision_threshold = 50

    disable_loggers = [
        "hyperopt.tpe",
        "hyperopt.fmin",
        "hyperopt.pyll.base",
    ]
    for logger in disable_loggers:
        logging.getLogger(logger).setLevel(logging.ERROR)

    # take the first reading
    # self.gaussian_idle_memory_counter.add(peek_current_memory_usage(self._device))
    TRAINING_DATA_HISTORY_SIZE = 2048
    historical_qps_training_input = GaussianMeter(
        TRAINING_DATA_HISTORY_SIZE, "qps_training_data"
    )
    historical_qps_training_label = GaussianMeter(
        TRAINING_DATA_HISTORY_SIZE, "qps_training_label"
    )
    historical_peak_mem_training_input = GaussianMeter(
        TRAINING_DATA_HISTORY_SIZE, "peak_mem_training_data"
    )
    # self.historical_idle_mem_training_input = GaussianMeter(
    #     self.TRAINING_DATA_HISTORY_SIZE, "idle_mem_training_data"
    # )
    historical_peak_mem_training_label = GaussianMeter(
        TRAINING_DATA_HISTORY_SIZE, "mem_training_label"
    )

    qps_training_data = dbg_download_training_data_untransferrable_from_manifold(
        target_manifold_filename_qps.replace(".pickle", ".json"),
        CURRENT_JOB_IDENTIFIER,
    )

    historical_qps_training_input.window_metric.extend(qps_training_data["inputs"])
    historical_qps_training_label.window_metric.extend(qps_training_data["labels"])
    updates = len(historical_qps_training_label.window_metric)

    mem_training_data = dbg_download_training_data_untransferrable_from_manifold(
        target_manifold_filename_mem.replace(".pickle", ".json"),
        CURRENT_JOB_IDENTIFIER,
    )

    historical_peak_mem_training_input.window_metric.extend(
        mem_training_data["inputs"]
    )
    historical_peak_mem_training_label.window_metric.extend(
        mem_training_data["labels"]
    )

    tpe_trials = Trials()

    hyperopt_use_hybrid_scheme = True
    hyperopt_swicth_to_fmin = False

    clear_memory_usage_stats(_device)

    def assert_training_data_integrity():
        assert len(historical_qps_training_input.window_metric) == len(
            historical_qps_training_label.window_metric
        )
        assert len(historical_peak_mem_training_input.window_metric) == len(
            historical_peak_mem_training_label.window_metric
        )
        assert len(historical_peak_mem_training_input.window_metric) == len(
            historical_peak_mem_training_input.window_metric
        )
        if search_engine == "knapsack":
            assert len(historical_peak_mem_training_label.window_metric) == len(
                historical_qps_training_label.window_metric
            ), f"{historical_peak_mem_training_label.window_metric} vs {historical_qps_training_label.window_metric}"

            for i in range(
                min(
                    len(historical_peak_mem_training_input.window_metric),
                    len(module_sequencer) + 1,
                )
            ):
                assert (
                    historical_peak_mem_training_input.window_metric[i]
                    == historical_qps_training_input.window_metric[i]
                )
                onehot = [1] * len(module_sequencer)
                if i != 0:
                    onehot[i - 1] = 0
                assert (
                    onehot == historical_peak_mem_training_input.window_metric[i]
                ), f"expected {i}-th exploration to be {onehot}, but got {historical_peak_mem_training_input.window_metric[i]}"

    def show_digest():
        qps_info = f"[{torch.distributed.get_rank()}]    last QPS train label (digested): {historical_qps_training_label.window_metric[-1]}, P50: {gaussian_qps_counter.percentile(50)}, MIN: {gaussian_qps_counter.percentile(0)}, MAX = {gaussian_qps_counter.percentile(100)}"
        mem_info = f"[{torch.distributed.get_rank()}]    last MEM train label (digested): {historical_peak_mem_training_label.window_metric[-1]}, P50: {gaussian_mem_counter.percentile(50)}, MIN: {gaussian_mem_counter.percentile(0)}, MAX = {gaussian_mem_counter.percentile(100)}"

        offload_all_qps = len(historical_qps_training_input.window_metric[-1]) * [
            0
        ]
        offload_all_mem = len(
            historical_peak_mem_training_input.window_metric[-1]
        ) * [0]

        offload_none_allqps_info = f"[{torch.distributed.get_rank()}]    Offload All Off QPS: {qps_model.inference_once([offload_all_qps])}"
        offload_none_allmem_info = f"[{torch.distributed.get_rank()}]    Offload All Off MEM: {memory_model.inference_once([offload_all_mem])}"
        output_information = "\n".join(
            [qps_info, mem_info, offload_none_allqps_info, offload_none_allmem_info]
        )

        if any(module_offloaded.values()):
            logging.info(output_information)


    with maybe_run_profiler(job_config) as torch_profiler:
        checkpoint.reset()
        # variables used to keep info for metrics logging
        losses_since_last_log: List[float] = []
        nwords_since_last_log = 0
        time_last_log = timer()
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            # get batch
            data_load_start = timer()
            batch = next(data_iterator)
            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            data_load_time = round(timer() - data_load_start, 4)
            train_state.data_load_times.append(data_load_time)
            nwords_since_last_log += labels.numel()

            optimizer.zero_grad()

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()
            #
            # forever training?
            if train_state.step == cycle:
                cycle += int(cycle_increment)
                cycle_increment *= cycle_increment_multiplier
                # logging.info(
                #     f" ✅ next {self.cycle} cycles, increment = {self.cycle_increment}"
                # )
                # read, then clear
                lat = gaussian_qps_counter.percentile(10)
                # .mean_confidence_interval()[-1]
                # clear
                # train performance model
                # obtain activated indices:
                # offloaded_indices = []
                offload_inputs = list(
                    digest_module_offload(module_sequencer, module_offloaded)
                )

                # local decision is not enough to determine global qps. we need to collect everyone's decisions.
                global_decisions = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(global_decisions, offload_inputs)

                assert global_decisions[torch.distributed.get_rank()] == offload_inputs
                global_decisions = sum(global_decisions, [])

                # idle_mem = self.gaussian_idle_memory_counter.percentile(100)
                # peak_mem = self.gaussian_mem_counter.mean_confidence_interval(0.99)[-1]
                peak_mem = gaussian_mem_counter.percentile(100)

                if historical_qps_training_input.add_if_not_exist(
                    global_decisions
                    if search_engine == "hyperopt"
                    else offload_inputs
                ):
                    historical_qps_training_label.add(lat)
                    logging.info(
                       f"a valid QPS config has resulted in a new training label: {lat}"
                    )
                else:
                    idx = historical_qps_training_input.find(
                        global_decisions
                        if search_engine == "hyperopt"
                        else offload_inputs
                    )
                    old = historical_qps_training_label.window_metric[idx]
                    historical_qps_training_label.replace_window_item(idx, lat)
                    new = historical_qps_training_label.window_metric[idx]
                    logging.info(
                        f"a valid QPS config has resulted in an updated training label@{idx} (out of {len(historical_qps_training_input.window_metric)}): {old} -> {new}"
                    )

                # total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if historical_peak_mem_training_input.add_if_not_exist(
                    offload_inputs
                ):
                    # assume heterogeneous
                    historical_peak_mem_training_label.add(peak_mem)
                    logging.info(
                        "a valid MEM config has resulted in a new training label"
                    )
                else:
                    idx = historical_peak_mem_training_input.find(offload_inputs)
                    old = historical_peak_mem_training_label.window_metric[idx]
                    historical_peak_mem_training_label.replace_window_item(
                        idx, peak_mem
                    )
                    new = historical_peak_mem_training_label.window_metric[idx]
                    logging.info(
                        f"a valid MEM config has resulted in an updated training label@{idx} (out of {len(historical_peak_mem_training_input.window_metric)}): {old} -> {new}"
                    )

                assert_training_data_integrity()

                if search_engine == "hyperopt":
                    loss_lat, pred_lat = qps_model.train_once(
                        historical_qps_training_input.window_metric,
                        # None,
                        historical_qps_training_label.window_metric,
                        True,  # because offloadis slower than not offload
                    )
                    loss_mem, pred_mem = memory_model.train_once(
                        historical_peak_mem_training_input.window_metric,
                        # self.historical_idle_mem_training_input.window_metric,
                        historical_peak_mem_training_label.window_metric,
                        False,  # because offloading causes higher memory usage
                    )

                    logging.info(
                        f"<=> at iter (window = {cycle_increment}) = {train_state.step}, updates = {updates}, latency loss = {loss_lat}, memory loss = {loss_mem}, lat = {lat} vs pred {pred_lat[-1]}, peak_mem = {peak_mem} vs pred {pred_mem[-1]}, idle_mem = OMITTED"
                    )

                atpe_ss = {
                    name: hp.randint(name, 0, 2) for name in module_sequencer
                }
                # beam search, or better yet, bayesian optimizers
                found_all_zeros = False
                ___DBG_PROBED_ITEMS = {}

                def atpe_objective(params):
                    nonlocal found_all_zeros

                    cannonical_repr = digest_module_offload(
                        module_sequencer, params
                    )
                    if cannonical_repr == tuple([0] * len(module_sequencer)):
                        found_all_zeros = True

                    INVALID_BOUND = MAXIMUM_LATENCY_ALLOWED
                    get_active_indices = list(cannonical_repr)

                    # how fast can this run? will it oom?
                    # assuming our choice is independent of reotes
                    expected_mem = memory_model.inference_once(
                        [get_active_indices]
                    )
                    # logging.info(f"inference memory okay {expected_mem}")

                    if expected_mem > MAXIMUM_ALLOWED_MEMORY_USAGE:
                        # thrashing or OOM
                        # if get_active_indices in hamming_configs:
                        ret = INVALID_BOUND
                        # bound it a bit, otherwise it causes really sharp curves in gaussian priors
                    else:
                        global_active_indices = list(global_decisions)
                        local_cnt = len(get_active_indices)
                        my_rank = torch.distributed.get_rank()
                        global_active_indices[
                            my_rank * local_cnt : (my_rank + 1) * local_cnt
                        ] = get_active_indices
                        expected_lat = qps_model.inference_once(
                            [global_active_indices]
                        )
                        # logging.info(f"inference QPS okay {expected_lat}")
                        ret = expected_lat

                    ___DBG_PROBED_ITEMS[cannonical_repr] = ret
                    assert ret != float(
                        "nan"
                    ), f"ret is NAN, atpe = {cannonical_repr}, expected_lat = {expected_lat}"
                    return ret

                found_all_zeros = found_all_zeros or [0] * len(
                    offload_inputs
                ) not in greedy_descend(module_sequencer, module_offloaded)
                # checked if found, or [0,...0] is not proposed

                # determine if we need to generate possible trials
                if updates > decision_threshold:
                    trials = (
                        generate_trials_to_calculate(
                            greedy_descend(module_sequencer, module_offloaded)
                        )
                        if random.randint(0, 1) == 0 or True
                        else tpe_trials  # record numbers
                    )
                else:
                    trials = None
                # else:
                #     trials = generate_trials_to_calculate(
                #         greedy_descend(self.module_offloaded, self.module_sequencer)
                #     )

                if search_engine == "hyperopt":
                    if (
                        hyperopt_use_hybrid_scheme
                        and not hyperopt_swicth_to_fmin
                    ):
                        best, finished_knapsack_bootstrap = naive_fmin(
                            cannonical_names=list(module_sequencer.keys()),
                            current_qps_explorations=historical_qps_training_input.window_metric,
                            current_qps_explorations_labels=historical_qps_training_label.window_metric,
                            current_memory_explorations=historical_peak_mem_training_input.window_metric,
                            current_memory_explorations_labels=historical_peak_mem_training_label.window_metric,
                            max_allowed_memory=MAXIMUM_ALLOWED_MEMORY_USAGE,
                        )
                        if finished_knapsack_bootstrap:
                            hyperopt_swicth_to_fmin = True
                    else:
                        best = fmin(
                            fn=atpe_objective,
                            space=atpe_ss,
                            algo=tpe.suggest,
                            max_evals=500 + len(tpe_trials.trials),
                            show_progressbar=False,
                            trials=trials,
                        )
                elif search_engine == "knapsack":
                    best, _ = naive_fmin(
                        cannonical_names=list(module_sequencer.keys()),
                        current_qps_explorations=historical_qps_training_input.window_metric,
                        current_qps_explorations_labels=historical_qps_training_label.window_metric,
                        current_memory_explorations=historical_peak_mem_training_input.window_metric,
                        current_memory_explorations_labels=historical_peak_mem_training_label.window_metric,
                        max_allowed_memory=MAXIMUM_ALLOWED_MEMORY_USAGE,
                    )

                proposal = [
                    best[name].item() if hasattr(best[name], "item") else best[name]
                    for name in module_sequencer
                ]
                global_proposal = list(global_decisions)
                my_rank = torch.distributed.get_rank()
                global_proposal[
                    my_rank
                    * len(module_sequencer) : (my_rank + 1)
                    * len(module_sequencer)
                ] = proposal

                if search_engine == "hyperopt":
                    show_digest()

                assert (
                    found_all_zeros or trials is None
                ), f"all 0s are not tested. Generated trials = {sorted(___DBG_PROBED_ITEMS)}"

                # CFG1 is much better then CFG2
                def is_dominated(cfg1, cfg2):
                    # returns true if cfg1 is specifically better
                    if all(x1 <= x2 for x1, x2 in zip(cfg1, cfg2)):
                        return True
                    else:
                        return False

                def jump_probability(global_decisions, target):
                    if (
                        search_engine == "hyperopt"
                        and hyperopt_swicth_to_fmin
                    ):
                        pred1 = qps_model.inference_once([global_decisions])
                        pred2 = qps_model.inference_once([target]) + 1e-9
                        return abs(pred1 / pred2), pred1, pred2
                    else:
                        return 1, 1, 1

                if is_dominated(offload_inputs, proposal) is False and (
                    is_dominated(proposal, offload_inputs)
                    or random.random()
                    <= jump_probability(global_decisions, global_proposal)[0]
                ):
                    # foreach best config, try to set appropriate offloading decisions
                    for name in module_offloaded:
                        on = best[name]
                        toggle_offload_simple(
                            name_to_module[name],
                            _legokit_original_fwd,
                            on,
                            already_on=module_offloaded[name],
                        )
                        module_offloaded[name] = (
                            on.item() if hasattr(on, "item") else on
                        )

                    updates += 1

                    all_offload_off = tuple([0] * len(module_sequencer))

                    logging.info(
                        f"⚙️ {offload_inputs} -> {proposal}, because is_dominated = {is_dominated(offload_inputs, proposal)}, jump_probability = {jump_probability(global_decisions, global_proposal)} REF = {___DBG_PROBED_ITEMS[all_offload_off] if all_offload_off in ___DBG_PROBED_ITEMS else 'N/A'}, popcnt = {sum(offload_inputs)}/{len(offload_inputs)} -> {sum(proposal)}, "  # ___DBG_PROBE_KEYS = {list(___DBG_PROBED_ITEMS.keys())}
                    )

                    # only resets performance counter if updated
                    gaussian_qps_counter.reset()
                    gaussian_mem_counter.reset()
                else:
                    # no need to change
                    logging.info(
                        f"⏩ A new proposal is skipped, because original proposal is better. Old = {offload_inputs} vs New = {proposal}"
                    )

                updated_global_decisions = [None] * torch.distributed.get_world_size()
                updated_local_decision = list(
                    digest_module_offload(module_sequencer, module_offloaded)
                )
                torch.distributed.all_gather_object(
                    updated_global_decisions, updated_local_decision
                )

                assert (
                    updated_global_decisions[torch.distributed.get_rank()]
                    == updated_local_decision
                )
                updated_global_decisions = sum(
                    updated_global_decisions, []
                )  # obtain updated global proposal

                if historical_qps_training_input.add_if_not_exist(
                    updated_global_decisions
                    if search_engine == "hyperopt"
                    else proposal
                ):
                    anticipated_qps = MAXIMUM_LATENCY_ALLOWED
                    historical_qps_training_label.add(anticipated_qps)
                    logging.info("a preemptive QPS config has been registered")

                if historical_peak_mem_training_input.add_if_not_exist(
                    updated_local_decision
                ):
                    anticipated_mem = MAXIMUM_MEMORY_ALLOWED
                    historical_peak_mem_training_label.add(anticipated_mem)
                    logging.info("a preemptive MEM config has been registered")

                assert_training_data_integrity()
                # best choices are not required to be synchronized across devices
                save_trainer(
                    target_manifold_filename_qps,
                    CURRENT_JOB_IDENTIFIER,
                    qps_model,
                )
                save_trainer(
                    target_manifold_filename_mem,
                    CURRENT_JOB_IDENTIFIER,
                    memory_model,
                )

                dbg_upload_training_data_untransferrable_to_manifold(
                    target_manifold_filename_qps.replace(".pickle", ".json"),
                    CURRENT_JOB_IDENTIFIER,
                    historical_qps_training_input.window_metric,
                    historical_qps_training_label.window_metric,
                )
                dbg_upload_training_data_untransferrable_to_manifold(
                    target_manifold_filename_mem.replace(".pickle", ".json"),
                    CURRENT_JOB_IDENTIFIER,
                    historical_peak_mem_training_input.window_metric,
                    historical_peak_mem_training_label.window_metric,
                )
                gc.collect()
                gc.collect()  # force finalizers
                clear_memory_usage_stats(_device)  # reset memory stats
                torch.distributed.barrier()
                torch.cuda.synchronize()

            # must have finished compute of actual workload
            # because hyperopt uses comp stream
            # here must be idle memory
            # also made sure we have cleared memory stat, now everything we record will reflect the current change

            pred = model(input_ids)

            with loss_parallel() if parallel_dims.loss_parallel_enabled else contextlib.nullcontext():
                loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

                # backward on scaled loss to create scaled gradients
                scaler.scale(loss).backward()

            # clip gradients (after unscaling gradients of the optimizer's params)
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(job_config.training.max_norm)

            # optimizer step
            # If gradients don't contain infs/NaNs, optimizer.step() is then called;
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # updates the scale for next iteration
            scaler.update()

            # training iteration complete
            end_timer.record()
            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)
            gaussian_qps_counter.add(curr_iter_time)
            train_state.iter_times.append(curr_iter_time)

            # if profiler is active
            if torch_profiler:
                torch_profiler.step()

            train_state.current_loss = loss.item()
            train_state.losses.append(train_state.current_loss)
            losses_since_last_log.append(train_state.current_loss)

            if train_state.step >= 5:
                # first iter has that fx things
                gaussian_qps_counter.add(curr_iter_time)
                # must be peak memory for this iteration
                peak_mem = get_peak_memory_usage(_device)[0]
                gaussian_mem_counter.add(peak_mem)

            # log metrics
            if (train_state.step - 1) % job_config.metrics.log_freq == 0:
                avg_loss, max_loss = (
                    np.mean(losses_since_last_log),
                    np.max(losses_since_last_log),
                )
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh),
                        dist_max(max_loss, dp_mesh),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (
                    time_delta * parallel_dims.model_parallel_size
                )

                gpu_mem_stats = gpu_metrics.get_current_stats(return_data=True)

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "memory_current/active(%)": gpu_mem_stats.active_curr,
                    "memory_current/allocated(%)": gpu_mem_stats.allocated_curr,
                    "memory_current/reserved(%)": gpu_mem_stats.reserved_curr,
                    "memory_peak/active(%)": gpu_mem_stats.active_peak,
                    "memory_peak/allocated(%)": gpu_mem_stats.allocated_peak,
                    "memory_peak/reserved(%)": gpu_mem_stats.reserved_peak,
                }
                metric_logger.log(metrics, step=train_state.step)

                losses_since_last_log.clear()
                nwords_since_last_log = 0
                time_last_log = timer()

            if _is_local_logging:
                rank0_log(
                    f"{Color.cyan}step: {train_state.step:>2}  {Color.green}loss: {round(train_state.current_loss,4):>7}"
                    f"  {Color.reset}iter: {Color.blue}{curr_iter_time:>7}{Color.reset}"
                    f"  data: {Color.blue}{data_load_time:>5}  {Color.reset}"
                    f"lr: {Color.yellow}{round(float(scheduler.get_last_lr()[0]), 8):<6}{Color.reset}"
                )
            else:
                rank0_log(
                    f"step: {train_state.step:>2}  loss: {round(train_state.current_loss,4):>7}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}  "
                    f"lr: {round(float(scheduler.get_last_lr()[0]), 8):<6}"
                )

            scheduler.step()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

    metric_logger.close()
    # calc and show average iter time, disregard first three iterations (warmup)
    if len(train_state.iter_times) > 3:
        avg_iter_time = np.mean(train_state.iter_times[3:])
        rank0_log(f"Average iter time: {avg_iter_time:.4f} seconds")
        avg_data_load_time = np.mean(train_state.data_load_times[3:])
        rank0_log(f"Average data load time: {avg_data_load_time:.4f} seconds")

    rank0_log(f"{gpu_metrics.get_current_stats()}")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)

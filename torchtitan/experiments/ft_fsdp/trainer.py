# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fault tolerant FSDP trainer.

Training runs plain FSDP over the active hosts. Every step the sharded
parameters and optimizer state are copied asynchronously to pinned CPU memory
and replicated to the next host. When a host fails, survivors catch the
communication error, the lighthouse quorum assigns a hot spare to the failed
slot, the ``nccl2`` process group is reconfigured in place, and every rank
restores the newest step that all ranks hold (the spare reads its shard from
its successor's replica).
"""

import json
import logging
import os
import pickle
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FSDPModule

from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim.utils import init_optim_state
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.ft_fsdp.config import FTFSDPConfig
from torchtitan.experiments.ft_fsdp.membership import (
    Assignment,
    DONE_KEY,
    EvictedError,
    Membership,
    STARTED_KEY,
    TrainingFinishedError,
)
from torchtitan.experiments.ft_fsdp.optim import FTOptim
from torchtitan.experiments.ft_fsdp.snapshot import (
    collect_state_tensors,
    Snapshotter,
    TransportPool,
)
from torchtitan.experiments.ft_fsdp.store import TimedStore
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

_ABORT_HOOK_ID = 0x46534450


def _event(name: str, **fields: Any) -> None:
    """Log a machine readable timing event."""
    logger.info(
        "FTFSDP_EVENT " + json.dumps({"event": name, "t": time.time(), **fields})
    )


def _exit(code: int) -> None:
    # NIXL can crash during interpreter teardown; skip it.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _connect_store(timeout: timedelta) -> dist.Store:
    addr = os.environ["FTFSDP_STORE_ADDR"]
    host, port = addr.rsplit(":", 1)
    return dist.TCPStore(
        host.strip("[]"),
        int(port),
        is_master=False,
        timeout=timeout,
        wait_for_workers=False,
        use_libuv=True,
    )


class FTFSDPTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        ft_fsdp: FTFSDPConfig = field(default_factory=FTFSDPConfig)

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            ft = self.ft_fsdp
            if ft.num_active_hosts < 2:
                raise ValueError("ft_fsdp.num_active_hosts must be >= 2")
            if ft.num_hosts < ft.num_active_hosts:
                raise ValueError("ft_fsdp.num_hosts must be >= num_active_hosts")
            p = self.parallelism
            if (
                p.data_parallel_replicate_degree != 1
                or p.tensor_parallel_degree != 1
                or p.context_parallel_degree != 1
                or p.pipeline_parallel_degree != 1
                or p.expert_parallel_degree != 1
            ):
                raise ValueError("ft_fsdp supports FSDP-only parallelism")
            if self.checkpointer is not None:
                raise ValueError(
                    "ft_fsdp keeps snapshots in memory; set checkpointer=None"
                )
            if not self.training.disable_cuda_graphs or self.optim.enable_cuda_graph:
                raise ValueError("ft_fsdp requires CUDA graphs to be disabled")
            if not isinstance(self.optim, FTOptim.Config):
                raise ValueError("ft_fsdp requires optim=FTOptim.Config(...)")
            if self.debug.seed is None:
                raise ValueError(
                    "ft_fsdp requires debug.seed so new hosts skip the seed broadcast"
                )
            if self.validator is not None:
                raise ValueError("ft_fsdp does not support validation")

    config: Config

    @record
    def __init__(self, config: Config) -> None:
        ft = config.ft_fsdp
        self.ft = ft
        self.finished = False
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # A recovery connects at most two links; spares create theirs while
        # they wait for a slot.
        self.transports = TransportPool(2)
        recovery_timeout = timedelta(seconds=ft.recovery_timeout_seconds)
        self.store = TimedStore(
            lambda: _connect_store(recovery_timeout),
            op_timeout=timedelta(seconds=ft.store_op_timeout_seconds),
            timeout=recovery_timeout,
        )
        self.membership = Membership(
            store=self.store,
            pg_store=_connect_store(recovery_timeout),
            lighthouse_addr=os.environ["TORCHFT_LIGHTHOUSE"],
            host=os.environ.get("FTFSDP_HOST_NAME", socket.gethostname()),
            host_index=int(os.environ["FTFSDP_HOST_INDEX"]),
            local_rank=local_rank,
            num_hosts=ft.num_hosts,
            num_slots=ft.num_active_hosts,
            procs_per_host=ft.procs_per_host,
            run_id=os.environ.get("FTFSDP_RUN_ID", "ftfsdp"),
            # Every reconfiguration resets to the init timeout so lazy init and
            # compilation on new hosts fit in the first step.
            pg_timeout=timedelta(seconds=config.comm.init_timeout_seconds),
            quorum_timeout=timedelta(seconds=ft.quorum_timeout_seconds),
            spare_quorum_timeout=recovery_timeout,
            heartbeat_interval=ft.heartbeat_interval_seconds,
        )
        _event("process_start", host=self.membership.host, uid=self.membership.uid)
        try:
            assignment = self.membership.next_assignment()
        except TrainingFinishedError:
            logger.info("training already finished; spare exiting")
            _exit(0)
        except EvictedError as e:
            logger.error(str(e))
            _exit(1)
        _event("assigned", gen=assignment.gen, rank=dist.get_rank())

        build_start = time.perf_counter()
        super().__init__(config)
        _event("trainer_built", seconds=time.perf_counter() - build_start)

        pg = dist.distributed_c10d._get_default_group()
        if pg.supports_abort_hooks:
            pg.register_abort_hook(_ABORT_HOOK_ID, self._on_abort)
        else:
            logger.warning(
                "process group has no abort hooks; snapshots may commit after a failed step"
            )

        self._init_optim_state()
        device_tensors, _ = self._state_tensors()
        self.snapshotter = Snapshotter(
            device_tensors,
            transports=self.transports,
            device=self.engine.device,
            ident=self.membership.ident,
            store=self.store,
            num_local=ft.num_local_snapshots,
            meta_capacity=ft.meta_capacity_bytes,
            interval=ft.snapshot_interval,
            procs_per_host=ft.procs_per_host,
            timeout=ft.recovery_timeout_seconds,
        )
        self.snapshotter.register_optimizer_hooks(self._inner_optimizers())
        self.num_recoveries = 0
        self._recover(assignment)

    # State helpers.

    def _inner_optimizers(self) -> list[torch.optim.Optimizer]:
        return list(self.engine.optim.optimizers.optimizers)

    def _init_optim_state(self) -> None:
        for opt in self._inner_optimizers():
            init_optim_state(opt)

    def _state_tensors(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return collect_state_tensors(self.engine.model_parts, self._inner_optimizers())

    def _meta(self) -> bytes:
        engine = self.engine
        _, cpu_tensors = self._state_tensors()
        state = {
            "engine": engine.state_dict(),
            "lr_schedulers": engine.optim.lr_schedulers.state_dict(),
            "dataloader": self.dataloader.state_dict(),
            "param_groups": [
                [
                    {k: v for k, v in g.items() if k != "params"}
                    for g in opt.param_groups
                ]
                for opt in self._inner_optimizers()
            ],
            "cpu_tensors": [t.clone() for t in cpu_tensors],
        }
        return pickle.dumps(state)

    def _load_meta(self, meta: bytes) -> None:
        state = pickle.loads(meta)
        engine = self.engine
        engine.load_state_dict(state["engine"])
        engine.optim.lr_schedulers.load_state_dict(state["lr_schedulers"])
        self.dataloader.load_state_dict(state["dataloader"])
        for opt, groups in zip(
            self._inner_optimizers(), state["param_groups"], strict=True
        ):
            for group, saved in zip(opt.param_groups, groups, strict=True):
                group.update(saved)
        _, cpu_tensors = self._state_tensors()
        for t, saved in zip(cpu_tensors, state["cpu_tensors"], strict=True):
            t.copy_(saved)

    # Failure handling.

    def _on_abort(self) -> None:
        self.snapshotter.abort()

    def _fsdp_roots(self) -> list[FSDPModule]:
        return [m for m in self.engine.model_parts if isinstance(m, FSDPModule)]

    def _handle_failure(self, err: BaseException) -> Assignment:
        _event(
            "failure_detected",
            step=self.engine.num_completed_steps,
            error=str(err)[:200],
        )
        self._dump_flight_recorder()
        self.snapshotter.abort()
        for root in self._fsdp_roots():
            try:
                root.reset_iter_state()
            except Exception as e:
                logger.warning(f"FSDP reset_iter_state failed: {e}")
        self.engine.optim.zero_grad()
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.exception("CUDA unusable after failure; restarting host")
            _exit(1)
        self.snapshotter.pause()
        try:
            assignment = self.membership.next_assignment()
        except TrainingFinishedError:
            self.finished = True
            _exit(0)
        except EvictedError as e:
            logger.error(str(e))
            _exit(1)
        _event("assigned", gen=assignment.gen, rank=dist.get_rank())
        return assignment

    def _dump_flight_recorder(self) -> None:
        """Write this rank's nccl2 flight recorder trace for each failure.

        The automatic dump on collective failure fires once per process, so
        later failures would otherwise leave no trace.
        """
        prefix = os.environ.get("TORCH_FR_DUMP_TEMP_FILE")
        if not prefix:
            return
        try:
            trace = torch._C._distributed_c10d._dump_fr_trace(
                True, False, False, "nccl2"
            )
            path = f"{prefix}{dist.get_rank()}.f{self.num_recoveries}"
            with open(path, "wb") as f:
                f.write(trace)
        except Exception:
            logger.exception("flight recorder dump failed")

    def _recover_from(self, err: Exception) -> None:
        """Recover, rejoining the quorum if recovery itself fails."""
        while True:
            self.num_recoveries += 1
            if self.num_recoveries > self.ft.max_recoveries:
                raise err
            try:
                self._recover(self._handle_failure(err))
                return
            except Exception as e:
                logger.exception("recovery failed; rejoining quorum")
                err = e

    def _recover(self, assignment: Assignment) -> None:
        start = time.perf_counter()
        ft = self.ft
        store = self.store
        rank, world = dist.get_rank(), dist.get_world_size()
        gen = assignment.gen
        timeout = ft.recovery_timeout_seconds
        snap = self.snapshotter
        idents = [m.ident for m in assignment.ranks]

        if not assignment.initial:
            # Survivors still use the train timeout, but the first step after
            # recovery may compile on new hosts. train() shortens it again.
            self._set_pg_timeouts(
                timedelta(seconds=self.config.comm.init_timeout_seconds)
            )
        if not assignment.initial and self.membership.host not in assignment.new_hosts:
            if assignment.new_hosts:
                self._replay_weight_init()
        snap.update_links(rank=rank, ident_of_rank=idents, gen=gen, timeout=timeout)
        links_s = time.perf_counter() - start

        if assignment.initial:
            if store.add(STARTED_KEY, 1) > world:
                logger.error(
                    "a quorum without trained hosts formed after training started; "
                    "refusing to restart from scratch"
                )
                _exit(1)
            snap.capture(self.engine.num_completed_steps, self._meta())
            snap.flush()
            restored = self.engine.num_completed_steps
        else:
            restored = self._restore_from_snapshots(assignment)

        restore_s = time.perf_counter() - start - links_s
        # Per-rank keys rather than a counter so store retries are idempotent.
        barrier = f"ftfsdp/restored/{gen}"
        store.set(f"{barrier}/{rank}", "1")
        keys = [f"{barrier}/{r}" for r in range(world)]
        deadline = time.monotonic() + timeout
        while not store.check(keys):
            if self.membership.recovery_pending():
                raise RuntimeError(f"a rank failed during recovery to gen {gen}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"recovery barrier {barrier} timed out")
            time.sleep(0.1)
        snap.resume()
        self._reset_metrics()
        self._shorten_timeout = True
        self.data_iterator = self.microbatch_generator(self.dataloader)
        _event(
            "recovered",
            gen=gen,
            rank=rank,
            step=restored,
            new=self.membership.host in assignment.new_hosts,
            links_s=links_s,
            restore_s=restore_s,
            total_s=time.perf_counter() - start,
        )

    def _set_pg_timeouts(self, timeout: timedelta) -> None:
        """Like dist_utils.set_pg_timeouts but without the barrier, which
        would be the first collective on a freshly reconfigured group."""
        meshes = self.engine.parallelism_context.get_all_one_dimensional_meshes()
        for group in [m.get_group() for m in meshes.values()] + [None]:
            dist.set_timeout(timeout, group)

    def _reset_metrics(self) -> None:
        """Restart throughput accounting from the restored step; steps after
        it were already logged and are replayed."""
        mp = self.metrics_processor
        mp.step_last_log = None
        mp.ntokens_since_last_log = 0
        mp.data_loading_times.clear()
        mp.time_last_log = time.perf_counter()

    def _replay_weight_init(self) -> None:
        """Match the collectives new hosts issue while initializing weights.

        DTensor weight init communicates over the default group, so survivors
        must run it too. The result is overwritten by the snapshot restore.
        """
        start = time.perf_counter()
        with self.engine.parallelism_context.activate_spmd(), torch.no_grad():
            for part in self.engine.model_parts:
                part.init_weights(buffer_device=None)
        _event("weight_init_replayed", seconds=time.perf_counter() - start)

    def _restore_from_snapshots(self, assignment: Assignment) -> int:
        store = self.store
        snap = self.snapshotter
        rank, world = dist.get_rank(), dist.get_world_size()
        g = self.ft.procs_per_host
        gen = assignment.gen
        timeout = timedelta(seconds=self.ft.recovery_timeout_seconds)

        replicas = snap.replica_steps()
        store.set(
            f"ftfsdp/avail/{gen}/{rank}",
            json.dumps(
                {
                    "local": snap.committed_steps(),
                    "replica": {str(s): v for s, v in replicas.items()},
                }
            ),
        )
        keys = [f"ftfsdp/avail/{gen}/{r}" for r in range(world)]
        store.wait(keys, timeout)
        avail = [json.loads(v) for v in store.multi_get(keys)]

        is_new = [m.host in assignment.new_hosts for m in assignment.ranks]
        common: set[int] | None = None
        for r in range(world):
            if is_new[r]:
                steps = {int(s) for s in avail[(r + g) % world]["replica"]}
            else:
                steps = set(avail[r]["local"])
            common = steps if common is None else common & steps
        if not common:
            raise RuntimeError(
                "no snapshot step is held by every rank; adjacent hosts likely "
                f"failed together: {avail}"
            )
        step = max(common)
        if is_new[rank]:
            index, meta_len = avail[(rank + g) % world]["replica"][str(step)]
            fetch_start = time.perf_counter()
            snap.fetch_replica(step, index, meta_len)
            _event(
                "replica_fetched", step=step, seconds=time.perf_counter() - fetch_start
            )
        lost = self.engine.num_completed_steps - step if not is_new[rank] else None
        self._load_meta(snap.restore(step))
        snap.replicate_now(step)
        logger.info(f"restored step {step} (lost {lost} steps)")
        return step

    # Training loop.

    def _reset_lm_head_reshard(self) -> None:
        # ChunkedLossWrapper ends each step with
        # lm_head.set_reshard_after_forward(True), overriding the [norm,
        # lm_head] group's configured False, and a failure inside its chunk
        # loop leaves False. Either way a process's collective sequence then
        # depends on its history: a new host's first step issues one fewer
        # all-gather than a survivor's and the step hangs. Restore the value
        # parallelize_fn used (torchtitan/distributed/fsdp.py) so every step
        # matches.
        loss_fn = self.engine.loss_fn
        if not isinstance(loss_fn, ChunkedLossWrapper):
            return
        lm_head = loss_fn.lm_head
        if isinstance(lm_head, FSDPModule):
            lm_head.set_reshard_after_forward(
                self.config.parallelism.fsdp_reshard_after_forward == "always"
            )

    @record
    def train(self) -> None:
        config = self.config
        engine = self.engine
        ft = self.ft
        logger.info(f"Training starts at step {engine.num_completed_steps + 1}")
        engine.start_profiler()
        try:
            while self.should_continue_training():
                current_step = engine.num_completed_steps + 1
                sl.set_step(current_step)
                try:
                    step_start = time.perf_counter()
                    self._reset_lm_head_reshard()
                    self.train_step(self.data_iterator)
                    if self._shorten_timeout:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallelism_context=engine.parallelism_context,
                        )
                        self._shorten_timeout = False
                    if engine.num_completed_steps % ft.snapshot_interval == 0:
                        self.snapshotter.capture(
                            engine.num_completed_steps, self._meta()
                        )
                    if engine.num_completed_steps % config.metrics.log_freq == 0:
                        _event(
                            "step",
                            step=engine.num_completed_steps,
                            seconds=time.perf_counter() - step_start,
                            snapshot_copy_s=self.snapshotter.last_copy_s,
                            snapshot_replicate_s=self.snapshotter.last_replicate_s,
                        )
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break
                except Exception as e:
                    logger.exception(f"step {current_step} failed")
                    self._recover_from(e)
                    continue
                engine.step_profiler()
        finally:
            engine.close_profiler()

        self.snapshotter.flush()
        self.finished = True
        if dist.get_rank() == 0:
            self.store.set(DONE_KEY, "1")
        _event("training_done", step=engine.num_completed_steps)
        logger.info("Training completed")

    def close(self) -> None:
        super().close()
        if hasattr(self, "snapshotter"):
            self.snapshotter.close()
        if hasattr(self, "membership"):
            self.membership.close()
        _exit(0 if self.finished else 1)

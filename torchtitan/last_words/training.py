# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Application transactions and logical data cursors around a TorchFT engine."""

from __future__ import annotations

import copy
import threading
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.api import CheckpointException
from torchft.last_words import LastWordsManager, TorchFTQuorumStoppedError

from torchtitan.components.checkpointer import DATALOADER
from .control import config_signature, LastWordsCoordinator, LastWordsExitError
from .rng import RNG_STATE, RNGCheckpointState


class CommitRejectedError(RuntimeError):
    """Internal control flow: no optimizer or scheduler update was executed."""

    def __init__(self, grad_norm):
        super().__init__("TorchFT rejected the optimizer attempt")
        self.grad_norm = grad_norm


class LastWordsTrainingCoordinator:
    """Own training completeness; leave writer authorization to TorchFT's service."""

    def __init__(self, trainer):
        from .checkpoint import LastWordsCheckpointManager

        self.trainer, self.engine = trainer, trainer.engine
        engine = self.engine
        self.manager = engine.ft_manager.manager
        self.checkpointer = engine.checkpointer
        if not isinstance(self.manager, LastWordsManager):
            raise TypeError("Last words requires TorchFT LastWordsManager")
        if not isinstance(self.checkpointer, LastWordsCheckpointManager):
            raise TypeError("Last words requires LastWordsCheckpointManager")
        self.elastic = trainer.config.fault_tolerance.enable_elastic
        self.num_updates = 0
        self.completed = None
        self.completed_train_state = None
        self.updating = False
        self.cursor_restore_pending = False
        self.cursors_initialized = False
        self.iterator = None
        self.rng_state = RNGCheckpointState(engine.device, dist.get_rank())
        self.rng_state.capture()
        self.completed_rng_state = None
        self.checkpointer.rng_state = self.rng_state
        self.checkpointer.ft_states[RNG_STATE] = self.rng_state
        self.control = None
        self.control = LastWordsCoordinator(
            trainer.config.fault_tolerance.last_words,
            folder=self.checkpointer.folder,
            replica_id=engine.ft_manager.replica_id,
            num_replicas=engine.ft_manager.group_size,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            elastic_enabled=self.elastic,
            on_stop=None if self.elastic else self._request_device_stop,
        )
        self.checkpointer.last_words = self.control
        self.checkpointer.last_words_signature = config_signature(trainer.config)
        self.checkpointer.last_words_on_stop = self.finish
        self.checkpointer._last_words_control_pg = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=self.control.config.timeout_seconds),
        )
        if self.elastic:
            self.checkpointer.enable_elastic_state()
        self.manager.set_last_words_hooks(
            request=self.control.request,
            stop_requested=lambda: self.control.requested,
            terminal_stop_requested=lambda: self.control.terminal_stop_requested,
            after_heal=self._after_heal,
            training_boundary=self._quorum_boundary,
            elastic_enabled=self.elastic,
        )
        engine.optim.last_words = self
        engine.bind_last_words_device_hooks(self)

    def _request_device_stop(self):
        control = self.control
        if control is None:
            return
        if (
            threading.current_thread() is control._thread
            and self.num_updates == 0
            and not self.updating
            and not self.manager.commit_authorized
        ):
            control.publish("ineligible")
            return
        self.engine.request_last_words_device_stop()

    @staticmethod
    def marker(state):
        return {
            key: state[key]
            for key in ("step", "ntokens_seen", "last_words_num_updates")
        }

    def restored_train_state(self, state):
        self.num_updates = state.get("last_words_num_updates", state["step"])
        state = dict(state, last_words_num_updates=self.num_updates)
        self.completed_train_state = copy.deepcopy(state)
        self.completed = self.marker(state)

    def _make_slot_cursor(self, replica):
        trainer = self.trainer
        context = self.engine.parallelism_context
        mesh = context.get_optional_mesh("dp")
        degree, rank = (
            (mesh.size(), mesh.get_local_rank()) if mesh is not None else (1, 0)
        )
        config = trainer.config
        try:
            return {
                DATALOADER: config.dataloader.build(
                    dp_world_size=degree * self.engine.ft_manager.group_size,
                    dp_rank=degree * replica + rank,
                    tokenizer=trainer.tokenizer,
                    max_context_length=config.training.max_context_length,
                    num_tokens_per_microbatch=config.training.num_tokens_per_microbatch_per_dp_rank,
                ),
                RNG_STATE: self.rng_state.fork(),
            }
        finally:
            # Constructing a dormant slot must not advance the active stream.
            self.rng_state.restore()

    def _after_heal(self):
        # Shard quorum callbacks must not independently enter replica-local DCP.
        self.cursor_restore_pending = True

    def _restore_cursor(self):
        self.checkpointer.restore_elastic_cursor(self.completed)
        self.completed_rng_state = self.rng_state.state_dict()
        self.iterator = self.trainer.microbatch_generator(self.trainer.dataloader)
        self.checkpointer.last_words_cursor_current = True

    def _quorum_boundary(self):
        if not self.elastic:
            return
        group = self.checkpointer._last_words_control_pg
        pending = torch.tensor(int(self.cursor_restore_pending), dtype=torch.int32)
        dist.all_reduce(pending, op=dist.ReduceOp.MAX, group=group)
        if pending.item():
            states = [None] * self.control.world_size
            dist.all_gather_object(states, self.completed, group=group)
            if any(state != self.completed for state in states):
                self.control.publish("ineligible")
                raise LastWordsExitError("Healed shards disagree on complete progress")
            self._restore_cursor()
            self.cursor_restore_pending = False
        snapshot = self.manager.quorum_snapshot()
        if any(member >= self.control.num_replicas for member in snapshot.members):
            raise LastWordsExitError("Native quorum exceeds logical slot capacity")
        if self.control.training_boundary(
            quorum_id=snapshot.quorum_id,
            members=list(snapshot.members),
            state=self.completed,
            torchft_state=dict(
                step=snapshot.step, batches_committed=snapshot.batches_committed
            ),
        ):
            self.finish()

    def _batches(self):
        while True:
            yield next(self.iterator)

    def before_optimizer_update(self):
        self.updating = True
        self.control.publish("updating")

    def update_completed(self):
        self.engine.synchronize_last_words_device()
        self.rng_state.capture()
        self.completed_rng_state = self.rng_state.state_dict()
        self.num_updates += 1
        self.completed_train_state = copy.deepcopy(self.engine.state_dict())
        self.completed = self.marker(self.completed_train_state)
        self.checkpointer.last_words_state = dict(self.completed)
        self.checkpointer.last_words_cursor_current = True
        self.updating = False
        self.control.publish("completed", state=self.completed)

    def finish(self):
        if self.completed is None or self.num_updates == 0:
            self.control.publish("ineligible")
            raise LastWordsExitError(
                "No complete optimizer update is available to save"
            )
        self.engine.load_state_dict(self.completed_train_state)
        target = self.checkpointer.save_last_words(self.completed)
        if self.elastic and self.control.can_resume_baseline:
            self.control.resume_baseline(self.completed)
            return target
        raise LastWordsExitError(f"Last words saved at {target}; restart the whole job")

    def run_step(self, step, data_iterator):
        if self.completed is None:
            self.restored_train_state(self.engine.state_dict())
        if self.iterator is None:
            self.iterator = data_iterator
        if self.completed_rng_state is None:
            self.rng_state.capture()
            self.completed_rng_state = self.rng_state.state_dict()
        if self.elastic and not self.cursors_initialized:
            self.checkpointer.initialize_elastic_cursors(
                self.completed,
                make_cursor=self._make_slot_cursor,
                initial_replicas=self.control.initial_replicas,
            )
            self.cursors_initialized = True
            self.checkpointer.last_words_cursor_current = True
        began, failures = time.monotonic(), 0
        while True:
            if self.checkpointer.last_words_stop_requested():
                self.finish()
            self.updating = False
            before = self.num_updates
            error, result = None, None
            try:
                # Establish and heal native state before consuming any data.
                self.engine.begin_last_words_attempt()
                self.manager.begin_step()
                self.engine.optim.optimizers.last_words_prepared = True
                self.control.publish("forward_backward", state=self.completed)
                self.checkpointer.last_words_cursor_current = False
                result = step(self._batches())
            except (Exception, CheckpointException) as caught:
                error = caught
            if (
                self.updating
                or self.manager.restoration_incomplete
                or self.manager.commit_unknown
            ):
                self.control.request(f"Unsafe update, restoration, or commit: {error}")
                self.control.publish("ineligible")
                raise LastWordsExitError(
                    "Incomplete or unknown transaction cannot be saved"
                ) from error
            if self.manager.commit_authorized and self.num_updates > before:
                if error is not None:
                    self.control.request_recovery(f"Post-update error: {error}")
                return result
            if isinstance(error, LastWordsExitError) and not isinstance(
                error, TorchFTQuorumStoppedError
            ):
                raise error
            terminal_quorum = (
                isinstance(error, TorchFTQuorumStoppedError)
                and self.control.terminal_stop_requested
            )
            error = error or RuntimeError("TorchFT rejected an uncommitted attempt")
            if self.elastic:
                self.control.request_recovery(str(error))
            else:
                self.control.request(str(error))
            try:
                self.engine.recover_last_words_attempt(error)
                if not terminal_quorum:
                    self.manager.discard_interrupted_step(error)
                if self.elastic:
                    self._restore_cursor()
                self.engine.load_state_dict(self.completed_train_state)
                self.rng_state.load_state_dict(self.completed_rng_state)
                self.rng_state.restore()
                self.engine.optim.optimizers.last_words_prepared = False
            except (Exception, CheckpointException) as cleanup_error:
                self.control.request(f"Recovery failed: {cleanup_error}")
                self.control.publish("ineligible")
                raise LastWordsExitError(
                    "A readable complete state could not be recovered"
                ) from cleanup_error
            if not self.elastic or terminal_quorum:
                return self.finish()
            failures += 1
            config = self.trainer.config.fault_tolerance
            if (
                failures >= config.max_consecutive_recoveries
                or time.monotonic() - began >= config.elastic_recovery_timeout_seconds
            ):
                self.control.request("Elastic recovery budget exhausted")
                return self.finish()

    def close(self):
        self.control.close()

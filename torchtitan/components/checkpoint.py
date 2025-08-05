# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import functools
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import Future
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Checkpoint as CheckpointConfig, TORCH_DTYPE_MAP
from torchtitan.protocols import StateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


# For now, we will manually pop the freqs_cis buffer, as we made this permanent
# temporarily and we don't want to include it in the exported state_dict.
# Context: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model.py#L404
excluded_parameters_for_model_only = {"freqs_cis"}


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        # Exclude parameters that should not be saved
        for excluded_key in excluded_parameters_for_model_only:
            state_dict.pop(excluded_key, None)
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager:
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is gauranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Note: TorchFT checkpointing flow

    There are two types of checkpoints: when TorchFT is enabled: 1) the full perisistent
    checkpoint, 2) the per-replica checkpoint.

    The full perisistent checkpoint is saved by the replica with
    ``ft_manager.participating_rank() == 0``. It contains everything including the model,
    optimizer, lr_scheduler, dataloader, and train_state. Right now the full perisistent
    checkpoint is loaded by all replicas. However, we can optimize it to only load if
    there are no other alive replicas.

    The per-replica checkpoint contains only the dataloader and is saved/loaded by all
    replicas to/from the its own folder. The folder name is prefixed with the ft_replica_id.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        checkpoint_config (Checkpoint): The config used to configure the checkpointing.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with checkpoint_config.folder
        sd_adapter (Optional[type[StateDictAdapter]]): The adapter used to convert model state
            dicts between native format and other formats.
        ft_manager (Optional[ft.Manager]): The FTManager from TorchFT.

    """

    def __init__(
        self,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        checkpoint_config: CheckpointConfig,
        sd_adapter: StateDictAdapter | None,
        base_folder: str = "",
        ft_manager: FTManager | None = None,
    ) -> None:
        self.enable_checkpoint = checkpoint_config.enable_checkpoint

        self.ft_manager = (
            ft_manager.manager if ft_manager and ft_manager.enabled else None
        )
        if self.ft_manager:
            optimizers.init_cache_state_dict()

            def state_dict():
                ret = {}
                for k, v in self.states.items():
                    if k in {
                        MODEL,
                        OPTIMIZER,
                        LR_SCHEDULER,
                        TRAIN_STATE,
                    }:
                        ret[k] = v.state_dict()
                return ret

            def load_state_dict(state_dict):
                assert state_dict is not None
                for k, v in state_dict.items():
                    self.states[k].load_state_dict(v)

            self.ft_manager.set_state_dict_fns(load_state_dict, state_dict)
            self.ft_replica_id = ft_manager.replica_id

        async_mode = checkpoint_config.async_mode.lower()
        self.enable_staging = (
            self.enable_checkpoint and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ) or self.ft_manager

        if not self.enable_checkpoint and self.ft_manager is None:
            return

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )
        self.ft_states = {DATALOADER: dataloader}

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None

        self.folder = os.path.join(base_folder, checkpoint_config.folder)

        # Checkpoint policy related fields.
        self.initial_load_model_only = checkpoint_config.initial_load_model_only
        self.initial_load_in_hf = checkpoint_config.initial_load_in_hf
        self.initial_load_path = checkpoint_config.initial_load_path
        self.last_save_model_only = checkpoint_config.last_save_model_only
        self.last_save_in_hf = checkpoint_config.last_save_in_hf
        if self.last_save_in_hf:
            assert (
                sd_adapter is not None
            ), "job_config.checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
        self.sd_adapter = sd_adapter
        self.export_dtype = TORCH_DTYPE_MAP[checkpoint_config.export_dtype]
        self.exclude_from_loading = checkpoint_config.exclude_from_loading
        self.interval = checkpoint_config.interval
        self.enable_first_step_checkpoint = (
            checkpoint_config.enable_first_step_checkpoint
        )

        # Async checkpoint related fields.
        async_mode = checkpoint_config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            or self.ft_manager
        ):
            self.pg = dist.new_group(backend="gloo")

        self.keep_latest_k = checkpoint_config.keep_latest_k
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(
                f"Unkown checkpoint async_mode {checkpoint_config.async_mode}"
            )

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable_checkpoint") and self.enable_checkpoint:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | None:
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret: Future | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        checkpoint_save_id: str | None = None
        if to_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            state_dict = self.sd_adapter.to_hf(state_dict)

            fqn_to_index_mapping = {}
            num_fqns_per_file = 30
            # the use of 30 is just a heuristic for now.
            # Once these fqns map to HF ones, we can use the fqn mapping
            # from the model.safetensors.index.json file
            for i, key in enumerate(state_dict.keys()):
                group_num = (i // num_fqns_per_file) + 1
                fqn_to_index_mapping[key] = group_num

            storage_writer = HuggingFaceStorageWriter(
                path=checkpoint_id,
                save_distributed=True,
                fqn_to_index_mapping=fqn_to_index_mapping,
                enable_consolidation=True,
                thread_count_consolidation=5,
            )

        else:
            checkpoint_save_id = checkpoint_id

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
    ) -> None:
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.
        """

        if from_hf:
            assert (
                self.sd_adapter is not None
            ), "trying to load checkpoint in HF safetensors format, but sd_adapter is not provided."
            hf_state_dict = self.sd_adapter.to_hf(state_dict)

            dcp.load(
                hf_state_dict,
                storage_reader=HuggingFaceStorageReader(path=checkpoint_id),
            )

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            self.states[MODEL].load_state_dict(state_dict)
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``last_step`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.

        Returns:
            None
        """

        if self.ft_manager:
            self._ft_save(curr_step)

        if not self._should_save(curr_step, last_step):
            return

        begin = time.monotonic()
        if not self.ft_manager or self.ft_manager.participating_rank() == 0:
            logger.info("Saving the checkpoint (or staging if async is enabled).")
            checkpoint_id = self._create_checkpoint_id(curr_step)
            self._async_wait()
            # This GC is called for async checkpoint as it is useless to do
            # GC right after async_save -- the CPU memory is not able to be
            # freed until _async_wait()
            if last_step:
                self._save_last_step(curr_step)
                return

            states = self._flattened_model_states_sd()
            if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.stager is None:
                    self.stager = DefaultStager(StagingOptions(True, True, True, True))
                result = self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=self.async_mode,
                )
                self.save_future = result.upload_completion
                self.staging_future = result.staging_completion
            elif self.async_mode == AsyncMode.ASYNC:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                self.save_future = self.dcp_save(
                    states, checkpoint_id=checkpoint_id, async_mode=self.async_mode
                )
                GarbageCollection.collect("GC collection invoked by checkpointer.")
            else:
                self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=AsyncMode.DISABLED,
                    enable_garbage_collection=True,
                )
            self._purge_stale_checkpoints()

            logger.info(
                "Finished saving the checkpoint (or staging if async is enabled)"
                f"in {time.monotonic() - begin:.2f} seconds."
            )
        elif self.ft_manager:
            logger.info(
                "Replica %d doesn't save checkpoint.",
                self.ft_manager.participating_rank(),
            )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """

        if self.ft_manager:
            self._ft_load()

        if not self.enable_checkpoint:
            return False

        model_only = False
        from_hf = False
        if not os.path.exists(self.folder):
            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "checkpoint.initial_load_path is specified but the path is not valid."
                    )
                model_only = self.initial_load_model_only
                from_hf = self.initial_load_in_hf
                if from_hf:
                    assert (
                        model_only
                    ), "Only model can be loaded when loading from HF's safetensors checkpoint."
            else:
                return False
        else:
            if self.initial_load_path:
                logger.warning(
                    "checkpoint.initial_load_path is provided but the checkpoint.folder exists. "
                    f"Checkpointer will use the checkpoints from the checkpoint.folder {self.folder}."
                )
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} but checkpoint {checkpoint_id} is not found."
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(model_only)
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            self.staging_future.result()

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            dcp_metadata_probe = os.path.join(folder, filename, ".metadata")
            safetensors_metadata_probe = os.path.join(
                folder, filename, "model.safetensors.index.json"
            )
            if match and os.path.isfile(dcp_metadata_probe):
                step_counts.append(int(match.group(1)))
            elif match and os.path.isfile(safetensors_metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _ft_folder(self) -> str:
        return os.path.join(self.folder, f"ft-replicat-{self.ft_replica_id}")

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _ft_save(self, step: int) -> None:
        begin = time.monotonic()
        self._async_wait()
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.save_future = self.dcp_save(
            self.ft_states, checkpoint_id=checkpoint_id, async_mode=AsyncMode.ASYNC
        )
        logger.info(f"Staging ft checkpoint took {time.monotonic() - begin} secs.")

    def _ft_load(self) -> None:
        step = self._find_load_step(folder=self._ft_folder())
        if step == -1:
            return

        begin = time.monotonic()
        logger.info(f"Loading the FT checkpoint at step {step}.")
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.dcp_load(
            self.ft_states,
            checkpoint_id=checkpoint_id,
            # FT checkpoints are always DCP because FT checkpoint currently only save/load dataloader.
            from_hf=False,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the ft checkpoint in {time.monotonic() - begin:.2f} seconds."
        )

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.

        Note that other states, such as optimizer states, are not flattened.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.

        This API is used to determine which states to load based on the
        configurations.

        Args:
            model_only (bool): Whether to load the model only.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        states_to_load = self._flattened_model_states_sd(states_to_load)

        if self.ft_manager:
            states_to_load.pop(DATALOADER)

        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So this
        # won't affect preemption and training resume. We also only allow dtype
        # conversion when we are checkpointing model only and the current dtype
        # is not the same as the export dtype at the end of the training.

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
        elif self.async_mode == AsyncMode.ASYNC or self.ft_manager is not None:
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled "
                "and fault tolerance is not active."
            )

    def _purge_stale_checkpoints(self):
        if (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
            and (not self.ft_manager or self.ft_manager.participating_rank() == 0)
        ):
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)

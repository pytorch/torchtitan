# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import math
import os
import pprint
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torchtitan.components.checkpoint import MODEL
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


def extract_layer_number(s):
    import re

    match = re.search(r"layers\.(\d+)", s)
    if match:
        return int(match.group(1))
    else:
        return None


def convert_to_titan_fqns(fqn: str) -> list[str]:
    # From the stored checkpoint keys to TorchTitan keys.
    if "language_model." not in fqn:
        # TODO: Not support video model yet
        return [fqn]

    layer = extract_layer_number(fqn)

    if layer is None:
        if "embed_tokens.weight" in fqn:
            return ["tok_embeddings.weight"]
        elif "norm.weight" in fqn:
            return ["norm.weight"]
        elif "lm_head.weight" in fqn:
            return ["output.weight"]
        else:
            raise ValueError(f"Unknown fqn {fqn}")

    if "feed_forward.experts.down_proj" in fqn:
        return [f"layers.{layer}.moe.experts.w2"]
    elif "feed_forward.experts.gate_up_proj" in fqn:
        return [f"layers.{layer}.moe.experts.w1", f"layers.{layer}.moe.experts.w3"]
    elif "feed_forward.router.weight" in fqn:
        return [f"layers.{layer}.moe.router.gate.weight"]
    elif "feed_forward.shared_expert.down_proj.weight" in fqn:
        return [f"layers.{layer}.moe.shared_expert.w2"]
    elif "feed_forward.shared_expert.gate_proj.weight" in fqn:
        return [f"layers.{layer}.moe.shared_expert.w3"]
    elif "feed_forward.shared_expert.up_proj.weight" in fqn:
        return [f"layers.{layer}.moe.shared_expert.w1"]
    elif "input_layernorm.weight" in fqn:
        return [f"layers.{layer}.ffn_norm.weight"]
    elif "self_attn.k_proj" in fqn:
        return [f"layers.{layer}.attention.wk.weight"]
    elif "self_attn.o_proj" in fqn:
        return [f"layers.{layer}.attention.wo.weight"]
    elif "self_attn.q_proj" in fqn:
        return [f"layers.{layer}.attention.wq.weight"]
    elif "self_attn.v_proj" in fqn:
        return [f"layers.{layer}.attention.wv.weight"]
    elif "post_attention_layernorm.weight" in fqn:
        return [f"layers.{layer}.attention_norm.weight"]
    else:
        raise ValueError(f"Unknown fqn {fqn}")


def convert_to_hf_shape(fqn: str, titan_fqns: list[str], dtensor: DTensor) -> list[str]:
    if "feed_forward.experts.gate_up_proj" in fqn:
        assert len(titan_fqns) == 2
        shape = dtensor.shape
        return torch.Size(list(shape[:-1]) + [shape[-1] * 2])
    elif "shared_expert" in fqn:
        s = dtensor.shape
        # TODO: this is not right but I have to do this to load the checkpoint.
        return torch.Size((s[2], s[1]))
    return dtensor.shape


def convert_to_titan_tensors(fqn: str, full_tensor: torch.Tensor) -> torch.Tensor:
    if "feed_forward.experts.gate_up_proj" in fqn:
        full_tensors = full_tensor.chunk(2, dim=-1)
    elif "shared_expert" in fqn:
        # TODO: this is not right but I have to do this to load the checkpoint.
        full_tensor = full_tensor.transpose(1, 0)
        full_tensors = [full_tensor.unsqueeze(0)]
    else:
        full_tensors = [full_tensor]
    return full_tensors


@dataclass
class _Assignment:
    loader_id: int
    filename: str
    fqns: list[str]
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]


@dataclass
class _AssignmentRound:
    loader_assignments: dict[int, _Assignment]  # List of assignments for each loader


@dataclass
class TensorMetadata:
    fqn: str
    shape: torch.Size
    dtype: torch.dtype


class CheckpointConverter:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        path: str,
        token: Optional[str] = None,
        loader_every_n_ranks: int = 8,
    ) -> None:
        self.path = path
        self.token = token
        self.pg = process_group
        self.my_rank = dist.get_rank(self.pg)

        self.loader_every_n_ranks = loader_every_n_ranks
        self.loader_id = self.my_rank // loader_every_n_ranks
        self.should_load = self.my_rank % loader_every_n_ranks == 0
        self.total_loader = dist.get_world_size(self.pg) // loader_every_n_ranks

        self.titan_fqn_to_stored_fqn: dict[str, str] = {}
        self.stored_fqn_to_titan_fqn: dict[str, list[str]] = {}
        self.total_send_bytes = 0
        self.total_recv_bytes = 0

    def convert(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        begin = time.time()
        self._load_metadata()
        self._create_fqn_mappings(state_dict)
        rounds = self._get_load_assignments(state_dict)

        logger.info(f"Got {len(rounds)} rounds of assignments.")
        for idx, assignments in enumerate(rounds):
            loader_assignments = assignments.loader_assignments
            loaded_state_dict = None
            # Let each loader to load its own data and move to its GPU.
            logger.info(f"Loading round {idx}")
            for i in range(self.total_loader):
                # This loader doesn't have any loading assignment for this round.
                if i not in loader_assignments:
                    continue
                # This rank is not the loader
                if i != self.loader_id or not self.should_load:
                    continue
                loaded_state_dict = self._load_round(loader_assignments[i])

            torch.cuda.synchronize()
            logger.info(f"Loading round {idx} finished")
            for i in range(self.total_loader):
                if i not in loader_assignments:
                    continue

                logger.info(f"Resharding round {idx} loader {i} data. ")
                if i == self.loader_id and self.should_load:
                    # This rank is the loader. It needs to send the loaded data to
                    # the other ranks.
                    assert loaded_state_dict is not None
                    results = self._reshard_send(
                        loader_assignments[i], loaded_state_dict
                    )
                else:
                    results = self._reshard_receive(loader_assignments[i], state_dict)
                torch.cuda.synchronize()

                logger.info(f"Communication round {idx} loader {i} is done.")
                self._reshard(results, state_dict)
                logger.info(f"Resharding round {idx} loader {i} is done.")
                self._reshard(results, state_dict)
                torch.cuda.synchronize()

        dist.barrier()
        torch.cuda.synchronize()
        logger.info(f"Checkpoint conversion took {time.time() - begin:.2f} seconds.")
        logger.info(f"Total send bytes: {self.total_send_bytes / 1e9:.2f} GB")
        logger.info(f"Total recv bytes: {self.total_recv_bytes / 1e9:.2f} GB")
        return state_dict

    def _load_metadata(self) -> None:
        metadata_path = os.path.join(self.path, "model.safetensors.index.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)["weight_map"]

    def _create_fqn_mappings(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not self.metadata:
            return

        # Create the mapping from the stored checkpoint keys to TorchTitan keys.
        for fqn in list(self.metadata.keys()):
            titan_fqns = convert_to_titan_fqns(fqn)
            # We don't know how to process _extra_state
            if "_extra_state" in fqn:
                self.metadata.pop(fqn)
                continue

            if titan_fqns[0] not in state_dict:
                for titan_fqn in titan_fqns:
                    assert titan_fqn not in state_dict
                self.metadata.pop(fqn)
                continue

            self.stored_fqn_to_titan_fqn[fqn] = titan_fqns
            for titan_fqn in titan_fqns:
                self.titan_fqn_to_stored_fqn[titan_fqn] = fqn

        torchtitan_extra = sorted(
            list(set(state_dict.keys()) - set(self.titan_fqn_to_stored_fqn.keys()))
        )
        converted_extra = sorted(
            list(set(self.titan_fqn_to_stored_fqn.keys()) - set(state_dict.keys()))
        )
        assert set(state_dict.keys()) == set(self.titan_fqn_to_stored_fqn.keys()), (
            f"{pprint.pformat(torchtitan_extra)}",
            f"{pprint.pformat(converted_extra)}",
        )

    def _get_load_assignments(
        self, state_dict: dict[str, Any]
    ) -> list[_AssignmentRound]:
        if self.my_rank == 0:
            filename_to_metas = defaultdict(list)
            for fqn, filename in self.metadata.items():
                titan_fqns = self.stored_fqn_to_titan_fqn[fqn]
                shape = convert_to_hf_shape(fqn, titan_fqns, state_dict[titan_fqns[0]])
                meta = TensorMetadata(
                    fqn=fqn,
                    shape=shape,
                    # TODO: don't hardcode this
                    dtype=torch.bfloat16,
                )
                filename_to_metas[filename].append(meta)

            loader_filename_to_metas = [{} for _ in range(self.total_loader)]
            for idx, (filename, metas) in enumerate(filename_to_metas.items()):
                loader_id = idx % self.total_loader
                loader_filename_to_metas[loader_id][filename] = metas

            rounds = []
            while any(len(remain) > 0 for remain in loader_filename_to_metas):
                round_assignment = _AssignmentRound(loader_assignments={})
                for loader_id in range(self.total_loader):
                    if not loader_filename_to_metas[loader_id]:
                        continue

                    filename, metas = loader_filename_to_metas[loader_id].popitem()
                    round_assignment.loader_assignments[loader_id] = _Assignment(
                        filename=filename,
                        fqns=[meta.fqn for meta in metas],
                        shapes=[meta.shape for meta in metas],
                        dtypes=[meta.dtype for meta in metas],
                        loader_id=loader_id,
                    )

                rounds.append(round_assignment)

            object_list: list[Any] = [
                rounds,
                self.titan_fqn_to_stored_fqn,
                self.stored_fqn_to_titan_fqn,
            ]
        else:
            object_list = [None, None, None]

        dist.broadcast_object_list(object_list, src=0, group=self.pg)
        rounds = object_list[0]
        self.titan_fqn_to_stored_fqn = object_list[1]
        self.stored_fqn_to_titan_fqn = object_list[2]
        return rounds

    def _load_round(self, assignment: _Assignment) -> dict[str, Any]:
        from safetensors.torch import load_file as hf_load_file

        path = os.path.join(self.path, assignment.filename)
        state_dict = hf_load_file(path)
        return {
            k: v.to(device="cuda")
            for k, v in state_dict.items()
            if k in assignment.fqns
        }

    def _reshard_send(
        self,
        assignment: _Assignment,
        loaded_state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        flatten_tensors = [t.flatten() for t in loaded_state_dict.values()]
        flatten_tensor = torch.concat(flatten_tensors)
        assert self.loader_id == assignment.loader_id
        rank = self.loader_id * self.loader_every_n_ranks
        assert rank == self.my_rank
        logger.info(
            f"Sending {assignment.filename} from {rank} {self.loader_id} "
            f"{flatten_tensor.shape=} {flatten_tensor.dtype=} {loaded_state_dict.keys()=}."
        )
        logger.info(f"Sending {assignment}")
        dist.broadcast(flatten_tensor, src=rank, group=self.pg)
        self.total_send_bytes += flatten_tensor.numel() * flatten_tensor.element_size()
        return loaded_state_dict

    def _reshard_receive(
        self, assignment: _Assignment, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:

        flatten_tensor = torch.empty(
            sum(math.prod(s) for s, d in zip(assignment.shapes, assignment.dtypes)),
            dtype=assignment.dtypes[0],
            device="cuda",
        )
        rank = assignment.loader_id * self.loader_every_n_ranks
        logger.info(
            f"Receiving {assignment.filename} from {rank} "
            f"{flatten_tensor.shape=} {flatten_tensor.dtype=}"
        )
        logger.info(f"Receiving {assignment}")
        dist.broadcast(flatten_tensor, src=rank, group=self.pg)
        self.total_recv_bytes += flatten_tensor.numel() * flatten_tensor.element_size()

        ret: dict[str, torch.Tensor] = {}
        loc = 0
        for fqn, shape, dtype in zip(
            assignment.fqns, assignment.shapes, assignment.dtypes
        ):
            n_ele = math.prod(shape)
            ret[fqn] = flatten_tensor[loc : loc + n_ele].view(shape)
            loc += n_ele
        return ret

    def _reshard(
        self,
        result: dict[str, torch.Tensor],
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        def _inplace_copy(fqn: str, full_tensors: list[torch.Tensor]):
            titan_fqns = self.stored_fqn_to_titan_fqn[fqn]
            assert len(titan_fqns) == len(full_tensors)
            for titan_fqn, full_tensor in zip(titan_fqns, full_tensors):
                dtensor = state_dict[titan_fqn]
                assert isinstance(dtensor, DTensor)
                assert dtensor.shape == full_tensor.shape, (
                    (fqn, titan_fqn),
                    dtensor.shape,
                    full_tensor.shape,
                )
                shape, offset = compute_local_shape_and_global_offset(
                    full_tensor.shape, dtensor.device_mesh, dtensor.placements
                )
                slices = [
                    slice(cur_offset, cur_offset + cur_shape)
                    for cur_shape, cur_offset in zip(shape, offset)
                ]
                logger.debug(
                    f"Copying {titan_fqn} with {slices=} {dtensor._local_tensor.shape=} "
                    f"{shape=} {offset=} {self.my_rank=} {dtensor.shape=} {full_tensor.shape=} "
                    f"{dtensor.placements=} {dtensor.device_mesh=} "
                )
                dtensor.to_local().copy_(full_tensor[slices].to(dtensor.dtype))

        for fqn, full_tensor in result.items():
            full_tensors = convert_to_titan_tensors(fqn, full_tensor)
            _inplace_copy(fqn, full_tensors)


def _create_verified_state_dict(
    pg: dist.ProcessGroup, mesh: DeviceMesh
) -> dict[str, torch.Tensor]:
    placements = [Shard(0)]
    state_dict = {
        "vision_model.vision_adapter.mlp.fc1.weight": torch.rand(
            4096, 5632, device="cuda", dtype=torch.bfloat16
        ),
        "vision_model.vision_adapter.mlp.fc2.weight": torch.rand(
            4096, 4096, device="cuda", dtype=torch.bfloat16
        ),
        "language_model.model.layers.3.feed_forward.experts.gate_up_proj": torch.rand(
            16, 5120, 16384, device="cuda", dtype=torch.bfloat16
        ),
    }
    return {k: distribute_tensor(v, mesh, placements) for k, v in state_dict.items()}


def _verify_state_dict(
    state_dict: dict[str, torch.Tensor], path: str, rank: int
) -> None:
    metadata_path = os.path.join(path, "model.safetensors.index.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)["weight_map"]
    all_filenames = set()
    for fqn, tensor in state_dict.items():
        filename = os.path.join(path, metadata[fqn])
        all_filenames.add(filename)

    stored_state_dict = {}
    from safetensors.torch import load_file as hf_load_file

    for filename in all_filenames:
        _sd = hf_load_file(filename)
        for k in list(_sd.keys()):
            if k not in state_dict:
                _sd.pop(k)
            else:
                stored_state_dict[k] = _sd[k]

    def read_and_verify_tensor(fqn: str, dtensor: DTensor) -> None:
        logger.info(f"Verifying {fqn} {dtensor.shape=} {dtensor.placements=} ")
        stored_tensor = stored_state_dict[fqn]
        full_tensor = dtensor.full_tensor()
        logger.info(f"Gather {fqn} {full_tensor.shape} completely.")

        if rank > 0:
            return

        stored_tensor = stored_tensor.to(device="cuda")
        logger.info(f"Move to GPU {fqn} completely.")

        assert stored_tensor.shape == full_tensor.shape, fqn
        assert stored_tensor.dtype == full_tensor.dtype, fqn
        assert stored_tensor.device == full_tensor.device, fqn
        assert torch.allclose(stored_tensor, full_tensor), fqn

    for k, v in state_dict.items():
        read_and_verify_tensor(k, v)


if __name__ == "__main__":
    init_logger()

    @dataclass
    class Checkpoint:
        convert_path: str = ""
        """Specify the path of the target checkpoint to convert."""

        convert_hf_token: str = ""
        """Specify the Hugging Face token to use when downloading checkpoints."""

        convert_load_every_n_ranks: int = 8
        """
        Specify the interval at which ranks are assigned to load checkpoints.

        For example, if this number is 4, then ranks 0, 4, 8, ... will load the
        checkpoint. Each loader is responsible for loading one file. If there
        are more loaders than files, only the first few loaders will be assigned
        to load the checkpoint. The default value is 8.
        """

        fake_model: bool = False
        """If true, the model will be fake."""

    @dataclass
    class MyJobConfig:
        checkpoint: Checkpoint = field(default_factory=Checkpoint)

    MergedJobConfig = ConfigManager._merge_configs(JobConfig, MyJobConfig)
    config_manager = ConfigManager(config_cls=MergedJobConfig)
    config = config_manager.parse_args()

    assert config.checkpoint.convert_path != ""

    trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(config)
        if os.path.exists(trainer.checkpointer.folder):
            raise RuntimeError(
                "The checkpoint folder already exists. Abort to avoid overwriting "
                f"the checkpoint. {trainer.checkpointer.folder=}"
            )
        if config.checkpoint.fake_model:
            state_dict = _create_verified_state_dict(
                trainer.world_mesh.get_group(), trainer.world_mesh
            )
        else:
            state_dict = trainer.checkpointer.states[MODEL].state_dict()

        size = 0
        for v in state_dict.values():
            size += v.numel() * v.element_size()
        logger.info(f"Total size of the model: {size / 1e9:.2f} GB")

        # Do not support PP yet, we will need to iterate over the PP dimension and
        # extract the corresponding state_dict and device_mesh.
        if "freqs_cis" in state_dict:
            state_dict.pop("freqs_cis")

        # Our tokenizer is not up-to-date yet.
        tok_embeddings_weight = state_dict.pop("tok_embeddings.weight")
        output_weight = state_dict.pop("output.weight")
        state_dict = CheckpointConverter(
            process_group=trainer.world_mesh.get_group(),
            path=config.checkpoint.convert_path,
            token=config.checkpoint.convert_hf_token,
            loader_every_n_ranks=config.checkpoint.convert_load_every_n_ranks,
        ).convert(state_dict)
        state_dict["tok_embeddings.weight"] = tok_embeddings_weight
        state_dict["output.weight"] = output_weight

        class DummyModel:
            def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
                self._state_dict = state_dict

            def state_dict(self) -> dict[str, torch.Tensor]:
                return self._state_dict

        if config.checkpoint.fake_model:
            begin = time.time()
            _verify_state_dict(
                state_dict,
                config.checkpoint.convert_path,
                trainer.world_mesh.get_rank(),
            )
            dist.barrier()
            logger.info(f"Verifies state_dict {time.time() - begin}.")
        else:
            # oh, this is pretty bad, when can we get rid of the freqs_cis issue?
            state_dict["freqs_cis"] = None
            trainer.checkpointer.states[MODEL] = DummyModel(state_dict)
            trainer.checkpointer.model_weights_only = True
            trainer.checkpointer.export_dtype = next(iter(state_dict.values())).dtype
            trainer.checkpointer.save(curr_step=0, force=True)
            time.sleep(2)
    finally:
        pass

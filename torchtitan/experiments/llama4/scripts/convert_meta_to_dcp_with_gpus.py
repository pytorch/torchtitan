# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
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

# Sharding dims for MP checkpoints

column_parallel = [
    "tok_embeddings",
    "wq",
    "wk",
    "wv",
    "wqkv",
    "w_in_shared_FD",
    "w_out_eF_D",
    "w_swiglu_FD",
    "output",
    "_linear",
    "c_fc",
    "vision_projection",
]

row_parallel = [
    "wo",
    "w_out_shared_DF",
    "w_in_eD_F",
    "moe_w_swiglu_eD_F",
    "c_proj",
]


def convert_to_titan_fqns(fqn: str) -> list[str]:
    # From the stored checkpoint keys to TorchTitan keys.
    if "wqkv" in fqn and "layer_norm_weight" not in fqn:
        ret = []
        for k in ("wq", "wk", "wv"):
            ret.append(fqn.replace("wqkv", k))
        return ret
    return [fqn]


def get_shard_dim(fqn: str) -> Optional[int]:
    if "bias" in fqn:
        # Some bias params are still sharded
        if "resblocks" in fqn:
            for k in ("wq", "wk", "wv", "c_fc"):
                if k in fqn:
                    return 0
        return None
    elif any([x in fqn for x in column_parallel]):
        return 0
    elif any([x in fqn for x in row_parallel]):
        return 1
    else:
        return None


def split_fused_qkv(shards: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    qkvs = [torch.split(shard, [640, 128, 128]) for shard in shards]
    q = torch.cat([qkv[0] for qkv in qkvs], dim=0)
    k = torch.cat([qkv[1] for qkv in qkvs], dim=0)
    v = torch.cat([qkv[2] for qkv in qkvs], dim=0)
    return q, k, v


@dataclass
class _Assignment:
    loader_id: int
    filename: str
    fqns: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    dtypes: tuple[torch.dtype, ...]


@dataclass
class _AssignmentRound:
    loader_assignments: dict[int, _Assignment]  # List of assignments for each loader


class CheckpointConverter:
    TOTAL_SHARDS = 8

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        path: str,
        loader_every_n_ranks: int = 8,
    ) -> None:
        self.path = path
        self.pg = process_group
        self.my_rank = dist.get_rank(self.pg)
        self.loader_every_n_ranks = loader_every_n_ranks
        self.loader_id = self.my_rank // loader_every_n_ranks
        self.should_load = (
            self.my_rank % loader_every_n_ranks == 0
            and self.loader_id < CheckpointConverter.TOTAL_SHARDS
        )
        self.total_loader = CheckpointConverter.TOTAL_SHARDS
        self.titan_fqn_to_stored_fqn: dict[str, str] = {}
        self.stored_fqn_to_titan_fqn: dict[str, list[str]] = {}
        self.total_send_bytes = 0
        self.total_recv_bytes = 0

    def convert(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        begin = time.time()
        self._load_metadata()
        self._create_fqn_mappings(state_dict)
        rounds = self._get_load_assignments(state_dict)

        for assignments in rounds:
            loader_assignments = assignments.loader_assignments
            loaded_state_dict = None
            # Let each loader to load its own data and move to its GPU.
            for i in range(self.total_loader):
                # This loader doesn't have any loading assignment for this round.
                if i not in loader_assignments:
                    continue
                # This rank is not the loader
                if i != self.loader_id or not self.should_load:
                    continue
                loaded_state_dict = self._load_round(loader_assignments[i])

            results = []
            for i in range(self.total_loader):
                if i not in loader_assignments:
                    continue

                if i == self.loader_id and self.should_load:
                    # This rank is the loader. It needs to send the loaded data to
                    # the other ranks.
                    assert loaded_state_dict is not None
                    results.append(
                        self._reshard_send(loader_assignments[i], loaded_state_dict)
                    )
                else:
                    results.append(
                        self._reshard_receive(loader_assignments[i], state_dict)
                    )

            self._reshard(results, state_dict)

        torch.cuda.synchronize()
        logger.info(f"Checkpoint conversion took {time.time() - begin:.2f} seconds.")
        logger.info(f"Total send bytes: {self.total_send_bytes / 1e9:.2f} GB")
        logger.info(f"Total recv bytes: {self.total_recv_bytes / 1e9:.2f} GB")
        return state_dict

    def _get_file_path(self, loader_id: int) -> str:
        return os.path.join(self.path, f"consolidated.0{loader_id}.pth")

    def _load_metadata(self) -> None:
        if not self.should_load:
            self.read_dict = {}
            return
        self.read_dict = torch.load(
            self._get_file_path(self.loader_id),
            mmap=True,
            weights_only=False,
        )

    def _create_fqn_mappings(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not self.read_dict:
            return

        # Create the mapping from the stored checkpoint keys to TorchTitan keys.
        for fqn in list(self.read_dict.keys()):
            titan_fqns = convert_to_titan_fqns(fqn)
            # We don't know how to process _extra_state
            if "_extra_state" in fqn:
                self.read_dict.pop(fqn)
                continue

            if titan_fqns[0] not in state_dict:
                for titan_fqn in titan_fqns:
                    assert titan_fqns[0] not in state_dict
                self.read_dict.pop(fqn)
                continue
            self.stored_fqn_to_titan_fqn[fqn] = titan_fqns
            for titan_fqn in titan_fqns:
                self.titan_fqn_to_stored_fqn[titan_fqn] = fqn

        assert set(state_dict.keys()) == set(self.titan_fqn_to_stored_fqn.keys()), (
            set(state_dict.keys()) - set(self.titan_fqn_to_stored_fqn.keys()),
            set(self.titan_fqn_to_stored_fqn.keys()) - set(state_dict.keys()),
        )

    def _get_load_assignments(
        self, state_dict: dict[str, torch.Tensor]
    ) -> list[_AssignmentRound]:
        if self.my_rank == 0:
            rounds: list[_AssignmentRound] = []
            size = 0
            fqns = []
            shapes = []
            dtypes = []

            # All loader must load all the FQNs because the checkpoint is purely TP sharded.
            all_keys = list(self.read_dict.keys())
            for fqn in all_keys:
                fqns.append(fqn)
                shapes.append(self.read_dict[fqn].shape)
                dtypes.append(self.read_dict[fqn].dtype)
                size += self.read_dict[fqn].numel() * self.read_dict[fqn].element_size()
                if size < 1e9 and fqn != all_keys[-1]:
                    continue

                logger.info(f"Adding {fqns} to round {len(rounds)}")
                round_assignment = _AssignmentRound(loader_assignments={})
                for loader_id in range(self.total_loader):
                    path = self._get_file_path(loader_id)
                    round_assignment.loader_assignments[loader_id] = _Assignment(
                        filename=path,
                        fqns=tuple(fqns),
                        shapes=tuple(shapes),
                        dtypes=tuple(dtypes),
                        loader_id=loader_id,
                    )
                rounds.append(round_assignment)
                size = 0
                fqns.clear()
                shapes.clear()
                dtypes.clear()

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

    def _load_round(self, assignment: _Assignment) -> dict[str, torch.Tensor]:
        ret = {}
        assert self.read_dict
        for fqn in assignment.fqns:
            ret[fqn] = self.read_dict[fqn].to(device="cuda")
        return ret

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
        logger.info(f"Sending {assignment.filename} from {rank} {self.loader_id}")
        logger.info(f"Sending {assignment.fqns}")
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
        results: list[dict[str, torch.Tensor]],
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        def _inplace_copy(fqn: str, full_tensors: tuple[torch.Tensor, ...]):
            titan_fqns = self.stored_fqn_to_titan_fqn[fqn]
            assert len(titan_fqns) == len(full_tensors)
            for titan_fqn, full_tensor in zip(titan_fqns, full_tensors):
                dtensor = state_dict[titan_fqn]
                logger.info(f"{titan_fqn} {full_tensor.sum()}")
                assert isinstance(dtensor, DTensor)
                shape, offset = compute_local_shape_and_global_offset(
                    full_tensor.shape, dtensor.device_mesh, dtensor.placements
                )
                slices = [
                    slice(cur_offset, cur_offset + cur_shape)
                    for cur_shape, cur_offset in zip(shape, offset)
                ]
                logger.info(
                    f"Copying {titan_fqn} with {slices=} {dtensor._local_tensor.shape=} "
                    f"{shape=} {offset=} {self.my_rank=} {dtensor.shape=} {full_tensor.shape=} "
                    f"{dtensor.placements=} {dtensor.device_mesh=} "
                )
                dtensor.to_local().copy_(full_tensor[slices])

        def _concat_shards(fqn, shards: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
            if "wqkv" in fqn:
                if "layer_norm" in fqn:
                    return (shards[0],)
                return split_fused_qkv(shards)

            shard_dim = get_shard_dim(fqn)
            if shard_dim is None:
                return (shards[0],)
            return (torch.cat(shards, dim=shard_dim),)

        fqns = list(results[0].keys())
        for result in results:
            assert list(result.keys()) == fqns

        for fqn in fqns:
            full_tensors = _concat_shards(fqn, [result[fqn] for result in results])
            _inplace_copy(fqn, full_tensors)


def _create_verified_state_dict(
    pg: dist.ProcessGroup, mesh: DeviceMesh
) -> dict[str, torch.Tensor]:
    placements = [Shard(0)]
    state_dict = {
        "tok_embeddings.weight": torch.rand(
            25256 * 8, 5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.attention.wqkv.layer_norm_weight": torch.rand(
            5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.attention.wq.weight": torch.rand(
            640 * 8, 5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.attention.wk.weight": torch.rand(
            128 * 8, 5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.attention.wv.weight": torch.rand(
            128 * 8, 5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.attention.wo.weight": torch.rand(
            5120, 640 * 8, device="cuda", dtype=torch.bfloat16
        ),
        # "layers.47.feed_forward.router_DE": torch.rand(5120, 128, device="cuda", dtype=torch.bfloat16),
        # "layers.47.feed_forward.running_gate_stats_3E": torch.rand(3, 128, device="cuda", dtype=torch.bfloat16),
        # "layers.47.feed_forward.global_gate_stats_3E": torch.rand(3, 128, device="cuda", dtype=torch.bfloat16),
        "layers.47.feed_forward.w_in_shared_FD.weight": torch.rand(
            1024 * 8, 5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.feed_forward.w_out_shared_DF.weight": torch.rand(
            5120, 1024 * 8, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.feed_forward.w_swiglu_FD.weight": torch.rand(
            1024 * 8, 5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.feed_forward.norm.weight": torch.rand(
            5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.feed_forward.experts.moe_w_in_eD_F": torch.rand(
            655360, 1024 * 8, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.feed_forward.experts.moe_w_out_eF_D": torch.rand(
            131072 * 8, 5120, device="cuda", dtype=torch.bfloat16
        ),
        "layers.47.feed_forward.experts.moe_w_swiglu_eD_F": torch.rand(
            655360, 1024 * 8, device="cuda", dtype=torch.bfloat16
        ),
    }
    return {k: distribute_tensor(v, mesh, placements) for k, v in state_dict.items()}


def _verify_state_dict(
    state_dict: dict[str, torch.Tensor], path: str, rank: int
) -> None:
    stored_state_dicts = [
        torch.load(
            os.path.join(path, f"consolidated.0{i}.pth"),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for i in range(8)
    ]

    def read_and_verify_tensor(fqn: str, dtensor: DTensor) -> None:
        logger.info(f"Verifying {fqn} {dtensor.shape=} {dtensor.placements=} ")
        shards = [stored_state_dicts[i][fqn] for i in range(8)]
        full_tensor = dtensor.full_tensor()
        logger.info(f"Gather {fqn} {full_tensor.shape} completely.")

        if rank > 0:
            return

        if len(shards[0].shape) == 1:
            assert full_tensor.shape == shards[0].shape, fqn
            assert torch.allclose(shards[0].to(device="cuda"), full_tensor), fqn
            return
        elif shards[0].shape[0] == full_tensor.shape[0]:
            concat_shards = torch.cat(shards, dim=1)
            logger.info(f"Load {fqn} completely.")
        elif shards[0].shape[1] == full_tensor.shape[1]:
            concat_shards = torch.cat(shards, dim=0)
            logger.info(f"Load {fqn} completely.")

        concat_shards = concat_shards.to(device="cuda")
        logger.info(f"Move to GPU {fqn} completely.")

        assert concat_shards.shape == full_tensor.shape, fqn
        assert concat_shards.dtype == full_tensor.dtype, fqn
        assert concat_shards.device == full_tensor.device, fqn
        assert torch.allclose(concat_shards, full_tensor), fqn

    for k, v in state_dict.items():
        if "wq" in k and "wqkv" not in k:
            pass
        elif "wk" in k:
            pass
        elif "wv" in k:
            pass
        else:
            assert v is not None, k
            read_and_verify_tensor(k, v)


if __name__ == "__main__":
    init_logger()

    @dataclass
    class Checkpoint:
        convert_path: str = ""
        """Specify the path of the target checkpoint to convert."""

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
        if "freq_cis" in state_dict:
            state_dict.pop("freqs_cis")

        state_dict = CheckpointConverter(
            process_group=trainer.world_mesh.get_group(),
            path=config.checkpoint.convert_path,
            loader_every_n_ranks=config.checkpoint.convert_load_every_n_ranks,
        ).convert(state_dict)

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

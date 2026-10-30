# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static Kimi K3 pipeline metadata tests on CPU."""

import unittest
from unittest.mock import Mock

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _MeshLayout, DeviceMesh
from torch.distributed.pipelining._utils import InferenceMode
from torch.distributed.pipelining.schedules import Schedule1F1B
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import TrainingConfig
from torchtitan.distributed.pipeline_parallel import TorchTitanPipelineStage
from torchtitan.models.kimi_k3.pipeline_parallel import (
    _build_attn_res_stage_metadata,
    _swap_in_attn_res_stages,
)
from torchtitan.models.kimi_k3.pipeline_parallel.layout import BlockLayoutTables


def _layout() -> BlockLayoutTables:
    return BlockLayoutTables(
        stage_to_rank={stage: stage for stage in range(8)},
        n_layers=8,
        layers_per_block=1,
        layer_to_stage={layer: layer for layer in range(8)},
    )


def _metadata(loss_fn):
    return _build_attn_res_stage_metadata(
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4,
            mixed_precision_param="bfloat16",
        ),
        model_config=Mock(dim=8, vocab_size=16),
        loss_fn=loss_fn,
        layout=_layout(),
    )


class TestAttnResStageMetadata(unittest.TestCase):
    def test_cp_metadata_uses_local_tokens_for_all_forward_and_backward_carriers(self):
        metadata = _build_attn_res_stage_metadata(
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=65_536,
                mixed_precision_param="bfloat16",
            ),
            model_config=Mock(dim=8, vocab_size=16),
            loss_fn=ChunkedLossWrapper(ChunkedLossWrapper.Config()),
            layout=_layout(),
            context_parallel_degree=16,
        )
        for stage_metadata in metadata.values():
            for field in ("input_args", "output_args", "input_grads", "output_grads"):
                for tensor in stage_metadata[field]:
                    if tensor is not None:
                        self.assertEqual(tensor.shape[0], 4096)
                        self.assertEqual(type(tensor), torch.Tensor)
        self.assertEqual(metadata[3]["input_args"][1].shape, (4096, 3, 8))
        self.assertEqual(metadata[3]["output_args"][1].shape, (4096, 4, 8))

    def test_cp_metadata_rejects_truncated_partition_shapes(self):
        with self.assertRaisesRegex(ValueError, "divisible by CP"):
            _build_attn_res_stage_metadata(
                training=TrainingConfig(num_tokens_per_microbatch_per_dp_rank=17),
                model_config=Mock(dim=8, vocab_size=16),
                loss_fn=ChunkedLossWrapper(ChunkedLossWrapper.Config()),
                layout=_layout(),
                context_parallel_degree=16,
            )

    def test_metadata_describes_each_carrier_boundary(self):
        metadata = _metadata(ChunkedLossWrapper(ChunkedLossWrapper.Config()))

        self.assertEqual([tuple(t.shape) for t in metadata[0]["input_args"]], [(4,)])
        self.assertEqual(
            [tuple(t.shape) for t in metadata[0]["output_args"]],
            [(4, 8), (4, 1, 8)],
        )
        self.assertEqual(
            [tuple(t.shape) for t in metadata[3]["input_args"]],
            [(4, 8), (4, 3, 8)],
        )
        self.assertEqual(
            [tuple(t.shape) for t in metadata[3]["output_args"]],
            [(4, 8), (4, 4, 8)],
        )
        self.assertEqual(
            [tuple(t.shape) for t in metadata[7]["input_args"]],
            [(4, 8), (4, 7, 8)],
        )
        self.assertEqual([tuple(t.shape) for t in metadata[7]["output_args"]], [(4, 8)])
        for stage_metadata in metadata.values():
            for tensor in (
                *stage_metadata["input_args"],
                *stage_metadata["output_args"],
            ):
                self.assertEqual(tensor.device.type, "meta")
            for tensor in (
                *stage_metadata["input_grads"],
                *stage_metadata["output_grads"],
            ):
                if tensor is not None:
                    self.assertFalse(tensor.requires_grad)

    def test_non_chunked_loss_declares_logits(self):
        metadata = _metadata(CrossEntropyLoss(CrossEntropyLoss.Config()))
        (logits_TV,) = metadata[7]["output_args"]
        self.assertEqual(tuple(logits_TV.shape), (4, 16))
        self.assertEqual(logits_TV.dtype, torch.bfloat16)


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return output.sum()


def _get_mesh(
    mesh_axis_names: tuple[str, ...], mesh_layout: _MeshLayout | None
) -> DeviceMesh:
    raise AssertionError("static metadata construction must not request a mesh")


class TestStaticMetadataSwap(DTensorTestBase):
    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 1

    @with_comms
    def test_rebuilt_stage_does_not_need_dynamic_inference(self):
        old_stage = TorchTitanPipelineStage(
            nn.Linear(2, 2),
            0,
            1,
            torch.device("cpu"),
            get_mesh=_get_mesh,
        )
        schedule = Schedule1F1B(old_stage, n_microbatches=1, loss_fn=_loss)
        input_TD = torch.empty(1, 2, device="meta").requires_grad_()
        output_TD = torch.empty(1, 2, device="meta").requires_grad_()
        metadata = {
            0: {
                "input_args": (input_TD,),
                "output_args": (output_TD,),
                "input_grads": (input_TD.detach(),),
                "output_grads": (None,),
            }
        }

        (stage,) = _swap_in_attn_res_stages(schedule, metadata)

        self.assertFalse(InferenceMode.needs_dynamic(stage._user_meta, True))


if __name__ == "__main__":
    unittest.main()

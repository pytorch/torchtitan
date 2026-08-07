# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reduced Kimi K3 numerical parity against the released HuggingFace model.

The reference tensors were generated with the released ``modeling_kimi_k3.py``
at moonshotai/Kimi-K3 commit c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721.
The reduced model uses the same deterministic parameter and input construction
as this test, then maps its state dict through ``KimiK3StateDictAdapter`` and
loads the HuggingFace model strictly. The HuggingFace eager attention paths and
a pure PyTorch implementation of the released FLA KDA API produced the frozen
float32 values below.
"""

import base64
import unittest

import torch

# Inherits test_kimi_k3's module-level skip when FLA is not installed.
from tests.unit_tests.test_kimi_k3 import _small_model_config


_TEXT_LOGITS_BASE64 = (
    "qvilPdqoTz1q2ZQ8yRJ4vMTJQ70Rj6C9X03YvVFcA75J6BS+9QggvohDJL5WaSG+"
    "2ZkXvldBB75jKOK9Rg6svdUSXb3KArG8+II/PI8yNj3RVJo9O+nSPQkzAT6mXxM+"
    "5jEfPlAnJD4rCSI+2u4YPtA8CT6FQOc9Xg+yPW1iaj2gtbM9feBrPUtVzDxJyg+8L"
    "oMsvfvRlr314dC9ZPgAvobwE74FiCC+9zMmvrq1JL7NHRy+BssMvvbM7r2RuLm9N"
    "0V5vdO06Lwwo6w7IJUePdtfkD3pO8s9IVX9PU5BEj4Hih8+H/IlPuQyJT6RVB0+/"
    "a0OPrPB8z1Zpb89HEaDPbBPsz2Ke249vx/YPNAh2LtI7iC9K/SPvYw8yb0m2Pm9Rd"
    "cPvhOPHL4mhyK+m30hvuZ9Gb5O4Aq+NYzsvVolub2Ohnu9S9XzvOAqTzsvQRM9Rpu"
    "JPWmnwz1aRPU9SicOPqWLGz5zOyI+5+whPmOjGj5Yrww+dlXxPe/kvj3KOYQ9/Y/L"
    "PY3JhD3NJuE8JBo1vO4sSb2WM669RU7wvYUGFL5QhCm+YrM3vi73Pb6TCjy+zAIy"
    "volOIL4xsQe+wXTSvZp0jL3WzAC9M+fmOy5IOT1X26Y9OeHpPQ1pET7CnCc+w5Y2"
    "PsKxPT5Znzw+WWszPlZ7Ij46igo+Z0DZPeYOlD0="
)
_VISION_OUTPUT_BASE64 = (
    "zZB1v4gVxT0AaYs/uXavP7LLQT9v+7O+komdv+wxp7+XQAe/6k8YP4c7qj8+IJk/"
    "3eWPPvtwUb+6DLG/EbyFvw=="
)
_MULTIMODAL_LOGITS_BASE64 = (
    "qvilPdqoTz1q2ZQ8yRJ4vMTJQ70Rj6C9X03YvVFcA75J6BS+9QggvohDJL5WaSG+"
    "2ZkXvldBB75jKOK9Rg6svdUSXb3KArG8+II/PI8yNj3RVJo9O+nSPQkzAT6mXxM+"
    "5jEfPlAnJD4rCSI+2u4YPtA8CT6FQOc9Xg+yPW1iaj2gtbM9feBrPUtVzDxJyg+8L"
    "oMsvfvRlr314dC9ZPgAvobwE74FiCC+9zMmvrq1JL7NHRy+BssMvvbM7r2RuLm9N"
    "0V5vdO06Lwwo6w7IJUePdtfkD3pO8s9IVX9PU5BEj4Hih8+H/IlPuQyJT6RVB0+/"
    "a0OPrPB8z1Zpb89HEaDPWJ9mrsLzzW63QhcO40E7jsF4TE8vRRlPDI0hzz1CZY85G"
    "eePMHxnzyGlpo8P5GOPGzNeDyMvkk8NP0RPNjhpzvLMZI6obhAu5kd4btQFSy8yjB"
    "gvBFRhbxxypS85dmdvFEboLzVdZu8wRyQvIoafbxmEk+8Gx0YvGoytbtTasm6Marc"
    "PQTmjD2lO9w89J6EvBnhb725jcm9Tz0JvjjJJ758GT++1ixOvuZcVL5jZVG+CmdF"
    "vjjmML4uxRS+uHTkveuFlb0FSgC9Hf4/PHI3Xj18bcE9VrQFPmfuJD5TDD0++ANN"
    "PiAlVD4cIVI+Kw5HPoJmMz4JAxg+ACTsPfoTnj3eefM9VzyZPQiP4TxLjKu8rVCM"
    "vcOx5737ihy+hn0+vto5WL78o2i+wQZvvrUba74TDl2+7XhFvo1gJb45Tvy9+/qi"
    "vT5CBb0RYII8gGmCPb6b3j2zmhg+alM7Ptr4VT73ZGc+eNduPjH+az6P+F4+S1ZI"
    "PigRKT49ggI+LKasPYLq2z3AyI09JynmPN94abz7TGW9myzDvWukBb607yO+1Ck7"
    "vnhSSr5ZwlC+azJOvvS+Qr5U5i6+koMTvsSJ471cPZa94u0EvYhnITxr5FM9QSa7"
    "PX4kAj6oGSE+7xw5PmElST4HglA+p+FOPi1WRD4AVDE+46wWPuIN6z0NoJ49"
)


def _decode_float32(encoded: str, shape: tuple[int, ...]) -> torch.Tensor:
    values = torch.frombuffer(
        bytearray(base64.b64decode(encoded)),
        dtype=torch.float32,
    )
    return values.reshape(shape)


def _fill_reference_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter_index, (name, parameter) in enumerate(model.named_parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
            ).reshape(parameter.shape)
            values = torch.sin(values * 0.013 + parameter_index * 0.17) * 0.02
            if name.endswith("norm.weight"):
                values = values + 1.0
            parameter.copy_(values.to(parameter.dtype))

        for module in model.modules():
            expert_bias = getattr(module, "expert_bias_E", None)
            if expert_bias is not None:
                expert_bias.copy_(torch.linspace(-0.01, 0.01, expert_bias.numel()))


class TestKimiK3HuggingFaceParity(unittest.TestCase):
    @torch.no_grad()
    def test_reduced_model_matches_released_huggingface_reference(self):
        config = _small_model_config()
        model = config.build()
        model.init_states()
        _fill_reference_parameters(model)
        model.eval()

        router_ids: list[torch.Tensor] = []
        moe_layer = next(
            layer for layer in model.layers.values() if layer.moe is not None
        )
        router_hook = moe_layer.moe.router.register_forward_hook(
            lambda _module, _inputs, output: router_ids.append(
                output[1] if len(output) == 3 else output[0]
            )
        )
        self.addCleanup(router_hook.remove)

        text_tokens_BL = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        text_logits_BLV = model(text_tokens_BL)
        torch.testing.assert_close(
            text_logits_BLV,
            _decode_float32(_TEXT_LOGITS_BASE64, (1, 4, 32)),
            atol=2e-4,
            rtol=2e-4,
        )
        torch.testing.assert_close(
            router_ids.pop().reshape(-1),
            torch.ones(4, dtype=torch.int64),
            atol=0,
            rtol=0,
        )

        patches_PCHW = torch.sin(torch.arange(48, dtype=torch.float32) * 0.07).reshape(
            4, 3, 2, 2
        )
        pixels_NPK = patches_PCHW.reshape(1, 4, 12)
        grid_thw_N3 = torch.tensor([[1, 2, 2]], dtype=torch.long)
        vision_output_NMD = model.vision_encoder(
            pixels_NPK,
            grid_thw=grid_thw_N3,
        )
        torch.testing.assert_close(
            vision_output_NMD,
            _decode_float32(_VISION_OUTPUT_BASE64, (1, 1, 16)),
            atol=2e-4,
            rtol=2e-4,
        )

        multimodal_tokens_BL = torch.tensor(
            [[1, 2, 7, 3, 4, 5]],
            dtype=torch.long,
        )
        multimodal_logits_BLV = model(
            multimodal_tokens_BL,
            pixel_values=pixels_NPK,
            grid_thw=grid_thw_N3,
            special_tokens={"image_id": 7},
        )
        torch.testing.assert_close(
            multimodal_logits_BLV,
            _decode_float32(_MULTIMODAL_LOGITS_BASE64, (1, 6, 32)),
            atol=2e-4,
            rtol=2e-4,
        )
        torch.testing.assert_close(
            router_ids.pop().reshape(-1),
            torch.ones(6, dtype=torch.int64),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()

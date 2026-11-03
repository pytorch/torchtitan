# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import math

import pytest
import torch
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.accelerator_graph import (
    accelerator_graph_teardown,
    wrap_fwd_bwd_with_accelerator_graph,
)
from torchtitan.observability import tensor_logger
from torchtitan.observability.tensor_logger.statistics import (
    accumulate_tensor_statistics,
    SUM_STATISTIC_FIELD_COUNT,
)
from torchtitan.tools.utils import device_type

# numel, nonfinite_count, zero_count, observation_count
_COUNT_COLUMNS = slice(0, 4)


def _accumulate(
    value: torch.Tensor, *, enabled: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_statistics = torch.zeros(1, SUM_STATISTIC_FIELD_COUNT, device=value.device)
    maxima = torch.full((1,), -math.inf, device=value.device)
    flag = torch.tensor(enabled, dtype=torch.int32, device=value.device)
    # Twice, so the test also covers adding into a non-empty row.
    for _ in range(2):
        accumulate_tensor_statistics(value, sum_statistics, maxima, flag, 0)
    return sum_statistics.cpu(), maxima.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_kernel_matches_reference(dtype):
    """The Triton kernel matches the CPU reference, including layouts it reads
    without a copy (transpose) and with a copy (slice with gaps)."""
    torch.manual_seed(0)
    base = torch.randn(64, 48, 32)
    base[0, 0, :4] = torch.tensor([math.nan, math.inf, -math.inf, 0.0])
    views = {
        "contiguous": lambda t: t,
        "transpose": lambda t: t.transpose(0, 1),
        "strided_slice": lambda t: t[:, ::2, 1:],
    }
    inputs = {name: (base, view) for name, view in views.items()}
    # Large enough for the kernel's grid-stride loop.
    inputs["loop_path"] = (torch.randn(5_000_000), lambda t: t)
    for name, (value, view) in inputs.items():
        cpu_sums, cpu_maxima = _accumulate(view(value.to(dtype)), enabled=1)
        gpu_sums, gpu_maxima = _accumulate(
            view(value.to(device_type, dtype)), enabled=1
        )
        torch.testing.assert_close(
            gpu_sums[:, _COUNT_COLUMNS],
            cpu_sums[:, _COUNT_COLUMNS],
            rtol=0,
            atol=0,
            msg=name,
        )
        # Float sums use atomic adds, so their order differs from the reference.
        torch.testing.assert_close(gpu_sums, cpu_sums, rtol=1e-5, atol=0, msg=name)
        torch.testing.assert_close(gpu_maxima, cpu_maxima, rtol=0, atol=0, msg=name)

    disabled_sums, disabled_maxima = _accumulate(base.to(device_type), enabled=0)
    assert not disabled_sums.any()
    assert disabled_maxima.item() == -math.inf


class _Logged(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16, device=device_type)
        tensor_logger.register_fwd_bwd(self, "y")


def test_graph_capture_on_a_non_recording_step_keeps_logging():
    """With log_freq=2 and capture on step 3 (not recorded), replays still record."""
    module = _Logged()
    logger = tensor_logger.TensorLogger.Config(log_freq=2).build(
        model_parts=[module],
        device=torch.device(device_type),
        accelerator_graph_enabled=True,
        pp_schedule=None,
    )

    def forward_backward(x: torch.Tensor) -> torch.Tensor:
        y = module.linear(x)
        tensor_logger.log_fwd_bwd(module, y=y)
        loss = y.square().mean()
        loss.backward()
        return loss.detach()

    # Steps 1 and 2 run eagerly; step 3 captures; step 4 replays.
    run = wrap_fwd_bwd_with_accelerator_graph(
        forward_backward, parameters=module.parameters(), num_warmup_iterations=2
    )
    x = torch.randn(8, 16, device=device_type)
    published = {}
    try:
        for step in range(1, 7):
            module.zero_grad(set_to_none=True)
            with logger.maybe_record(step):
                run(x)
            published[step] = logger.collect()
    finally:
        logger.close()
        accelerator_graph_teardown()

    for step in (1, 3, 5):
        assert published[step] == {}
    for step in (2, 4, 6):
        assert published[step]["tensor_logger/y.fwd.observation_count"] == 1
        assert published[step]["tensor_logger/y.bwd.observation_count"] == 1
        assert published[step]["tensor_logger/y.fwd.numel"] == 8 * 16
    # The same input gives the same statistics eagerly (step 2) and on replay.
    assert (
        published[4]["tensor_logger/y.fwd.abs_mean"]
        == published[2]["tensor_logger/y.fwd.abs_mean"]
    )


# kimi_k3 and muse_glimmer override Decoder.forward and some module forwards.
@pytest.mark.parametrize("name", ["llama3", "deepseek_v3", "kimi_k3", "muse_glimmer"])
def test_shared_callsites_record_each_tensor_once(name):
    """Every registered row is observed once per microbatch, forward and backward."""
    torch.manual_seed(0)
    module = importlib.import_module(f"torchtitan.models.{name}")
    model_config = module.build_model_config(
        "debugmodel", attn_backend="varlen", parallelism=ParallelismConfig()
    )
    with torch.device(device_type):
        model = model_config.build()
    model.init_states()
    num_documents, seq_len = 2, 64
    tokens = torch.randint(
        0, model_config.vocab_size, (num_documents * seq_len,), device=device_type
    )
    positions = torch.arange(seq_len, device=device_type).repeat(num_documents)
    logger = tensor_logger.TensorLogger.Config(log_freq=1).build(
        model_parts=[model],
        device=torch.device(device_type),
        accelerator_graph_enabled=False,
        pp_schedule=None,
    )
    try:
        with logger.maybe_record(step=1):
            output = model(
                tokens,
                attention_metadata=model._get_attention_metadata(positions),
                positions=positions,
                # MoE models normalize the aux loss by the routed token count.
                aux_loss_denominators=torch.tensor(
                    [float(tokens.numel())], device=device_type
                ),
            )
            output.float().square().mean().backward()
        metrics = logger.collect()
    finally:
        logger.close()
    counts = {
        row: value
        for row, value in metrics.items()
        if row.endswith(".observation_count")
    }
    assert len(counts) == len(logger.row_names)
    assert set(counts.values()) == {1}

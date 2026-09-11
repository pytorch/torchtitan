# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig


@pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW"])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("implementation", ["fused", "fused_opt_states_bf16"])
@pytest.mark.parametrize("native_load", [False, True])
def test_load_preserves_configured_state_dtype(
    optimizer_name, amsgrad, implementation, native_load
):
    state_dtype = (
        torch.bfloat16 if implementation == "fused_opt_states_bf16" else torch.float32
    )
    moments = ["exp_avg", "exp_avg_sq"]
    if amsgrad:
        moments.append("max_exp_avg_sq")

    def build():
        model = torch.nn.Linear(4, 4)
        container = OptimizersContainer.Config(
            implementation=implementation,
            param_groups=[
                ParamGroupConfig(
                    pattern=pattern,
                    optimizer_name=optimizer_name,
                    optimizer_kwargs={"lr": lr, "amsgrad": amsgrad},
                )
                for pattern, lr in (("weight", 0.01), ("bias", 0.02))
            ],
        ).build(model_parts=[model])
        # CPU does not support the fused fp32-params/bf16-states update. Seed
        # nonempty state directly to exercise the real checkpoint load path.
        for optimizer in container.optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    optimizer.state[param] = {
                        "step": torch.tensor(1024.0),
                        **{
                            key: torch.full_like(param, 0.25, dtype=state_dtype)
                            for key in moments
                        },
                        "extra_state": torch.tensor(1.25),
                    }
        return container

    source, resumed = build(), build()
    checkpoint = copy.deepcopy(
        source.optimizers[0].state_dict() if native_load else source.state_dict()
    )
    expected = copy.deepcopy(source.state_dict())
    for optimizer in resumed.optimizers:
        for state in optimizer.state.values():
            for value in state.values():
                value.zero_()
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    # Repeated loads also exercise preservation of param_names in flat state.
    for _ in range(2):
        if native_load:
            resumed.optimizers[0].load_state_dict(checkpoint)
        else:
            resumed.load_state_dict(checkpoint)
        actual = resumed.state_dict()
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            else:
                assert actual[key] == value

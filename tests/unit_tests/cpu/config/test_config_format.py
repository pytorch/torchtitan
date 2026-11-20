# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from torchtitan.config.configurable import Configurable
from torchtitan.config.format import format_config, format_config_json


def _make_init_closure(scale: float) -> Callable[[float], float]:
    def _init(x: float) -> float:
        return x * scale

    return _init


def _scaled_fn(x: float, *, std: float) -> float:
    return x * std


class _DenseLinear(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        param_init: dict[str, Callable] = field(default_factory=dict)

    def __init__(self, config: Config):
        self.config = config


class _QuantLinear(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        # Intentionally identical field names to _DenseLinear.Config.
        dim: int
        param_init: dict[str, Callable] = field(default_factory=dict)

    def __init__(self, config: Config):
        self.config = config


class _MoEBlock(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        num_experts: int

    def __init__(self, config: Config):
        self.config = config


class _Layer(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        ffn: Configurable.Config
        layer_id: int

    def __init__(self, config: Config):
        self.config = config


@dataclass(frozen=True, slots=True)
class _OpaqueLayout:
    spec: str

    def to_dict(self) -> dict[str, str]:
        return {"repr": repr(self)}


class _Model(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        layers: list[_Layer.Config]
        sharding_by_fqn: dict[str, _OpaqueLayout] = field(default_factory=dict)

    def __init__(self, config: Config):
        self.config = config


def test_folds_homogeneous_layers_with_closures_and_varying_scalars() -> None:
    """Closures with distinct memory addresses and depth-scaled scalars still fold."""
    layers = [
        _Layer.Config(
            layer_id=i,
            ffn=_DenseLinear.Config(
                dim=4096,
                param_init={
                    "w1": _make_init_closure(1.0),
                    "w2": partial(_scaled_fn, std=0.02 / (i + 1)),
                },
            ),
        )
        for i in range(10)
    ]
    cfg = _Model.Config(layers=layers)

    py_out = format_config(cfg)
    assert "_Model.Config(" in py_out
    assert "'[0..9] (10 items)': _Layer.Config(" in py_out
    assert "dim=4096," in py_out
    assert "at 0x" not in py_out
    assert "<varies across 10 items: [0]=0, ..., [9]=9>" in py_out

    parsed = json.loads(format_config_json(cfg))
    assert list(parsed["layers"].keys()) == ["[0..9] (10 items)"]
    group = parsed["layers"]["[0..9] (10 items)"]
    assert group["ffn"]["dim"] == 4096
    assert (
        group["ffn"]["param_init"]["w1"]
        == "<function _make_init_closure.<locals>._init>"
    )
    assert group["layer_id"] == "<varies across 10 items: [0]=0, ..., [9]=9>"


def test_preserves_distinct_config_subclasses_with_identical_fields() -> None:
    """Sibling Config classes with identical fields must never merge together."""
    layers = [
        _Layer.Config(layer_id=0, ffn=_DenseLinear.Config(dim=1024)),
        _Layer.Config(layer_id=1, ffn=_DenseLinear.Config(dim=1024)),
        _Layer.Config(layer_id=2, ffn=_QuantLinear.Config(dim=1024)),
        _Layer.Config(layer_id=3, ffn=_MoEBlock.Config(dim=1024, num_experts=8)),
        _Layer.Config(layer_id=4, ffn=_MoEBlock.Config(dim=1024, num_experts=8)),
    ]
    cfg = _Model.Config(layers=layers)

    parsed = json.loads(format_config_json(cfg))
    assert list(parsed["layers"].keys()) == [
        "[0,1] (2 items)",
        "[2]",
        "[3,4] (2 items)",
    ]

    py_out = format_config(cfg)
    assert "_DenseLinear.Config(" in py_out
    assert "_QuantLinear.Config(" in py_out
    assert "_MoEBlock.Config(" in py_out


def test_surfaces_small_per_layer_overrides_by_index_range() -> None:
    """A single-layer scalar override stays explicitly visible with its index."""
    layers = [
        _Layer.Config(
            layer_id=0,
            ffn=_DenseLinear.Config(dim=2048 if i == 3 else 4096),
        )
        for i in range(6)
    ]
    cfg = _Model.Config(layers=layers)

    parsed = json.loads(format_config_json(cfg))
    dim_entry = parsed["layers"]["[0..5] (6 items)"]["ffn"]["dim"]
    assert dim_entry == {"[0..2, 4,5]": 4096, "[3]": 2048}


def test_folds_indexed_dict_keys_losslessly_and_keeps_distinct_values_separate() -> None:
    """Dotted FQN maps fold keys only when their values match."""
    owned = _OpaqueLayout("Owned()")
    sharded = _OpaqueLayout("Shard(0)")
    cfg = _Model.Config(
        layers=[],
        sharding_by_fqn={
            **{f"layers.{i}.wq.weight": owned for i in range(5)},
            "layers.0.wo.weight": owned,
            "layers.1.wo.weight": sharded,
            "layers.2.wo.weight": owned,
        },
    )

    parsed = json.loads(format_config_json(cfg))
    assert parsed["sharding_by_fqn"] == {
        "layers.{0..4}.wq.weight": {"repr": "_OpaqueLayout(spec='Owned()')"},
        "layers.{0,2}.wo.weight": {"repr": "_OpaqueLayout(spec='Owned()')"},
        "layers.1.wo.weight": {"repr": "_OpaqueLayout(spec='Shard(0)')"},
    }

    py_out = format_config(cfg)
    assert "'layers.{0..4}.wq.weight': _OpaqueLayout(spec='Owned()')," in py_out


def test_trainer_maybe_log_compacts_stdout_but_saves_full_to_dict(
    tmp_path, caplog, monkeypatch
) -> None:
    """debug.print_config logs compacted JSON while save_config_file keeps full JSON."""
    import torch.distributed as dist

    from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel

    cfg = llama3_debugmodel()
    cfg.dump_folder = str(tmp_path)
    cfg.debug.print_config = True
    cfg.debug.save_config_file = "config.json"

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    with caplog.at_level(logging.INFO, logger="torchtitan.trainer"):
        cfg.maybe_log()

    marker = "Running with configs: "
    log_msg = next(
        rec.getMessage() for rec in caplog.records if marker in rec.getMessage()
    )
    logged = json.loads(log_msg.split(marker, 1)[1])
    num_layers = len(cfg.model.layers)
    assert isinstance(logged["model"]["layers"], dict)
    assert list(logged["model"]["layers"].keys()) == [
        f"[0..{num_layers - 1}] ({num_layers} items)"
    ]

    with open(tmp_path / "config.json") as f:
        saved = json.load(f)
    assert isinstance(saved["model"]["layers"], list)
    assert len(saved["model"]["layers"]) == num_layers

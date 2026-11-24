# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from torchtitan.config.configurable import Configurable
from torchtitan.config.format import format_config


def _make_init_closure(scale: float) -> Callable[[float], float]:
    def _init(x: float) -> float:
        return x * scale

    return _init


def _scaled_fn(x: float, *, std: float) -> float:
    return x * std


def _expand_segment(seg: str) -> list[str]:
    if not (seg.startswith("{") and seg.endswith("}")):
        return [seg]
    expanded: list[str] = []
    for part in seg[1:-1].split(","):
        if ".." in part:
            start_s, end_s = part.split("..", 1)
            expanded.extend(str(i) for i in range(int(start_s), int(end_s) + 1))
        else:
            expanded.append(part)
    return expanded


def _expand_folded(mapping: dict[str, Any]) -> dict[str, Any]:
    """Expand every ``{...}`` index segment back into flat dotted keys."""
    expanded: dict[str, Any] = {}
    for key, val in mapping.items():
        seg_choices = [_expand_segment(seg) for seg in re.split(r"\.(?![^{}]*\})", key)]
        for combo in itertools.product(*seg_choices):
            flat_key = ".".join(combo)
            assert flat_key not in expanded, f"overlapping folded key: {flat_key}"
            expanded[flat_key] = val
    return expanded


class _DenseLinear(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        param_init: dict[str, Callable] = field(default_factory=dict)


class _QuantLinear(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        # Intentionally identical field names to _DenseLinear.Config.
        dim: int
        param_init: dict[str, Callable] = field(default_factory=dict)


class _MoEBlock(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        num_experts: int


@dataclass(frozen=True, slots=True)
class _HeadSpec:
    head_dim: int


class _Layer(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        ffn: Configurable.Config
        layer_id: int
        heads: list[_HeadSpec] = field(default_factory=list)


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


def test_format_config_is_lossless_and_folds_repeated_entries() -> None:
    """Folding shrinks repeated paths without dropping any per-layer value."""
    num_layers = 10
    layers = [
        _Layer.Config(
            layer_id=i,
            ffn=_DenseLinear.Config(
                dim=2048 if i == 3 else 4096,
                param_init={
                    "w1": _make_init_closure(1.0),
                    "w2": partial(_scaled_fn, std=0.02 / (i + 1)),
                },
            ),
            heads=[_HeadSpec(head_dim=64), _HeadSpec(head_dim=64)],
        )
        for i in range(num_layers)
    ]
    owned = _OpaqueLayout("Owned()")
    sharded = _OpaqueLayout("Shard(0)")
    cfg = _Model.Config(
        layers=layers,
        sharding_by_fqn={
            **{f"layers.{i}.wq.weight": owned for i in range(5)},
            "layers.0.wo.weight": owned,
            "layers.1.wo.weight": sharded,
            "layers.2.wo.weight": owned,
        },
    )

    formatted = format_config(cfg)
    assert " at 0x" not in formatted
    folded = json.loads(formatted)
    expanded = _expand_folded(folded)

    # Repeated layer/head/sharding entries actually folded.
    assert len(folded) < len(expanded)

    # Every per-layer varying value and single-layer override is preserved.
    for i in range(num_layers):
        assert expanded[f"layers.{i}.layer_id"] == i
        assert expanded[f"layers.{i}.ffn.dim"] == (2048 if i == 3 else 4096)
        assert f"std={0.02 / (i + 1)}" in expanded[f"layers.{i}.ffn.param_init.w2"]
        assert (
            expanded[f"layers.{i}.ffn.param_init.w1"]
            == "<function _make_init_closure.<locals>._init>"
        )
        assert expanded[f"layers.{i}.heads.0.head_dim"] == 64
        assert expanded[f"layers.{i}.heads.1.head_dim"] == 64

    # Indexed dict values remain lossless and distinct values stay separate.
    for i in range(5):
        assert expanded[f"sharding_by_fqn.layers.{i}.wq.weight"] == repr(owned)
    assert expanded["sharding_by_fqn.layers.0.wo.weight"] == repr(owned)
    assert expanded["sharding_by_fqn.layers.1.wo.weight"] == repr(sharded)
    assert expanded["sharding_by_fqn.layers.2.wo.weight"] == repr(owned)


def test_format_config_preserves_sibling_config_subclasses() -> None:
    """Sibling Config classes with identical fields must never merge together."""
    layers = [
        _Layer.Config(layer_id=0, ffn=_DenseLinear.Config(dim=1024)),
        _Layer.Config(layer_id=1, ffn=_DenseLinear.Config(dim=1024)),
        _Layer.Config(layer_id=2, ffn=_QuantLinear.Config(dim=1024)),
        _Layer.Config(layer_id=3, ffn=_MoEBlock.Config(dim=1024, num_experts=8)),
        _Layer.Config(layer_id=4, ffn=_MoEBlock.Config(dim=1024, num_experts=8)),
    ]
    cfg = _Model.Config(layers=layers)

    folded = json.loads(format_config(cfg))
    expanded = _expand_folded(folded)

    assert expanded["layers.0.ffn.__class__"] == "_DenseLinear.Config"
    assert expanded["layers.1.ffn.__class__"] == "_DenseLinear.Config"
    assert expanded["layers.2.ffn.__class__"] == "_QuantLinear.Config"
    assert expanded["layers.3.ffn.__class__"] == "_MoEBlock.Config"
    assert expanded["layers.4.ffn.__class__"] == "_MoEBlock.Config"
    assert expanded["layers.3.ffn.num_experts"] == 8
    assert "layers.2.ffn.num_experts" not in expanded


def test_trainer_maybe_log_compacts_stdout_but_saves_full_to_dict(
    tmp_path, caplog, monkeypatch
) -> None:
    """debug.print_config logs compacted JSON while save_config_file keeps full JSON."""
    import torch.distributed as dist

    from torchtitan.models.llama3 import Llama3Model
    from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel

    cfg = llama3_debugmodel()
    assert isinstance(cfg.model, Llama3Model.Config)
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
    expanded = _expand_folded(logged)
    num_layers = len(cfg.model.layers)
    assert len(logged) < len(expanded)
    assert (
        expanded["model.layers.0.__class__"] == type(cfg.model.layers[0]).__qualname__
    )
    assert (
        expanded[f"model.layers.{num_layers - 1}.__class__"]
        == type(cfg.model.layers[-1]).__qualname__
    )

    with open(tmp_path / "config.json") as f:
        saved = json.load(f)
    assert isinstance(saved["model"]["layers"], list)
    assert len(saved["model"]["layers"]) == num_layers

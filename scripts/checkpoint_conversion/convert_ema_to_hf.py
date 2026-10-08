# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpointer import EMA, ModelWrapper
from torchtitan.components.optim import EMA as EMAContainer  # noqa: N811
from torchtitan.components.optim.ema import parse_ema_key
from torchtitan.config import TORCH_DTYPE_MAP

if __package__:
    from scripts.checkpoint_conversion.utils import build_model_config_for_conversion
else:
    from utils import (  # pyrefly: ignore [missing-import]
        build_model_config_for_conversion,
    )

# CheckpointManager keeps the EMA container under the EMA state key, so DCP
# flattens its per-tensor state to "ema.state.<fqn>.ema_params.<ema_key>",
# one entry per configured decay (e.g. "decay_0p999", "half_life_0p05").
logger = logging.getLogger(__name__)

_EMA_KEY_PREFIX = f"{EMA}.state."
_EMA_STATE_KEY = "ema_params"


def _ema_state_dict(ema: EMAContainer, ema_key: str) -> dict[str, torch.Tensor]:
    """Native FQN -> tensor for one EMA copy, flattening the container's
    per-tensor state."""
    return {
        name: ema_opt.state[tensor][_EMA_STATE_KEY][ema_key]
        for ema_opt in ema.optimizers
        for group in ema_opt.param_groups
        for name, tensor in zip(group["param_names"], group["params"])
    }


def _select_ema_key(available: list[str], ema_key: str | None) -> str:
    """The EMA copy to export: ``ema_key`` if given, else the only one."""
    if ema_key is None:
        if len(available) == 1:
            return available[0]
        raise ValueError(
            f"The checkpoint holds {len(available)} EMA copies: {available}. "
            "Pass --ema_key to choose which one to export."
        )
    if ema_key not in available:
        raise ValueError(
            f"The checkpoint holds no EMA copy {ema_key!r}; available: {available}."
        )
    return ema_key


def _checkpoint_ema_copy(
    input_dir: Path, ema_key: str | None
) -> tuple[str, bool, float, set[str]]:
    """Find the EMA copy to export in the on-disk checkpoint.

    Returns ``(ema_key, is_decay_mode, decay_or_half_life_fraction, ema_fqns)``:
    the selected EMA key (``ema_key`` if given, else the only copy), whether
    it is an ``EMA.Config.decays`` entry (True) or a ``half_life_fractions``
    entry (False), that entry's value, and the FQNs the checkpoint holds the
    copy for.
    """
    metadata = dcp.FileSystemReader(input_dir).read_metadata()
    fqns_by_key: dict[str, set[str]] = {}
    for key in metadata.state_dict_metadata:
        if not key.startswith(_EMA_KEY_PREFIX):
            continue
        fqn, sep, key_suffix = key[len(_EMA_KEY_PREFIX) :].rpartition(
            f".{_EMA_STATE_KEY}."
        )
        if sep:
            fqns_by_key.setdefault(key_suffix, set()).add(fqn)
    if not fqns_by_key:
        raise ValueError(
            f"The checkpoint at {input_dir} holds no EMA weights. Either EMA was "
            "disabled for that training run, or this is a model-only export, which "
            "never carries EMA state. Use convert_to_hf.py for the trained weights."
        )
    ema_key = _select_ema_key(sorted(fqns_by_key), ema_key)

    is_decay_mode, decay_or_half_life_fraction = parse_ema_key(ema_key)
    return ema_key, is_decay_mode, decay_or_half_life_fraction, fqns_by_key[ema_key]


@torch.inference_mode()
def convert_ema_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
    *,
    ema_key=None,
):
    # load model and model args so that we can get the state dict shape
    model_config = build_model_config_for_conversion(model_name, model_flavor)

    with torch.device("cpu"):
        model = model_config.build()

    adapter_cls = type(model).state_dict_adapter_cls
    assert adapter_cls is not None, (
        "trying to convert checkpoint from DCP to HF safetensors format, "
        "but the model has no state dict adapter."
    )
    sd_adapter = adapter_cls(model_config, hf_assets_path)

    (
        ema_key,
        is_decay_mode,
        decay_or_half_life_fraction,
        ema_fqns,
    ) = _checkpoint_ema_copy(input_dir, ema_key)
    logger.info("Exporting EMA copy %r", ema_key)

    # Rebuild the container the run trained with. Parameters are always tracked,
    # so only buffer EMA has to be recovered from the checkpoint: naming those
    # buffers exactly is what keeps one that was EMA-ed during training from
    # silently exporting its live value instead.
    param_fqns = {name for name, _ in model.named_parameters()}
    buffer_fqns = ema_fqns - param_fqns
    # EMA tracks the parameters that required grad during training, so a run
    # with frozen parameters has no entry for them. Track exactly what the
    # checkpoint holds: without this the rebuilt container asks DCP for keys
    # that were never saved and the load fails with a bare missing-key error.
    # Those parameters still reach the export, from the trained weights.
    untracked_params = sorted(param_fqns - ema_fqns)
    if untracked_params:
        logger.info(
            "%d parameter(s) have no EMA state and will be exported from the "
            "trained weights, e.g. %s",
            len(untracked_params),
            untracked_params[:3],
        )
        for name, param in model.named_parameters():
            if name not in ema_fqns:
                param.requires_grad_(False)
    # Track only the selected copy, so DCP is asked for just its keys.
    ema = EMAContainer.Config(
        decays=[decay_or_half_life_fraction] if is_decay_mode else [],
        half_life_fractions=[] if is_decay_mode else [decay_or_half_life_fraction],
        buffer_patterns=[f"^{re.escape(fqn)}$" for fqn in sorted(buffer_fqns)],
    ).build(model_parts=[model])

    uncovered = ema_fqns - set(_ema_state_dict(ema, ema_key))
    if uncovered:
        raise ValueError(
            f"Model '{model_name}/{model_flavor}' has no parameter or buffer for "
            f"{len(uncovered)} EMA-tracked FQN(s) in the checkpoint, e.g. "
            f"{sorted(uncovered)[:5]}. Exporting would silently drop them; check "
            "that model_name/model_flavor match the training run."
        )

    # Load the trained weights alongside the EMA ones. EMA covers parameters
    # plus the buffers named above, so the remaining tensors (unaveraged
    # buffers, frozen parameters) still come from the checkpoint, keeping the
    # export complete and loadable.
    state_dict = ModelWrapper(model)._get_state_dict()
    dcp.load(
        {EMA: ema, **state_dict},
        checkpoint_id=input_dir,
    )

    # EMA is keyed by named_parameters() FQNs. Those coincide with the model
    # state dict's keys, but a module that split a fused parameter in a
    # state-dict hook would break that, and sd_adapter.to_hf() drops keys it
    # does not recognise without a word -- which would export the trained
    # weights instead of the average. Fail loudly rather than silently.
    ema_state = _ema_state_dict(ema, ema_key)
    unmatched = sorted(set(ema_state) - set(state_dict))
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} EMA tensor(s) have no matching key in the model "
            f"state dict, e.g. {unmatched[:3]}. The EMA average cannot be "
            "applied to them, and exporting would silently emit the trained "
            "weights instead."
        )
    state_dict.update(ema_state)

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the EMA weights of a DCP checkpoint to HF format."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    parser.add_argument(
        "--ema_key",
        type=str,
        default=None,
        help="EMA copy to export, e.g. half_life_0p05 or decay_0p999 "
        "(default: the checkpoint's only EMA copy)",
    )
    args = parser.parse_args()

    convert_ema_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        ema_key=args.ema_key,
    )

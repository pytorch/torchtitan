# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe.moe import GroupedExperts, TokenChoiceTopKRouter
from torchtitan.protocols.model_converter import ConverterCheckpointHooks
from torchtitan.tools.logging import logger

# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}

# Cache for dynamically created expert LoRA classes
_expert_lora_class_cache: dict[type, type] = {}


def apply_lora(linear: nn.Linear, rank: int, alpha: float) -> nn.Linear:
    parent_cls = type(linear)
    assert issubclass(
        parent_cls, nn.Linear
    ), f"parent_cls must be a subclass of nn.Linear, got {parent_cls}"

    if parent_cls not in _lora_class_cache:

        class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("LoRALinear should not be instantiated directly.")

            @classmethod
            def from_linear(
                cls, linear: nn.Linear, rank: int, alpha: float
            ) -> "LoRALinear":
                linear.__class__ = cls
                linear._init_lora(rank, alpha)  # type: ignore[attr-defined]
                return linear  # type: ignore[return-value]

            def _init_lora(
                self,
                rank: int,
                alpha: float,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,
            ) -> None:
                self._lora_scaling = alpha / rank
                device = device if device is not None else self.weight.device
                dtype = dtype if dtype is not None else self.weight.dtype
                self.lora_a = (
                    Linear.Config(bias=False)
                    .build(in_features=self.in_features, out_features=rank)
                    .to(device=device, dtype=dtype)
                )
                self.lora_b = (
                    Linear.Config(bias=False)
                    .build(in_features=rank, out_features=self.out_features)
                    .to(device=device, dtype=dtype)
                )

            def init_weights(self, **kwargs) -> None:
                super().init_weights(**kwargs)  # pyrefly: ignore [not-callable]
                nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
                # Standard LoRA uses zeros for lora_b so the initial
                # contribution is exactly zero.  When the adapter has QAT
                # fake quantization (e.g. float8), zeros cause NaN because
                # dynamic scaling divides by max(abs(0)) = 0.  Use std=0.02
                # so values are representable in float8; the initial LoRA
                # contribution is still negligible.
                try:
                    from torchao.quantization.qat.linear import FakeQuantizedLinear

                    is_fake_quantized = isinstance(self.lora_b, FakeQuantizedLinear)
                except ImportError:
                    is_fake_quantized = False

                if is_fake_quantized:
                    nn.init.normal_(self.lora_b.weight, mean=0.0, std=0.02)
                else:
                    nn.init.zeros_(self.lora_b.weight)

            def named_parameters(self, *args, **kwargs):
                # Force recurse=False so ColwiseParallel._partition_linear_fn
                # only sees direct params (weight, bias), not dotted names like
                # "lora_a.weight". The adapter submodules are visited separately
                # by distribute_module and made Replicate automatically.
                kwargs["recurse"] = False
                yield from super().named_parameters(*args, **kwargs)

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                base_out = super().forward(input)
                lora_out = self.lora_b(self.lora_a(input))
                return base_out + self._lora_scaling * lora_out

        LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
        LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
        _lora_class_cache[parent_cls] = LoRALinear

    # pyrefly: ignore [missing-attribute]
    return _lora_class_cache[parent_cls].from_linear(linear, rank, alpha)


def _compute_expert_lora_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    target_weight: nn.Parameter,
) -> torch.Tensor:
    """Compute the LoRA weight delta for expert weights.

    Args:
        lora_a: (E, in, r) — projects input dim to rank.
        lora_b: (E, r, out) — projects rank to output dim.
        scaling: alpha / rank.
        target_weight: The base weight parameter to match DTensor placements.

    Returns:
        delta matching target_weight's shape and placements.
        Math: delta = scaling * B^T @ A^T  →  shape (E, out, in).
    """
    from torch.distributed.tensor import distribute_tensor, DTensor

    delta = scaling * torch.bmm(lora_b.transpose(-2, -1), lora_a.transpose(-2, -1))
    # When the base weight is a DTensor (TP/EP sharded), distribute the delta
    # to match its placements so the in-place add_/sub_ operates on matching shapes.
    if isinstance(target_weight, DTensor) and not isinstance(delta, DTensor):
        delta = distribute_tensor(
            delta, target_weight.device_mesh, target_weight.placements
        )
    return delta


def apply_expert_lora(
    experts: GroupedExperts, rank: int, alpha: float
) -> GroupedExperts:
    """Apply LoRA adapters to a GroupedExperts module via class swapping.

    LoRA parameters are registered as direct parameters on the module. EP partition
    functions that use ``named_parameters(recurse=False)`` with ``Shard(0)`` will
    correctly shard them on the expert dimension. TP/ETP partition functions only
    touch w1/w2/w3 by name and leave LoRA parameters unsharded.

    Forward uses merge-per-forward: LoRA deltas are merged into base weights before
    calling the base forward, then unmerged after. This reuses the base
    GroupedExperts.forward without duplicating its DTensor/EP/padding logic.
    """
    parent_cls = type(experts)
    assert issubclass(
        parent_cls, GroupedExperts
    ), f"parent_cls must be a subclass of GroupedExperts, got {parent_cls}"

    if parent_cls not in _expert_lora_class_cache:

        class LoRAGroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "LoRAGroupedExperts should not be instantiated directly."
                )

            @classmethod
            def from_experts(
                cls, experts: GroupedExperts, rank: int, alpha: float
            ) -> "LoRAGroupedExperts":
                experts.__class__ = cls
                experts._init_expert_lora(rank, alpha)  # type: ignore[attr-defined]
                return experts  # type: ignore[return-value]

            def _init_expert_lora(self, rank: int, alpha: float) -> None:
                self._lora_scaling = alpha / rank
                num_experts = self.num_experts
                # w1: (E, hidden_dim, dim) -> A1: (E, dim, r), B1: (E, r, hidden_dim)
                dim_w1_in = self.w1.shape[2]  # dim
                dim_w1_out = self.w1.shape[1]  # hidden_dim
                # w2: (E, dim, hidden_dim) -> A2: (E, hidden_dim, r), B2: (E, r, dim)
                dim_w2_in = self.w2.shape[2]  # hidden_dim
                dim_w2_out = self.w2.shape[1]  # dim
                # w3: (E, hidden_dim, dim) -> A3: (E, dim, r), B3: (E, r, hidden_dim)
                dim_w3_in = self.w3.shape[2]  # dim
                dim_w3_out = self.w3.shape[1]  # hidden_dim

                device = self.w1.device
                dtype = self.w1.dtype

                self.lora_a_w1 = nn.Parameter(
                    torch.empty(
                        num_experts, dim_w1_in, rank, device=device, dtype=dtype
                    )
                )
                self.lora_b_w1 = nn.Parameter(
                    torch.empty(
                        num_experts, rank, dim_w1_out, device=device, dtype=dtype
                    )
                )
                self.lora_a_w2 = nn.Parameter(
                    torch.empty(
                        num_experts, dim_w2_in, rank, device=device, dtype=dtype
                    )
                )
                self.lora_b_w2 = nn.Parameter(
                    torch.empty(
                        num_experts, rank, dim_w2_out, device=device, dtype=dtype
                    )
                )
                self.lora_a_w3 = nn.Parameter(
                    torch.empty(
                        num_experts, dim_w3_in, rank, device=device, dtype=dtype
                    )
                )
                self.lora_b_w3 = nn.Parameter(
                    torch.empty(
                        num_experts, rank, dim_w3_out, device=device, dtype=dtype
                    )
                )

            def init_weights(self, init_std: float) -> None:
                super().init_weights(init_std)
                for name in ("lora_a_w1", "lora_a_w2", "lora_a_w3"):
                    nn.init.kaiming_uniform_(getattr(self, name), a=math.sqrt(5))
                for name in ("lora_b_w1", "lora_b_w2", "lora_b_w3"):
                    nn.init.zeros_(getattr(self, name))

            def forward(
                self,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
            ) -> torch.Tensor:
                # Merge LoRA deltas into base weights, run base forward, unmerge.
                # This reuses all base GroupedExperts logic (DTensor, EP, padding).
                deltas = {}
                for w_name, a_name, b_name in (
                    ("w1", "lora_a_w1", "lora_b_w1"),
                    ("w2", "lora_a_w2", "lora_b_w2"),
                    ("w3", "lora_a_w3", "lora_b_w3"),
                ):
                    lora_a = getattr(self, a_name)
                    lora_b = getattr(self, b_name)
                    w = getattr(self, w_name)
                    delta = _compute_expert_lora_delta(
                        lora_a, lora_b, self._lora_scaling, w
                    )
                    w.data.add_(delta)
                    deltas[w_name] = delta

                try:
                    return super().forward(x, num_tokens_per_expert)
                finally:
                    # Unmerge: subtract deltas to restore original weights
                    for w_name, delta in deltas.items():
                        getattr(self, w_name).data.sub_(delta)

        LoRAGroupedExperts.__name__ = f"LoRA{parent_cls.__name__}"
        LoRAGroupedExperts.__qualname__ = f"LoRA{parent_cls.__name__}"
        _expert_lora_class_cache[parent_cls] = LoRAGroupedExperts

    # pyrefly: ignore [missing-attribute]
    return _expert_lora_class_cache[parent_cls].from_experts(experts, rank, alpha)


class LoRAConverter(Configurable):
    """Apply LoRA adapters to Linear layers and GroupedExperts in a model.

    When ``target_modules`` is None (default), every ``nn.Linear`` (except
    router gates) and ``GroupedExperts`` receives a LoRA adapter.  When
    specified, only modules whose attribute name matches one of the entries
    are converted (e.g. ``["wq", "wv"]`` targets the query and value
    projections).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rank: int = 8
        """Rank of the LoRA matrices (lora_a: in_features x rank, lora_b: rank x out_features)."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        target_modules: list[str] | None = None
        """Module attribute names to apply LoRA to (e.g. ["wq", "wv"]).
        None means all nn.Linear layers."""

        merge_adapter: bool = False
        """When True, adapters are folded into base weights
        (base + alpha/rank * B @ A) and the checkpoint contains no LoRA keys.
        When False (default), adapter weights are saved separately.
        Use checkpoint.last_save_in_hf=True to save in PEFT format."""

    def __init__(self, config: Config, **kwargs):
        self.rank = config.rank
        self.alpha = config.alpha
        self.target_modules = (
            set(config.target_modules) if config.target_modules else set()
        )
        self.merge_adapter = config.merge_adapter
        # Set by ModelConvertersContainer when QAT is also active.
        self.qat_scheme: str | None = None
        self.qat_group_size: int = 128
        if self.target_modules:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
                f"target_modules={sorted(self.target_modules)}"
            )
        else:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha} "
                f"(all Linear layers)"
            )

    @staticmethod
    def _is_lora_key(key: str) -> bool:
        """Check if a state dict key belongs to a LoRA adapter."""
        return ".lora_a." in key or ".lora_b." in key

    def _save_peft(
        self,
        state_dict: dict[str, Any],
        checkpoint_dir: str,
        hooks: "ConverterCheckpointHooks",
    ) -> None:
        """Save adapter weights in PEFT format.

        Writes ``adapter_model.safetensors`` and ``adapter_config.json``
        into the checkpoint directory. Only rank 0 performs file I/O.
        Keys are remapped from torchtitan to HF PEFT naming.
        """
        from safetensors.torch import save_file

        # Collect full tensors from DTensors on CPU
        cpu_states = {}
        for k, v in state_dict.items():
            if hasattr(v, "full_tensor"):
                cpu_states[k] = v.full_tensor().cpu()
            else:
                cpu_states[k] = v.cpu() if isinstance(v, torch.Tensor) else v

        # Remap keys to HF PEFT naming
        if hooks.from_hf_map is not None:
            hf_states = remap_lora_keys_to_hf(cpu_states, hooks.from_hf_map)
        else:
            logger.warning(
                "No from_hf_map available; saving PEFT with torchtitan keys."
            )
            hf_states = cpu_states

        if dist.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

            safetensors_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
            save_file(hf_states, safetensors_path)
            logger.info(f"Saved PEFT adapter weights to {safetensors_path}")

            # Write adapter_config.json with HF module names
            target_modules = sorted(
                {
                    k.rsplit(".lora_A.", 1)[0].rsplit(".", 1)[-1]
                    if ".lora_A." in k
                    else k.rsplit(".lora_B.", 1)[0].rsplit(".", 1)[-1]
                    for k in hf_states
                }
            )
            config_dict = {
                "peft_type": "LORA",
                "r": self.rank,
                "lora_alpha": self.alpha,
                "target_modules": target_modules,
            }
            config_path = os.path.join(checkpoint_dir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved PEFT adapter config to {config_path}")

        # Ensure all ranks wait for rank 0 to finish writing
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _lora_module_names(state_dict: dict[str, Any]) -> set[str]:
        """Extract leaf module names that have LoRA keys in a state dict."""
        names = set()
        for k in state_dict:
            for marker in (".lora_a.", ".lora_b."):
                if marker in k:
                    names.add(k.rsplit(marker, 1)[0].rsplit(".", 1)[-1])
                    break
        return names

    def _load_peft(
        self,
        path: str,
        model_parts: list[nn.Module],
        hooks: "ConverterCheckpointHooks",
    ) -> None:
        """Load adapter weights from a PEFT directory.

        Loads ``adapter_model.safetensors``, remaps keys from HF PEFT naming
        to torchtitan naming, and broadcasts from rank 0.

        Warns when the loaded adapter targets different modules than
        the current model's LoRA modules.
        """
        import functools

        from safetensors.torch import load_file

        safetensors_path = os.path.join(path, "adapter_model.safetensors")
        adapter_sd = load_file(safetensors_path)
        if hooks.from_hf_map is not None:
            adapter_sd = remap_lora_keys_from_hf(adapter_sd, hooks.from_hf_map)

        # Warn on mismatch between checkpoint and model LoRA targets
        ckpt_targets = self._lora_module_names(adapter_sd)
        model_targets = {
            name.rsplit(".", 1)[-1]
            for part in model_parts
            for name, mod in part.named_modules()
            if hasattr(mod, "lora_a")
        }
        only_in_ckpt = ckpt_targets - model_targets
        only_in_model = model_targets - ckpt_targets
        if only_in_ckpt:
            logger.warning(
                f"Loaded adapter has LoRA for {sorted(only_in_ckpt)} but the "
                f"current model does not (will be ignored)."
            )
        if only_in_model:
            logger.warning(
                f"Current model has LoRA for {sorted(only_in_model)} but the "
                f"loaded adapter does not (will start from init)."
            )

        func = functools.partial(
            set_model_state_dict,
            model_state_dict=adapter_sd,
            options=StateDictOptions(
                strict=False,
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        list(map(func, model_parts))

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

        # QATConverter must run before LoRAConverter: quantize_() replaces
        # nn.Linear via from_linear() which discards lora_a/lora_b.
        # With QAT first, LoRA inherits from FakeQuantizedLinear instead.
        if self.qat_scheme is not None:
            self._apply_adapter_qat(model, self.qat_scheme, self.qat_group_size)

        if self.merge_adapter:
            hooks = ConverterCheckpointHooks(key_filter=self._is_lora_key)
        else:
            hooks = ConverterCheckpointHooks(
                key_filter=self._is_lora_key,
                save_last_fn=self._save_peft,
                load_additional_fn=self._load_peft,
            )
        model._converter_hooks = hooks  # type: ignore[attr-defined]

    @staticmethod
    def _qat_compatible(rank: int, scheme: str, group_size: int) -> bool:
        """Check if QAT scheme is compatible with the given in_features size.

        Tests prepare + forward on a temporary linear, since some schemes
        (e.g. int4_weight_only) only fail at forward time.
        """
        import torch

        from torchtitan.components.quantization.qat import apply_qat_prepare

        try:
            probe = nn.ModuleDict({"l": nn.Linear(rank, rank, bias=False).to("cuda")})
            apply_qat_prepare(probe, scheme, group_size)
            probe.l(torch.randn(1, rank, device="cuda", dtype=probe.l.weight.dtype))
            return True
        except (ValueError, RuntimeError, AssertionError):
            return False

    def _apply_adapter_qat(
        self, model: nn.Module, scheme: str, group_size: int
    ) -> None:
        """Apply QAT to LoRA adapter linears.

        Always applies to lora_a. For lora_b, skips if rank is incompatible
        with the scheme's block/group size (Unsloth approach).
        """
        from torchtitan.components.quantization.qat import apply_qat_prepare

        def _is_lora(suffix: str):
            def fn(mod: nn.Module, fqn: str) -> bool:
                return isinstance(mod, nn.Linear) and fqn.endswith(suffix)

            return fn

        apply_qat_prepare(model, scheme, group_size, filter_fn=_is_lora(".lora_a"))

        skip_lora_b = not self._qat_compatible(self.rank, scheme, group_size)
        if not skip_lora_b:
            apply_qat_prepare(model, scheme, group_size, filter_fn=_is_lora(".lora_b"))

        logger.info(
            f"Applied adapter QAT (scheme={scheme}, group_size={group_size}, "
            f"lora_b={'skipped' if skip_lora_b else 'applied'})"
        )

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        # Collect router gate linears so we can skip them — routing scores
        # must stay frozen to preserve expert load balancing.
        router_gate_ids: set[int] = set()
        for child in module.modules():
            if isinstance(child, TokenChoiceTopKRouter):
                router_gate_ids.add(id(child.gate))

        matched = set()
        for _, parent in list(module.named_modules()):
            for attr_name, child in list(parent.named_children()):
                if self.target_modules and attr_name not in self.target_modules:
                    continue
                if isinstance(child, nn.Linear) and id(child) not in router_gate_ids:
                    apply_lora(child, self.rank, self.alpha)
                    matched.add(attr_name)
                elif isinstance(child, GroupedExperts):
                    apply_expert_lora(child, self.rank, self.alpha)
                    matched.add(attr_name)
        unmatched = self.target_modules - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"nn.Linear or GroupedExperts in the model. "
                f"Check module attribute names."
            )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass

    def finalize(self, model: nn.Module) -> None:
        """Merge LoRA adapters into base weights at end of training.

        Only runs when merge_adapter=True. After merging, adapter modules
        are removed and the linear class is restored to its parent class.
        Hooks are cleaned up so state_dict() returns all keys.
        """
        if not self.merge_adapter:
            return

        scaling = self.alpha / self.rank
        for name, mod in list(model.named_modules()):
            if not (hasattr(mod, "lora_a") and hasattr(mod, "lora_b")):
                continue
            assert isinstance(mod, nn.Linear)
            lora_a = mod.lora_a
            lora_b = mod.lora_b
            assert isinstance(lora_a, nn.Linear)
            assert isinstance(lora_b, nn.Linear)
            with torch.no_grad():
                mod.weight.add_(scaling * (lora_b.weight @ lora_a.weight))
            del mod.lora_a, mod.lora_b
            if hasattr(mod, "_lora_scaling"):
                del mod._lora_scaling
            # Restore the parent class (e.g. FakeQuantizedLinear or nn.Linear)
            parent_cls = type(mod).__mro__[1]
            if parent_cls is not nn.Module and issubclass(parent_cls, nn.Linear):
                mod.__class__ = parent_cls

        # Clean up hooks — model is now a plain base model
        if hasattr(model, "_converter_hooks"):
            del model._converter_hooks

        logger.info("LoRA adapters merged into base weights")


def remap_lora_keys_to_hf(
    adapter_sd: dict[str, Any], from_hf_map: dict[str, str | None]
) -> dict[str, Any]:
    """Remap torchtitan LoRA keys to HF PEFT naming for saving.

    Converts keys like ``layers.0.attention.wq.lora_a.weight`` to
    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``.
    """
    to_hf_map = {v: k for k, v in from_hf_map.items() if v is not None}
    remapped = {}
    for key, value in adapter_sd.items():
        lora_suffix = None
        base_key = key
        for suffix in (".lora_a.weight", ".lora_b.weight"):
            if key.endswith(suffix):
                lora_suffix = suffix
                base_key = key[: -len(suffix)] + ".weight"
                break
        if lora_suffix is None:
            remapped[key] = value
            continue

        abstract_base = re.sub(r"(?<=\.)\d+(?=\.)", "{}", base_key, count=1)
        layer_match = re.search(r"(?<=\.)\d+(?=\.)", base_key)
        layer_num = layer_match.group(0) if layer_match else None

        if abstract_base in to_hf_map:
            hf_abstract = to_hf_map[abstract_base]
            hf_base = hf_abstract.format(layer_num) if layer_num else hf_abstract
            hf_base_no_weight = hf_base.rsplit(".weight", 1)[0]
            peft_suffix = lora_suffix.replace(".lora_a.", ".lora_A.").replace(
                ".lora_b.", ".lora_B."
            )
            hf_key = f"base_model.model.{hf_base_no_weight}{peft_suffix}"
        else:
            hf_key = f"base_model.model.{key}"
            hf_key = hf_key.replace(".lora_a.", ".lora_A.").replace(
                ".lora_b.", ".lora_B."
            )
            logger.warning(
                f"No HF mapping for base key '{abstract_base}', using '{hf_key}'"
            )
        remapped[hf_key] = value
    return remapped


def remap_lora_keys_from_hf(
    adapter_sd: dict[str, Any], from_hf_map: dict[str, str | None]
) -> dict[str, Any]:
    """Remap HF PEFT keys to torchtitan naming for loading.

    Converts keys like ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    to ``layers.0.attention.wq.lora_a.weight``.
    """
    remapped = {}
    for key, value in adapter_sd.items():
        stripped = key
        if stripped.startswith("base_model.model."):
            stripped = stripped[len("base_model.model.") :]
        stripped = stripped.replace(".lora_A.", ".lora_a.").replace(
            ".lora_B.", ".lora_b."
        )

        lora_suffix = None
        hf_base_key = stripped
        for suffix in (".lora_a.weight", ".lora_b.weight"):
            if stripped.endswith(suffix):
                lora_suffix = suffix
                hf_base_key = stripped[: -len(suffix)] + ".weight"
                break
        if lora_suffix is None:
            remapped[stripped] = value
            continue

        abstract_hf = re.sub(r"(?<=\.)\d+(?=\.)", "{}", hf_base_key, count=1)
        layer_match = re.search(r"(?<=\.)\d+(?=\.)", hf_base_key)
        layer_num = layer_match.group(0) if layer_match else None

        tt_abstract = from_hf_map.get(abstract_hf)
        if tt_abstract is not None:
            tt_base = tt_abstract.format(layer_num) if layer_num else tt_abstract
            tt_base_no_weight = tt_base.rsplit(".weight", 1)[0]
            tt_key = f"{tt_base_no_weight}{lora_suffix}"
        else:
            tt_key = stripped
            logger.warning(
                f"No torchtitan mapping for HF key '{abstract_hf}', using '{tt_key}'"
            )
        remapped[tt_key] = value
    return remapped

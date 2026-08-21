# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 experiment: KDA + MLA + MoE backbone with Block Attention
Residuals (AttnRes, arXiv:2603.15031), the architecture family Kimi K3
confirmed in production.

Flavors follow ``kimi_linear_<size>_<variant>`` (size from the AttnRes
tech-report Table 2 scaling-law sweep plus the 48B-A3B layout; variant in
{baseline, block_attn_res, full_attn_res}). Trainer-level configuration
lives in :mod:`.config_registry`; architecture-side builders in
:mod:`.model_configs`.

The cross-stage pipeline-parallel cache adapter (``pipeline_adapter.py``)
is private to this experiment by design -- see the AttnRes RFC history.
"""

from dataclasses import dataclass
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger

# fla-core (triton) is required by the KDA path; guard so environments
# without it (e.g. CPU-only dev boxes) can still import the package and
# fail with a clear error only when a Kimi flavor is requested.
try:
    from torchtitan.models.kimi_k3.attn_res_model import (
        KimiAttnResDecoderLayer,
        KimiK3AttnResModel,
    )
    from torchtitan.models.kimi_k3.model import (
        KimiDecoderLayer,
        KimiDeltaAttention,
        KimiK3Config,
        KimiK3Model,
        KimiK3Spec,
        KimiMLAAttention,
        KimiMLP,
        KimiMoE,
    )
    from torchtitan.models.kimi_k3.model_configs import (
        attn_res_block_size,
        build_kimi_linear_config,
        flavor_names,
        resolve_num_blocks,
        SCALING_LAW_TABLE,
    )
    from torchtitan.models.kimi_k3.parallelize import parallelize_kimi_k3
    from torchtitan.models.kimi_k3.pipeline_adapter import (
        pipeline_kimi_k3_with_cache_adapter,
    )

    _KIMI_IMPORT_ERROR: ImportError | None = None
except ImportError as _err:
    _KIMI_IMPORT_ERROR = _err

__all__ = [
    # Imported in the guarded block above for re-export; listed here so that is
    # deliberate rather than an unused import.
    "SCALING_LAW_TABLE",
    "KimiAttnResDecoderLayer",
    "KimiDecoderLayer",
    "KimiDeltaAttention",
    "KimiK3AttnResModel",
    "KimiK3Config",
    "KimiK3Model",
    "KimiK3Spec",
    "KimiMLAAttention",
    "KimiMLP",
    "KimiMoE",
    "build_kimi_linear_config",
    "flavor_names",
    "model_registry",
    "attn_res_block_size",
    "resolve_num_blocks",
]


@dataclass(frozen=True)
class _GraftSuffix:
    """One post-train graft suffix and the spec flags it implies.

    A table rather than a chain of endswith/elif (finding 36). The ordering rule that
    made the chain work -- try ``_gated_lora`` before ``_gated``, or the longer name
    decomposes as the shorter one plus a bogus size -- is now enforced by sorting on
    length instead of by the order somebody wrote the branches in.
    """

    suffix: str
    gated: bool = False
    lora_rank: int | None = None


_GRAFT_SUFFIXES: tuple[_GraftSuffix, ...] = (
    _GraftSuffix("_gated_lora", gated=True, lora_rank=16),
    _GraftSuffix("_gated", gated=True),
)


@dataclass(frozen=True)
class _GraftDecomposition:
    base_flavor: str
    gated: bool
    lora_rank: int | None


def _decompose_graft(flavor: str) -> _GraftDecomposition:
    """Split a flavor into its base name and the graft flags its suffix implies."""
    for entry in sorted(_GRAFT_SUFFIXES, key=lambda e: -len(e.suffix)):
        if flavor.endswith(entry.suffix):
            return _GraftDecomposition(
                flavor[: -len(entry.suffix)], entry.gated, entry.lora_rank
            )
    return _GraftDecomposition(flavor, False, None)


def _parse_flavor(flavor: str) -> tuple[str, str]:
    """Parse ``kimi_k3_<size>_<variant>`` -> (size, variant).

    Both prefixes are accepted. ``kimi_k3_`` is this model's own naming;
    ``kimi_linear_`` is kept for the sizes that ARE Kimi Linear -- the paper's
    Table 2 scaling-law rows (194m..528m) and the released 48B -- where
    renaming would misattribute a real published model.
    """
    for prefix in ("kimi_k3_", "kimi_linear_"):
        if flavor.startswith(prefix):
            rest = flavor[len(prefix) :]
            break
    else:
        raise ValueError(
            f"Unknown flavor '{flavor}'. Kimi K3 flavors follow "
            "'kimi_k3_<size>_<variant>'; see flavor_names()."
        )
    for variant in ("baseline", "block_attn_res", "full_attn_res"):
        suffix = f"_{variant}"
        if rest.endswith(suffix):
            size = rest[: -len(suffix)]
            return size, variant
    raise ValueError(f"Unknown flavor '{flavor}'.")


def model_registry(flavor: str, attn_backend: str | None = None) -> ModelSpec:
    """Return a :class:`ModelSpec` for a ``kimi_linear_<size>_<variant>``
    flavor. The ``baseline`` variant disables AttnRes (plain backbone);
    the cache-adapter ``pipelining_fn`` is always wired and passes
    through untouched for baseline / pp=1 runs."""
    if _KIMI_IMPORT_ERROR is not None:
        raise ImportError(
            "Kimi K3 flavors require fla-core (KDA kernels)."
        ) from _KIMI_IMPORT_ERROR
    # attn_backend is accepted for registry-interface compatibility
    # (veRL's torchtitan engine passes it): KDA runs on fla kernels and
    # MLA on SDPA here, so backend selection does not apply yet.
    if attn_backend is not None:
        logger.warning(
            "kimi_k3.model_registry ignores attn_backend=%r (KDA=fla, "
            "MLA=SDPA are fixed in this implementation).",
            attn_backend,
        )
    graft = _decompose_graft(flavor)
    gated, lora_rank = graft.gated, graft.lora_rank
    size, variant = _parse_flavor(graft.base_flavor)
    kimi_config = build_kimi_linear_config(size)
    num_blocks = resolve_num_blocks(size, variant)
    spec_config = KimiK3Spec(
        kimi_config=kimi_config,
        num_blocks=num_blocks,
        attn_res_block_size=(
            attn_res_block_size(size) if variant == "block_attn_res" else None
        ),
        attn_res_gated=gated,
        lora_rank=lora_rank,
    )
    from torchtitan.models.kimi_k3.state_dict_adapter import KimiLinearStateDictAdapter

    return ModelSpec(
        name="kimi_linear",
        flavor=flavor,
        model=spec_config,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_kimi_k3_with_cache_adapter,
        post_optimizer_build_fn=None,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )


def _model_registry_accepts(flavor: str) -> bool:
    """True when :func:`model_registry` can build this flavor's ModelSpec."""
    try:
        model_registry(flavor)
    except (ValueError, KeyError, ImportError):
        # Only the answers that mean "this name is not one of ours". A bare
        # `except Exception` here is how 37 flavors went missing once: any bug inside
        # model_registry reported as "not a flavor" and the name silently vanished from
        # discovery instead of failing.
        return False
    return True


def _discovered_flavor_names() -> list[str]:
    """Every flavor actually registered, discovered rather than enumerated.

    ``flavor_names()`` builds a product over the scaling-law table, so it lists
    only ``kimi_linear_{size}_{baseline,block_attn_res,full_attn_res}`` and
    silently omits everything hand-registered in ``config_registry`` -- the K3
    flavors, the QAT/QLoRA/KCP/quantile-balancing variants. A consumer that
    discovers flavors from this dict then cannot see them, which is how veRL's
    engine failed to resolve k3mini.

    Discovering from the registry module means adding a flavor function is
    enough; there is no second list to keep in sync. Same failure class as the
    stale init map, and the same fix.
    """
    from torchtitan.models.kimi_k3 import config_registry

    out = []
    for name, obj in vars(config_registry).items():
        # BOTH prefixes. The rename kimi_linear -> kimi_k3 left this filter
        # matching only the old one, so the 37 flavors registered under the new
        # name became invisible to discovery -- which is the exact failure this
        # function's docstring was written to fix, reintroduced by the rename.
        if not (name.startswith(("kimi_linear_", "kimi_k3_")) and callable(obj)):
            continue
        # config_registry holds Trainer.Config factories, which are a SUPERSET
        # of model flavors: some (e.g. the _n4 AttnRes-block variants) exist only
        # as trainer configs and model_registry cannot parse them. veRL calls
        # model_registry on every name it discovers here, so listing one it
        # cannot build turns flavor resolution into a hard error for everyone.
        if _model_registry_accepts(name):
            out.append(name)
    return sorted(out)


# Flavor-name dict for registry-discovery consumers (veRL's torchtitan
# engine looks for a module-level ``*_configs`` dict and uses its KEYS
# with ``model_registry``). Values are unused.
def _flavor_config_dict() -> dict[str, None]:
    """Names for registry discovery, or empty when fla-core is absent.

    Discovery reads this as a module-level dict, so it stays one rather than
    becoming a lazy attribute. But building it walks config_registry ->
    model_configs -> model, and model imports fla-core at module scope. Without
    the try, importing this package at all fails on a machine without fla --
    which turns the pointed "requires fla-core" message from something raised
    when a model is built into something raised when the package is read.
    """
    try:
        return {name: None for name in _discovered_flavor_names()}
    except ImportError as err:
        logger.warning(
            "kimi_k3 flavor discovery unavailable (%s); the flavor list is empty. "
            "Building any kimi_k3 model still raises with instructions.",
            err,
        )
        return {}


kimi_k3_configs: dict[str, None] = _flavor_config_dict()
# The pre-rename name, same object. Discovery takes the first module-level
# ``*_configs`` dict it finds, so both spellings resolve identically.
kimi_linear_configs = kimi_k3_configs

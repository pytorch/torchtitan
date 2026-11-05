# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym kernels with placeholders when the package or a backend is missing."""

# TODO: Revisit Attention Gym imports and dependency handling to remove this
# placeholder workaround.

from importlib import import_module
from types import ModuleType
from typing import Any, NoReturn, TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from attn_gym.linear import (
        chunk_gdn,
        KernelOptions,
        paged_chunk_gdn,
        recurrent_gdn,
        recurrent_gdn_decode,
    )
    from attn_gym.linear._delta_rule.gate import _FusedGate, gate_transform
    from attn_gym.linear.context_parallel import (
        _ContextParallelChunk,
        context_parallel_conv_history,
        ContextParallelRouting,
    )
    from attn_gym.linear.gdn.context_parallel import context_parallel_gdn
    from attn_gym.linear.gdn.impl.cudnn import ChunkGdnCudnnPacked
    from attn_gym.linear.gdn.ops import _ChunkGDN
    from attn_gym.linear.kda import (
        bound_gate,
        chunk_kda,
        paged_chunk_kda,
        recurrent_kda_decode,
    )
    from attn_gym.linear.kda.context_parallel import context_parallel_kda
    from attn_gym.linear.kda.fwd.triton.l2norm_fwd import _L2Norm, l2norm
    from attn_gym.linear.kda.impl.cudnn import ChunkKdaCudnn
    from attn_gym.linear.kda.impl.fused import _ChunkKDA
    from attn_gym.linear.kda.masking import _MaskRows
    from attn_gym.linear.short_conv import (
        causal_conv1d,
        causal_conv1d_decode,
        paged_causal_conv1d,
    )
    from attn_gym.linear.short_conv.cute import _ConfiguredShortConv, _ShortConv
    from attn_gym.sparse import lightning_indexer
    from attn_gym.sparse.gather_attn import AuxRequest, gather_attn

__all__ = [
    "AuxRequest",
    "_ChunkGDN",
    "_ChunkKDA",
    "_ConfiguredShortConv",
    "_ContextParallelChunk",
    "_FusedGate",
    "_L2Norm",
    "_MaskRows",
    "_ShortConv",
    "bound_gate",
    "causal_conv1d",
    "causal_conv1d_decode",
    "ChunkGdnCudnnPacked",
    "ChunkKdaCudnn",
    "chunk_gdn",
    "chunk_kda",
    "context_parallel_conv_history",
    "context_parallel_gdn",
    "context_parallel_kda",
    "ContextParallelRouting",
    "gate_transform",
    "gather_attn",
    "KernelOptions",
    "l2norm",
    "lightning_indexer",
    "paged_causal_conv1d",
    "paged_chunk_gdn",
    "paged_chunk_kda",
    "recurrent_gdn",
    "recurrent_gdn_decode",
    "recurrent_kda_decode",
    "is_missing_kernel",
    "require_attn_gym",
]

_EXPORTS = {
    "AuxRequest": "attn_gym.sparse.gather_attn",
    "_ChunkGDN": "attn_gym.linear.gdn.ops",
    "_ChunkKDA": "attn_gym.linear.kda.impl.fused",
    "_ConfiguredShortConv": "attn_gym.linear.short_conv.cute",
    "_ContextParallelChunk": "attn_gym.linear.context_parallel",
    "_FusedGate": "attn_gym.linear._delta_rule.gate",
    "_L2Norm": "attn_gym.linear.kda.fwd.triton.l2norm_fwd",
    "_MaskRows": "attn_gym.linear.kda.masking",
    "_ShortConv": "attn_gym.linear.short_conv.cute",
    "bound_gate": "attn_gym.linear.kda",
    "causal_conv1d": "attn_gym.linear.short_conv",
    "causal_conv1d_decode": "attn_gym.linear.short_conv",
    "ChunkGdnCudnnPacked": "attn_gym.linear.gdn.impl.cudnn",
    "ChunkKdaCudnn": "attn_gym.linear.kda.impl.cudnn",
    "chunk_gdn": "attn_gym.linear",
    "chunk_kda": "attn_gym.linear.kda",
    "context_parallel_conv_history": "attn_gym.linear.context_parallel",
    "context_parallel_gdn": "attn_gym.linear.gdn.context_parallel",
    "context_parallel_kda": "attn_gym.linear.kda.context_parallel",
    "ContextParallelRouting": "attn_gym.linear.context_parallel",
    "gate_transform": "attn_gym.linear._delta_rule.gate",
    "gather_attn": "attn_gym.sparse.gather_attn",
    "KernelOptions": "attn_gym.linear",
    "l2norm": "attn_gym.linear.kda.fwd.triton.l2norm_fwd",
    "lightning_indexer": "attn_gym.sparse",
    "paged_causal_conv1d": "attn_gym.linear.short_conv",
    "paged_chunk_gdn": "attn_gym.linear",
    "paged_chunk_kda": "attn_gym.linear.kda",
    "recurrent_gdn": "attn_gym.linear",
    "recurrent_gdn_decode": "attn_gym.linear",
    "recurrent_kda_decode": "attn_gym.linear.kda",
}


# Import errors raised from inside attn_gym, keyed by the attn_gym module that
# raised them, or by "<module>.<export>" for an export the module resolves on
# first access. The package is installed, but a kernel backend it imports
# (cutlass, triton, ...) is not available on this platform.
_BACKEND_ERRORS: dict[str, ModuleNotFoundError] = {}


def _raise_for(error: ModuleNotFoundError | None) -> NoReturn:
    """Raise for a missing export.

    ``error`` is the backend error recorded for the export or its module.
    Without one, attn_gym itself is not installed.
    """
    if error is not None:
        # A fresh exception each time: re-raising the recorded instance would
        # keep appending tracebacks to one shared object.
        raise ModuleNotFoundError(
            str(error), name=error.name, path=error.path
        ) from error
    # Keep the version in sync with the attn-gym pin in .ci/docker/requirements.txt.
    raise ModuleNotFoundError(
        "Attention Gym kernels require attn_gym. "
        "Install it with: python -m pip install 'attn-gym[linear,cudnn]==0.0.17'.",
        name="attn_gym",
    )


class _MissingKernel:
    """Placeholder for an export that did not import.

    The class itself stands in when attn_gym is not installed. When attn_gym
    is installed but one of its modules needs a backend that is not, the
    exports of that module get a subclass carrying the backend's error, so a
    kernel reports the backend it needs rather than the first one recorded.
    """

    _error: ModuleNotFoundError | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        _raise_for(cls._error)

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> NoReturn:
        _raise_for(cls._error)

    @classmethod
    def from_fragments(cls, *args: Any, **kwargs: Any) -> NoReturn:
        _raise_for(cls._error)


_BACKEND_KERNELS: dict[str, type[_MissingKernel]] = {}


def _missing_backend_kernel(key: str) -> type[_MissingKernel]:
    kernel = _BACKEND_KERNELS.get(key)
    if kernel is None:
        kernel = type(
            f"_MissingKernel[{key}]",
            (_MissingKernel,),
            {"__module__": __name__, "_error": _BACKEND_ERRORS[key]},
        )
        _BACKEND_KERNELS[key] = kernel
    return kernel


def is_missing_kernel(kernel: object) -> TypeGuard[type[_MissingKernel]]:
    """Whether ``kernel`` is a placeholder for an export that did not import."""
    return isinstance(kernel, type) and issubclass(kernel, _MissingKernel)


def _is_backend_error(error: ModuleNotFoundError) -> bool:
    """Whether ``error`` is a kernel backend (cutlass, triton, ...) missing.

    A missing attn_gym module (moved in a release, or a typo in _EXPORTS) is
    breakage rather than an unsupported platform, so it is not deferred.
    """
    return error.name is not None and error.name.split(".")[0] != "attn_gym"


def _load_module(name: str) -> ModuleType | None:
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name == "attn_gym":
            return None
        if not _is_backend_error(error):
            raise
        # attn_gym is installed but this module needs a backend (cutlass,
        # triton, ...) that is not. Keep the models importable and report the
        # backend when a kernel is built, as when attn_gym itself is missing.
        _BACKEND_ERRORS.setdefault(name, error)
        return None


def _load_export(module: ModuleType, name: str) -> Any:
    """Return ``module.<name>``, or a placeholder when its backend is missing.

    attn_gym resolves backend-backed exports on first access (see
    ``attn_gym.linear._lazy``) and wraps the backend's ModuleNotFoundError in
    an ImportError, so for those the backend fails here, not in _load_module.
    This relies on attn_gym raising that ImportError with ``from``: only
    ``__cause__`` is inspected, an implicitly chained error (``__context__``)
    raises at import.
    """
    try:
        return getattr(module, name)
    except ImportError as error:
        cause = error.__cause__ or error
        if not isinstance(cause, ModuleNotFoundError) or not _is_backend_error(cause):
            raise
        key = f"{module.__name__}.{name}"
        # Keep attn_gym's message, which names the optional backend and how to
        # get it, where the chained ModuleNotFoundError only says what import
        # failed. The backend's name stays on the error for callers.
        recorded = ModuleNotFoundError(str(error), name=cause.name, path=cause.path)
        recorded.__cause__ = error
        _BACKEND_ERRORS.setdefault(key, recorded)
        return _missing_backend_kernel(key)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = _EXPORTS[name]
    module = _load_module(module_name)
    if module is not None:
        value = _load_export(module, name)
    elif module_name in _BACKEND_ERRORS:
        value = _missing_backend_kernel(module_name)
    else:
        value = _MissingKernel
    globals()[name] = value
    return value


def require_attn_gym(kernel: object, *kernels: object) -> None:
    """Raise unless every given export imported.

    A module calls this from ``__init__`` with every attn_gym export it
    uses, so a missing backend fails at build, naming the backend that
    module needs, rather than at the first forward.
    """
    for export in (kernel, *kernels):
        if is_missing_kernel(export):
            _raise_for(export._error)

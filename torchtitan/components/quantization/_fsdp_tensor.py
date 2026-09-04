# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private FSDP shard/unshard lifecycle for quantized parameters.

The lifecycle uses two tensor subclasses, one per state:

``_ShardedFSDPTensor``
    The persistent parameter. Holds the high-precision shard, owns the FSDP
    pre/post-all-gather hooks, and knows how to build a format's unsharded
    operands. A format subclasses this and implements one method.

``_UnshardedFSDPTensor``
    The unsharded tensor for one unshard lifetime. Holds only the format's
    operands and no high-precision storage, which is what lets FSDP release
    the all-gather output. Generic: it derives its unsharded inner tensors from the
    operands dataclass fields, so formats do not subclass it.

Both present the logical high-precision metadata of the model parameter, so
autograd returns a high-precision parameter gradient in either state.

Instance timeline, from module construction to reshard::

    module __init__      S = _ShardedFSDPTensor(storage_shard)
                         |     one instance, lives for the whole run;
                         |     it is the nn.Parameter and the checkpoint
                         v
    ---- unshard ----------------------------------------------------------
    fsdp_pre_all_gather  S._tensor.to(param_dtype)  ->  comm tensor
                         |
                         v
    (all-gather)         replicated param_dtype tensor, temporary
                         |
    fsdp_post_all_gather |  out is None: first unshard
      S builds ------->  C = _UnshardedFSDPTensor(operands)
                         |     new instance; holds only qdata/scales.
                         |     Returned to FSDP with the operands'
                         |     tensors so FSDP can manage their storage.
                         v
                         comm tensor released -- C never referenced it
                         |
    forward/backward     compute reads C.operands
                         |
    ---- reshard ----------------------------------------------------------
                         FSDP frees the storage of C's unsharded inner tensors.
                         C itself stays alive: autograd and the module may
                         still hold it, and its addresses must not move.
                         |
    ---- unshard again --------------------------------------------------->
    fsdp_post_all_gather |  out is C: refill
      S refills ------>  C's existing tensors are written in place
                         |     no new instance; _validate_refilled_tensor_
                         |     identity() enforces that
                         v
                         (repeats until the final reshard)

So S is created once per parameter and C once per *distinct* unshard
lifetime -- not once per unshard. RAF=False keeps a single C alive across
forward, backward, recomputation, and pipeline microbatches; RAF=True
reuses that same C object, refilling its storage before backward.

GraphTrainer's SimpleFSDP reaches the same place by a different route: it
reconstructs the replicated tensor itself, then quantizes and wraps it from
inside its own parametrization, constructing C directly. That path owns the
gradient edge FSDP2 creates internally, so it lives with SimpleFSDP rather
than here.

Two dtypes appear above and they are not the same one. S holds the parameter's
own storage dtype, set by ``training.dtype`` -- float32 by default, so S is
normally an fp32 master weight, not a BF16 one. ``mp_policy.param_dtype``
(``training.mixed_precision_param``, bfloat16 by default) is only what
``fsdp_pre_all_gather`` casts *to*, so it is the dtype of the comm tensor, the
all-gather output, and hence the logical tensor a format quantizes. Nothing
here requires either to be BF16; MXFP8 separately rejects a non-BF16 weight in
``_MXFP8LinearFunction.forward``, because its kernels need one.

Terminology
-----------

Several names here are a word apart, so:

operands
    A format's quantized output: a frozen dataclass of qdata, scales, any
    workspace. Plain data; not a tensor subclass.

unsharded tensor
    The ``_UnshardedFSDPTensor`` that *holds* the operands and presents the
    parameter's logical high-precision metadata. "C" below.

``_build_operands(logical_tensor, out=None)``
    The format's quantizer, and the only method a format implements. Returns
    a fresh operands dataclass, or with ``out`` set refills that one's
    existing tensors in place instead of allocating.

logical tensor
    The unsharded high-precision tensor handed to the quantizer. "Logical"
    because it matches the size the model declares, excluding any padding the
    all-gather added to make every shard the same size.

unsharded inner tensors
    The operands' dataclass *fields*, whose storage FSDP allocates, frees, and
    refills across the unshard lifecycle. A strict subset of what the operands
    expose: a derived view belongs in a property, and a property is never one
    of these, since FSDP must not free the same allocation twice. They are
    exactly the tensors ``__tensor_flatten__`` reports, which FSDP holds as
    ``FSDPParam._unsharded_inner_tensors`` and frees in
    ``free_unsharded_param``.

metadata source
    The tensor an unsharded tensor copies its shape, dtype, device, and
    layout from -- normally the unsharded high-precision tensor it was built
    from, since an unsharded tensor has no storage of its own to describe.

Adding a format
---------------

Subclass ``_ShardedFSDPTensor``; do not subclass ``_UnshardedFSDPTensor``.

A format differs from every other format in exactly one way: how it turns a
high-precision tensor into its operands. That belongs on the sharded class,
because the sharded tensor is the parameter FSDP calls hooks on, so it is
what *produces* the operands. ``_UnshardedFSDPTensor`` only *holds* them, and
holding is format-independent -- it reads the operand tensors off the
operands dataclass fields, which works for any format. Subclassing
it would add a type that overrides nothing.

So a format supplies a frozen dataclass of the tensors one unshard lifetime
owns, and one method, ``_build_operands``. Everything else --
flattening, refill, reshard, the SimpleFSDP bridge -- comes from here.
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


# The unsharded-tensor machinery is internal. A data-parallel implementation
# reaches it through the FSDP hooks, or by calling
# ``_build_operands`` itself when it reconstructs the
# unsharded tensor, as GraphTrainer's SimpleFSDP does.
__all__: list[str] = []

# Ops FSDP performs on the sharded parameter's real storage. The wrapper must
# survive them so the parameter keeps its identity across FSDP bookkeeping.
_FSDP_SHARDED_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.c10d.scatter_.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.alias.default,
}

# The unsharded tensor has no high-precision storage, so only metadata-level ops
# are answerable. Views re-wrap; factories allocate fresh plain tensors.
_FSDP_UNSHARDED_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
}

_FSDP_UNSHARDED_FACTORY_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.zeros_like.default,
}


def _unsharded_inner_tensor_names(operands_cls: type) -> tuple[str, ...]:
    """Return the names of the unsharded inner tensors an operands dataclass owns."""
    if not is_dataclass(operands_cls):
        raise TypeError(
            "An operands must be a dataclass of tensors; got "
            f"{operands_cls.__name__}."
        )
    return tuple(field.name for field in fields(operands_cls))


def _unsharded_inner_tensors(operands: Any) -> tuple[torch.Tensor, ...]:
    """Return the unsharded inner tensors an operands dataclass owns.

    Field order defines the unsharded inner tensor order, so no format restates
    it. Every field must be a distinct allocation: FSDP takes ownership of
    each one's storage, so listing two views of the same storage would make it
    free the same memory twice and leave ``__tensor_unflatten__`` unable to
    tell which field was the derived one. Derived views belong in properties,
    which ``fields()`` skips.
    """
    tensors = tuple(
        getattr(operands, name)
        for name in _unsharded_inner_tensor_names(type(operands))
    )
    storages = {tensor.untyped_storage()._cdata for tensor in tensors}
    if len(storages) != len(tensors):
        raise ValueError(
            f"{type(operands).__name__} fields must be distinct "
            "allocations; a field aliasing another field's storage should be "
            "a property instead."
        )
    return tensors


def _validate_refilled_tensor_identity(
    unsharded_inner_tensors: tuple[torch.Tensor, ...],
    refilled_tensors: tuple[torch.Tensor, ...],
) -> None:
    """Require a refill to preserve every tensor object managed by FSDP."""
    if len(unsharded_inner_tensors) != len(refilled_tensors) or any(
        previous is not current
        for previous, current in zip(
            unsharded_inner_tensors, refilled_tensors, strict=True
        )
    ):
        raise RuntimeError(
            "FSDP unsharded-operands refill replaced inner tensor storage"
        )


class _FSDPTensorBase(torch.Tensor):
    """Logical high-precision metadata shared by both lifecycle states."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any):
        del args
        return torch.Tensor._make_wrapper_subclass(
            cls,
            kwargs.get("_logical_size", tensor.size()),
            strides=kwargs.get("_logical_stride", tensor.stride()),
            storage_offset=kwargs.get(
                "_logical_storage_offset", tensor.storage_offset()
            ),
            dtype=kwargs.get("_logical_dtype", tensor.dtype),
            layout=tensor.layout,
            device=kwargs.get("_logical_device", tensor.device),
            pin_memory=tensor.is_pinned(),
            requires_grad=kwargs.get("_logical_requires_grad", tensor.requires_grad),
        )


class _ShardedFSDPTensor(_FSDPTensorBase):
    """Persistent high-precision parameter that owns the FSDP hooks.

    This is the sharded half of the lifecycle. It is the ``nn.Parameter`` the
    optimizer updates and the checkpoint stores, it holds the high-precision
    shard in ``_tensor``, and it lives for the whole run. It is *not* what
    compute sees under FSDP: the post-all-gather hook hands back a
    :class:`_UnshardedFSDPTensor` holding the quantized operands, and that is
    what forward and backward read for the duration of one unshard. See the
    instance timeline at the top of this module for how the two hand off.

    **This is the class a format subclasses**, because a format differs only
    in how it turns a high-precision tensor into operands, and this is the
    side that produces them. ``_UnshardedFSDPTensor`` only holds them, which is
    format-independent, so it is generic and is never subclassed.

    A subclass supplies:

    * a frozen dataclass of the tensors one unshard lifetime owns, whose
      fields are distinct allocations -- derived views belong in properties;
    * ``_build_operands(logical_tensor, out=None)``, which
      allocates a new operands when ``out`` is None and otherwise
      refills ``out``'s existing tensors in place.

    Everything else -- flattening, refill, reshard, and the SimpleFSDP bridge
    -- is inherited.
    """

    def __init__(self, tensor: torch.Tensor, **logical_metadata: Any) -> None:
        del logical_metadata
        self._tensor = tensor

    def __tensor_flatten__(self):
        return ["_tensor"], (self.dtype,)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        del metadata, outer_size, outer_stride
        return cls(inner_tensors["_tensor"])

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        del types
        template = None
        preserve_wrapper = func in _FSDP_SHARDED_OPS

        def unwrap(tensor: _ShardedFSDPTensor) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif preserve_wrapper and type(tensor) is not type(template):
                raise RuntimeError("FSDP operation mixed sharded tensor types")
            return tensor._tensor

        output = func(
            *pytree.tree_map_only(cls, unwrap, args or ()),
            **pytree.tree_map_only(cls, unwrap, kwargs or {}),
        )
        if not preserve_wrapper:
            return output
        assert template is not None
        return pytree.tree_map_only(torch.Tensor, type(template), output)

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: Any = None,
    ) -> Any:
        """Quantize ``logical_tensor``, into ``out``'s tensors when refilling."""
        raise NotImplementedError

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        """Release the high-precision all-gather output after state construction."""
        return True

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        """Return the high-precision communication tensor and the logical size.

        All-gather needs every rank to contribute the same number of elements,
        so an expert count that does not divide the mesh size leaves the last
        rank short. FSDP's contract is that this returns the *padded* shard;
        the logical size travels in the metadata so ``fsdp_post_all_gather``
        can drop the padding before quantizing the logical tensor.
        """
        del outer_stride, module
        # FSDP hands the hook no shard dimension, but the local shard differs
        # from the logical size exactly along it. The default, non-extension
        # path has it directly as ``fsdp_placement.dim`` and rejects the same
        # case there:
        # https://github.com/pytorch/pytorch/blob/c7da99c173f2b67905ee798576a644b6b32cbfee/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L323-L331
        sharded_dims = [
            dim
            for dim, (local, logical) in enumerate(
                zip(self._tensor.shape, outer_size, strict=True)
            )
            if local != logical
        ]
        if len(sharded_dims) > 1:
            raise RuntimeError(
                f"FSDP sharded more than one dimension: local "
                f"{tuple(self._tensor.shape)} against logical {tuple(outer_size)}"
            )
        if sharded_dims and sharded_dims[0] != 0:
            raise NotImplementedError(
                "FSDP unsharded tensors support sharding dimension 0 only, but "
                f"this parameter of shape {tuple(outer_size)} is sharded on "
                f"dimension {sharded_dims[0]}. TorchTitan selects Shard(1) for "
                "grouped experts when the FSDP degree exceeds the expert "
                "count, so either lower the degree or raise the expert count."
            )
        dtype = mp_policy.param_dtype or self._tensor.dtype
        # Pad to what FSDP calls ``padded_sharded_param_size``. The default
        # path pre-pads to ``chunks[0].size()``, and torch.chunk puts the
        # remainder in the earlier chunks, so that equals ceil(dim0 / world):
        # https://github.com/pytorch/pytorch/blob/c7da99c173f2b67905ee798576a644b6b32cbfee/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L332-L345
        # An extension must return exactly that size; only the short ranks
        # would trip the check, so the rest hang in the all-gather instead:
        # https://github.com/pytorch/pytorch/blob/c7da99c173f2b67905ee798576a644b6b32cbfee/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L1143-L1158
        #
        # Note the padding happens per unshard here, where the default path
        # pads once. It pre-pads the sharded parameter at init and keeps that
        # buffer for the run ("Pre-pad the sharded parameter to avoid padding
        # before all-gather"), but an extension is handed the unpadded shard
        # every time. TODO(anijain2305): hold a persistent padded buffer on the
        # sharded tensor and copy into it, to drop the per-unshard allocation.
        padded_rows = math.ceil(outer_size[0] / mesh.size())
        if self._tensor.size(0) != padded_rows:
            # Allocate the padded buffer directly in the comm dtype and let the
            # copy do the cast, rather than casting the whole shard first and
            # then copying that into a second buffer.
            source = self._tensor.new_zeros(
                (padded_rows, *self._tensor.shape[1:]), dtype=dtype
            )
            source.narrow(0, 0, self._tensor.size(0)).copy_(self._tensor)
        else:
            source = self._tensor.to(dtype)
        return (source,), outer_size

    def fsdp_post_all_gather(
        self, all_gather_outputs, metadata, param_dtype, *, out=None
    ):
        """Create or refill the unsharded tensor operands after all-gather."""
        del param_dtype
        (logical_tensor,) = all_gather_outputs
        # ``metadata`` is the logical size returned by fsdp_pre_all_gather. An
        # unevenly sharded parameter gathers padding rows past it, which must
        # not reach the quantizer: they would occupy real scale tiles and, for
        # a grouped expert tensor, appear as extra experts.
        if metadata is not None and logical_tensor.size(0) != metadata[0]:
            logical_tensor = logical_tensor.narrow(0, 0, metadata[0])

        # On the first unshard, FSDP has no unsharded-tensor container or managed
        # tensors yet. Build both and return them to FSDP. With RAF=False, FSDP
        # keeps these operands alive through forward and backward.
        if out is None:
            with torch.no_grad():
                operands = self._build_operands(logical_tensor)
            return (
                _UnshardedFSDPTensor(logical_tensor, operands),
                _unsharded_inner_tensors(operands),
            )

        # After FSDP releases and later unshards the tensor again, ``out`` is
        # the same unsharded-tensor object returned above. This occurs between
        # forward and backward with RAF=True, or after a later reshard. Refill
        # the same unsharded inner tensor objects so existing module and autograd
        # references remain valid. ``out`` is what this hook returned, so it is
        # always the bare unsharded tensor -- FSDP2 never re-wraps it.
        target = out
        if not isinstance(target, _UnshardedFSDPTensor):
            raise RuntimeError("FSDP output does not own operands")
        existing = target.operands
        unsharded_inner_tensors = _unsharded_inner_tensors(existing)
        with (
            torch.no_grad(),
            # Refilling lifecycle-managed storage is not a user-visible tensor
            # mutation and must not invalidate saved-tensor version checks.
            torch.autograd._unsafe_preserve_version_counter(unsharded_inner_tensors),
        ):
            refilled = self._build_operands(logical_tensor, out=existing)
        _validate_refilled_tensor_identity(
            unsharded_inner_tensors, _unsharded_inner_tensors(refilled)
        )
        target._operands = refilled


class _UnshardedFSDPTensor(_FSDPTensorBase):
    """Unsharded tensor holding one unshard lifetime's format operands.

    The unsharded half of the lifecycle, built by
    :class:`_ShardedFSDPTensor`'s post-all-gather hook and alive until the
    final reshard. Carries no high-precision storage -- that is what lets FSDP
    release the all-gather output -- so reading it as a high-precision tensor
    is an error; only format-aware consumers may read
    ``operands``. See the instance timeline at the top of this
    module for how the two classes hand off.

    Generic by design: the unsharded inner tensors come from the operands'
    dataclass fields, which works for any format. Do not subclass it; formats
    subclass :class:`_ShardedFSDPTensor` instead.
    """

    def __init__(
        self,
        metadata_source: torch.Tensor,
        operands: Any,
        **logical_metadata: Any,
    ) -> None:
        # ``__new__`` already consumed both to build the wrapper subclass with
        # this tensor's logical size, stride, dtype, device and requires_grad.
        # Python hands ``__init__`` the same arguments, so drop them here
        # rather than store them: the metadata lives on the tensor itself.
        del metadata_source, logical_metadata
        self._operands = operands
        # __tensor_flatten__ reports the unsharded inner tensors by attribute name and
        # the subclass machinery fetches them with a plain getattr, so each has
        # to exist as an attribute here -- reaching into the operands is
        # not an option. Mirror rather than copy: these are the same tensor
        # objects, so an in-place refill updates both views, and a refill that
        # substituted objects is rejected by
        # _validate_refilled_tensor_identity.
        for name in _unsharded_inner_tensor_names(type(operands)):
            setattr(self, f"_{name}", getattr(operands, name))

    def __tensor_flatten__(self):
        operands_cls = type(self._operands)
        names = [f"_{name}" for name in _unsharded_inner_tensor_names(operands_cls)]
        return names, (operands_cls, self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        operands_cls, dtype = metadata
        unsharded_inner_tensors = [
            inner_tensors[f"_{name}"]
            for name in _unsharded_inner_tensor_names(operands_cls)
        ]
        operands = operands_cls(*unsharded_inner_tensors)
        # FSDP supplies the logical shape; any unsharded inner tensor can stand in for
        # the rest, since they share the unsharded tensor's device and layout.
        return _UnshardedFSDPTensor(
            unsharded_inner_tensors[0],
            operands,
            _logical_size=outer_size,
            _logical_stride=outer_stride,
            _logical_dtype=dtype,
        )

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        del types
        template = None

        def unwrap(tensor: _UnshardedFSDPTensor) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif tensor._operands is not template._operands:
                raise RuntimeError("FSDP operation mixed unsharded tensor operands")
            # There is no high-precision storage to hand the op; a meta tensor
            # carries the logical metadata that view ops need.
            return torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device="meta",
                requires_grad=tensor.requires_grad,
            )

        def wrap_view(tensor: torch.Tensor):
            assert template is not None
            operands = template._operands
            # __new__ reads layout and pinning off a real tensor, and the
            # template has no storage to answer with, so borrow a managed
            # tensor for those two and give the view's logical metadata for
            # everything else. Which one does not matter: they
            # share the unsharded tensor's device, layout, and pinning.
            layout_source = _unsharded_inner_tensors(operands)[0]
            return _UnshardedFSDPTensor(
                layout_source,
                operands,
                _logical_size=tensor.size(),
                _logical_stride=tensor.stride(),
                _logical_storage_offset=tensor.storage_offset(),
                _logical_dtype=template.dtype,
                _logical_device=template.device,
                _logical_requires_grad=tensor.requires_grad,
            )

        original_args, original_kwargs = args, kwargs or {}
        args, kwargs = pytree.tree_map_only(
            cls, unwrap, (original_args, original_kwargs)
        )
        assert template is not None
        if func in _FSDP_UNSHARDED_FACTORY_OPS:
            kwargs["device"] = template.device
            return func(*args, **kwargs)
        if func not in _FSDP_UNSHARDED_VIEW_OPS:
            raise RuntimeError(
                f"{func} attempted to read a storage-free FSDP unsharded tensor"
            )
        wrapped = pytree.tree_map_only(torch.Tensor, wrap_view, func(*args, **kwargs))
        return return_and_correct_aliasing(
            func, original_args, original_kwargs, wrapped
        )

    @property
    def operands(self) -> Any:
        """Return the operands for the current unshard lifetime."""
        return self._operands

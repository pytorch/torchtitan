# FSDP1 -> FSDP2

## Why FSDP2?
PyTorch's fully sharded data parallelism (FSDP) API, [`FullyShardedDataParallel`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py), looks to offer a performant eager-mode implementation, including communication bucketing and communication/computation overlap. It defines a `FlatParameter` by flattening and concatenating a group of parameters to represent a communication bucket. However, this `FlatParameter` complicates applying different behaviors to individual parameters within the `FlatParameter`, e.g. parameter freezing, parameter casting, etc, hurting composability, and complicates the internal implementation, e.g. making state dict logic thousands of lines.

With these limitations in mind, we designed and implemented an FSDP rewrite removing the `FlatParameter`.  We refer to this rewrite as FSDP2 and the original as FSDP1. FSDP2 targets the same use cases as FSDP1 plus more. FSDP2 still strives for the same performance in eager mode, using several of the same techniques.

Compared to FSDP1:
- FSDP2 represents sharded parameters as `DTensor`s sharded on dim-0, allowing for easy manipulation of individual parameters, communication-free sharded state dicts, and a simpler meta-device initialization flow.
- FSDP2 implements an improved memory management system that achieves lower and deterministic GPU memory by avoiding `recordStream` and does so without any CPU synchronization.

In the future, FSDP2 will offer an extension point to customize the all-gather (e.g. for fp8 all-gather for fp8 linears) and improved `torch.compile` support.

We have validated FSDP2 numerics and performance using torchtrain (e.g. see this [PR](https://github.com/pytorch/torchtrain/pull/165)).

For more details on motivation, API, and system design, refer to [here](https://github.com/pytorch/pytorch/issues/114299). In this README, we try to provide more user-facing info and less system design details.

## FSDP1 <> FSDP2 Differences
We go over some differences between FSDP1 and FSDP2, starting with the FSDP2 API. Overall, we are trying to minimize the API surface (e.g. number of arguments) and avoid having a monolithic API.
```
@contract(state_cls=FSDPState)
def fully_shard(
  module: nn.Module,
  *,
  mesh: Optional[DeviceMesh] = None,
  reshard_after_forward: Union[bool, int] = True,
  mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
  offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:  # return value only used by `contract` for checks
```
- `fully_shard(module)` is similar to `FullyShardedDataParallel(module)`, constructing one communication bucket of `module.parameters()` (except those already assigned to a nested FSDP call).
    - `fully_shard(module)` adds an `FSDPState` object on `module`, accessible via `fully_shard.state(module)`, instead of being an `nn.Module` wrapper. This is done via the `@contract` decorator.
- FSDP1's `sharding_strategy` and `process_group`/`device_mesh` maps to FSDP2's `mesh` and `reshard_after_forward`.
  - `mesh` should be 1D for FSDP and 2D for HSDP. For HSDP, we assume sharding on the 0th mesh dim and replication on the 1st mesh dim. If `mesh is None`, then FSDP2 initializes a 1D global mesh over the default process group.
  - `reshard_after_forward=True` means parameters are resharded (freed) after forward and re-all-gathered in backward. This with 1D mesh equals `ShardingStrategy.FULL_SHARD`. This with 2D mesh equals `ShardingStrategy.HYBRID_SHARD`.
  - `reshard_after_forward=False` means parameters are not resharded (freed) after forward, not needing any all-gather in backward. This with 1D mesh equals `ShardingStrategy.SHARD_GRAD_OP`. This with 2D mesh equals `ShardingStrategy._HYBRID_SHARD_ZERO2`.
  - (Experimental) `reshard_after_forward: int` means that parameters are resharded to a smaller world size after forward (e.g. `reshard_after_forward=8` can mean intra-node) so that the backward all-gather is over a smaller world size.
- For the other arguments:
  - FSDP1's `mixed_precision` maps to FSDP2's `mp_policy`. We remove `buffer_dtype`, simplify `cast_forward_inputs` and `cast_root_forward_inputs` into just `cast_forward_inputs`, and add an `output_dtype`. 
  - FSDP2 removes `auto_wrap_policy`. Auto wrapping refers to "automatically" traversing `module.modules()`, calling `FullyShardedDataParallel(submodule, **kwargs)` according to the policy, and assigning to the parent module if called. We prefer for this to exist above `fully_shard`.
  - FSDP2 removes `backward_prefetch` and follows `BACKWARD_PRE` always. Only `BACKWARD_PRE` always prefetches the correct order; `BACKWARD_POST` can prefetch [incorrectly](https://github.com/pytorch/pytorch/issues/108190) in nested cases. FSDP2 does not support `forward_prefetch` yet.
  - FSDP2 does not support `ignored_modules` or `ignored_states` for now. We want to evaluate the use cases before adding an extra argument.
  - FSDP2 does not support `param_init_fn` since it supports a hopefully simpler meta-device initialization flow.
  - FSDP2 does not support `device_id`. It always moves managed modules' parameters/buffers to the current CUDA device if given a CUDA `mesh`.
  - FSDP2 removes `limit_all_gathers`, instead using an improved memory management system without any CPU rate limiting.
  - FSDP2 removes `use_orig_params` since the original parameters are always exposed under dim-0 per-parameter sharding.
- Calling `model.named_parameters()` for a `model` with FSDP2 applied returns unchanged parameter names and `DTensor` sharded parameters. This means that the optimizer and gradient norm clipping see `DTensor`s.
- `fully_shard(module)` performs a dynamic class swap on `module`. E.g., if `type(module) is Transformer`, then FSDP2 constructs a new class `FSDPTransformer` that inherits from a class `FSDP` and `Transformer` and sets `module.__class__` to be `FSDPTransformer`. This allows us to add new methods and override methods via `FSDP` without constructing an `nn.Module` wrapper.
- Calling `model.state_dict()` for an FSDP2 `model` returns a "sharded state dict" (without any communication), whereas for FSDP1, it returns a "full state dict".

## Meta-Device Initialization
Before with FSDP1:
```
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
with torch.device("meta"):
    model = Transformer()
policy = ModuleWrapPolicy({TransformerBlock})
# Call `reset_parameters()` on every module
model = FSDP(model, auto_wrap_policy=policy)
# Call `param_init_fn` on every module
def param_init_fn(module: nn.Module) -> None: ...
model = FSDP(model, auto_wrap_policy=policy, param_init_fn=param_init_fn)
```
After with FSDP2:
```
with torch.device("meta"):
    model = Transformer()
for module in model.modules():
if isinstance(module, TransformerBlock):
    fully_shard(module)
fully_shard(model)
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")
# Allocate buffers and sharded parameters on GPU
model.to_empty("cuda")
# Run user-defined initializers
model.init_weights() # or `model.apply(init_weights)`
```
FSDP1 requires either `reset_parameters` or `param_init_fn` to materialize a module onto GPU immediately before sharding. To do this correctly without re-initializing any tensors requires care and can be unwieldy. However, FSDP2 allows materializing tensors onto GPU _after_ sharding (taking advantage of `DTensor` and a new `swap_tensors` path for `nn.Module._apply` methods).

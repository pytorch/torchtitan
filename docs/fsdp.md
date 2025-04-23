# FSDP1 -> FSDP2

## Why FSDP2?
PyTorch's fully sharded data parallelism (FSDP) API, [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html), looks to offer a performant eager-mode implementation, including communication bucketing and communication/computation overlap. It defines a `FlatParameter` by flattening and concatenating a group of parameters to represent a communication bucket. However, this `FlatParameter` complicates applying different behaviors to individual parameters within the `FlatParameter`, e.g. parameter freezing, parameter casting, etc., hurting composability, and it complicates the internal implementation, e.g. making state dict logic thousands of lines and requiring additional communications.

With these limitations in mind, we designed and implemented an FSDP rewrite removing the `FlatParameter`.  We refer to this rewrite as FSDP2 and the original as FSDP1. FSDP2 targets the same use cases as FSDP1 plus more, and FSDP2 still strives for good performance in eager mode, using several of the same techniques.

Compared to FSDP1:
- FSDP2 represents sharded parameters as `DTensor`s sharded on dim-0, allowing for easy manipulation of individual parameters, communication-free sharded state dicts, and a simpler meta-device initialization flow.
- FSDP2 implements an improved memory management system that achieves lower and deterministic GPU memory by avoiding `recordStream` and does so without any CPU synchronization.

In the future, FSDP2 will offer an extension point to customize the all-gather (e.g. for fp8 all-gather for fp8 linears) and improved `torch.compile` support.

We have validated FSDP2 numerics and performance using torchtitan (e.g. see this [PR](https://github.com/pytorch/torchtitan/pull/165)). For example, on some Llama-7B runs on 8x H100s, FSDP2 achieves higher MFU with 7% lower peak memory than FSDP1, matching the same loss curve.

For more details on motivation, API, and system design, refer to [here](https://github.com/pytorch/pytorch/issues/114299). In this README, we try to provide more user-facing info and less system design details.

## FSDP1 <> FSDP2 API Differences
We go over some API differences between FSDP1 and FSDP2. Overall, we hope to minimize the API surface (including the number of arguments) to avoid having a monolithic API.
```python
@contract(state_cls=FSDPState)
def fully_shard(
  module: nn.Module,
  *,
  mesh: Optional[DeviceMesh] = None,
  reshard_after_forward: Union[bool, int] = True,
  mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
  offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:  # returns `module` for `contract` checks
```

| FSDP1 | FSDP2 |
| ----- | ----- |
| `module` | `module` |
| `process_group`/`device_mesh` | `mesh` |
| `sharding_strategy` | `reshard_after_forward` |
| `cpu_offload` | `offload_policy` |
| `auto_wrap_policy` | removed |
| `backward_prefetch` | removed |
| `mixed_precision` | `mp_policy` |
| `param_init_fn` | removed |
| `device_id` | removed |
| `sync_module_states` | removed |
| `forward_prefetch` | not yet implemented |
| `limit_all_gathers` | removed |
| `use_orig_params` | removed |
| `no_sync` | `set_requires_gradient_sync` |
| `ignored_modules`, `ignored_states` | `ignored_params` |

- `fully_shard(module)` is similar to `FullyShardedDataParallel(module)`, constructing one communication bucket from `module.parameters()` except those already assigned to a nested `fully_shard`/`FullyShardedDataParallel` call.
    - `fully_shard(module)` adds an `FSDPState` object on `module`, accessible via `fully_shard.state(module)`, instead of being an `nn.Module` wrapper. This is done via the `@contract` decorator.
    - Calling `model.named_parameters()` for a `model` with FSDP2 applied returns unchanged parameter names and `DTensor` sharded parameters. This means that the optimizer and gradient norm clipping see `DTensor`s.
    - `fully_shard(module)` performs a dynamic class swap on `module`. E.g., if `type(module) is Transformer`, then FSDP2 constructs a new class `FSDPTransformer` that inherits from a class `FSDPModule` and `Transformer` and sets `module.__class__` to be `FSDPTransformer`. This allows us to add new methods and override methods via `FSDPModule` without constructing an `nn.Module` wrapper.
- FSDP1's `sharding_strategy` and `process_group`/`device_mesh` maps to FSDP2's `mesh` and `reshard_after_forward`.
  - `mesh` should be 1D for FSDP and 2D for HSDP. For HSDP, we assume replication on the 0th mesh dim and sharding on the 1st mesh dim. If `mesh is None`, then FSDP2 initializes a 1D global mesh over the default process group.
  - `reshard_after_forward=True` or `False` determines whether parameters are resharded (freed) after forward. If `True`, then they are re-all-gathered in backward. This trades off saving memory at the cost of extra communication.
  - (Experimental) `reshard_after_forward: int` means that parameters are resharded to a smaller world size after forward (e.g. `reshard_after_forward=8` can mean intra-node) so that the backward all-gather is over a smaller world size.
  - | FSDP1 | FSDP2 | DeepSpeed |
    | --- | --- | --- |
    | 1 `process_group` + `FULL_SHARD` | 1D `mesh` + `reshard_after_forward=True` | ZeRO-3 |
    | 1 `process_group` + `SHARD_GRAD_OP` | 1D `mesh` + `reshard_after_forward=False` | ZeRO-2 |
    | 2 `process_group`s/2D `device_mesh` + `HYBRID_SHARD` | 2D `mesh` + `reshard_after_forward=True` | MiCS |
    | 2 `process_group`s/2D `device_mesh` + `_HYBRID_SHARD_ZERO2` | 2D `mesh` + `reshard_after_forward=False` | - |
    | - | 1D/2D `mesh` + `reshard_after_forward=8` (`int`) | ZeRO++ hpZ |
- FSDP2 maps `mixed_precision` to `mp_policy` and `cpu_offload` to `offload_policy`.
  - For `mp_policy`, we remove `buffer_dtype`, simplify `cast_forward_inputs` and `cast_root_forward_inputs` into just `cast_forward_inputs`, and add an `output_dtype`.
  - For `offload_policy`, we add a `pin_memory` option to avoid pinning CPU memory. (This feature may not have landed yet.)
- FSDP2 removes `auto_wrap_policy`, `backward_prefetch`, `param_init_fn`, `device_id`, `sync_module_states`, `limit_all_gathers`, and `use_orig_params`.
  - `auto_wrap_policy` provides a syntactic sugar for calling `FullyShardedDataParallel` on modules based on a predicate given by the policy and assigning the wrapped module to its parent. FSDP2 is no longer an `nn.Module` wrapper, so there is need to assign the module back to its parent. We prefer for this functionality to exist above `fully_shard`, and we may provide a utility like `auto_wrap_policy` in the future.
  - FSDP2 always follows `backward_prefetch=BACKWARD_PRE` without option since that is the only way to overlap collectives in backward correctly. `BACKWARD_POST` can prefetch [incorrectly](https://github.com/pytorch/pytorch/issues/108190) in nested-module cases.
  - FSDP2 supports a new meta-device initialization flow that does not require materializing a module on GPU *before* sharding it, removing the need for `param_init_fn`. See [Meta-Device Initialization](#meta-device-initialization) for more details.
  - FSDP2 always moves managed parameters/buffers to the `mesh`'s corresponding device, removing the need for `device_id`. For example, if `mesh.device_type` is `"cuda"`, then FSDP2 uses the current CUDA device.
  - FSDP2 uses a new memory management system that preserves communication/computation overlap while achieving deterministic and lower memory usage than FSDP1. This system does not require any CPU synchronization, so there is no need for `limit_all_gathers`.
  - FSDP2 always "uses the original parameters" since there is no more `FlatParameter`, removing the need for `use_orig_params`.
- How to implement `forward_prefetch` in FSDP2 is under discussion.

| FSDP1 | FSDP2 |
| ----- | ----- |
| `model.state_dict()`: full state dict | `model.state_dict()`: sharded state dict (no communication) |
| `optim.state_dict()`: local state dict | `optim.state_dict()`: sharded state dict (no communication) |
| `summon_full_params()` | use `DTensor` APIs like `full_tensor()` |
| `FSDP.clip_grad_norm_()` | `nn.utils.clip_grad_norm_()` |
| `ShardedGradScaler` | `amp.grad_scaler.GradScaler` |


## Meta-Device Initialization
Before with FSDP1:
```python
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
```python
with torch.device("meta"):
    model = Transformer()
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)
fully_shard(model)
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")
# Allocate buffers and sharded parameters on GPU
model.to_empty(device="cuda")
# Run user-defined initializers
model.init_weights() # or `model.apply(init_weights)`
```
FSDP1 requires either `reset_parameters` or `param_init_fn` to materialize a module onto GPU immediately before sharding. To do this correctly without re-initializing any tensors requires care and can be unwieldy. However, FSDP2 allows materializing tensors onto GPU _after_ sharding (taking advantage of `DTensor` and a new `swap_tensors` path for `nn.Module._apply` methods).

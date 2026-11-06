# Copy-engine FSDP collectives

This experiment replaces FSDP2's communication hooks while keeping its prefetch,
streams, padding, gradient scaling, and parameter management.

* **All-gather:** one multicast copy-engine transfer between two single-CTA barriers.
* **Reduce-scatter:** an intra-domain CE all-to-all and local sum. Across domains,
  a second NCCL all-to-all and local sum finish the reduction.

The payload copies within NVLink do not consume SMs. The barriers use one CTA;
the local sums use multiple CTAs for large tensors. This is not a promise that
the complete reduce-scatter occupies one SM. No NVSHMEM, FARM, or ParallelKittens
dependency is required. NCCL supplies process groups and inter-domain transport.

## Enable

Use a current PyTorch nightly with FSDP2 custom communication hooks, Triton,
`cuda-bindings>=12.3`, and multicast-capable NVIDIA GPUs in one NVLink domain.
Install the extra dependency:

```bash
pip install -r torchtitan/experiments/ce_collectives/requirements.txt
```

Convert an ordinary TorchTitan recipe:

```python
from torchtitan.experiments.ce_collectives.trainer import to_ce_trainer_config
from torchtitan_recipes.models.llama3 import llama3_8b

def my_recipe():
    return to_ce_trainer_config(llama3_8b())
```

Or run the small C4 example:

```bash
MODULE=torchtitan_recipes.ce_collectives CONFIG=ce_debug NGPU=4 ./run_train.sh
```

The symmetric-memory backend must be `CUDA`. Do not combine this with
`parallelism.fsdp_symm_mem_scope`, CPU offload, or another experimental trainer.
`scope="dense"` skips expert-containing transformer blocks. Multiple parameter
groups within one selected FSDP module are unsupported and raise an error.

## Two domains

For 128 FSDP ranks arranged as two consecutive 64-rank NVLink domains:

```python
config = to_ce_trainer_config(base, all_gather=False, ranks_per_domain=64)
```

All-gather then uses the ordinary FSDP implementation. CE multicast all-gather
supports a single domain. The rank layout is relative to the FSDP shard process
group, including with HSDP; it is not necessarily global rank order. This version
requires equal domains of 2-72 ranks and a domain-major rank layout. It does not
discover topology automatically. Place the job and specify the layout correctly.

For a full local partial gradient `gradient.view(num_domains, ranks_per_domain, chunk)`:

```python
# Exchange within each domain; each lane receives all local contributions.
received = intra_domain_all_to_all(gradient)  # [source_lane, destination_domain, chunk]
partial = local_sum(received, dim=0)           # [destination_domain, chunk]
# Exchange among corresponding lanes in different domains.
received = inter_domain_all_to_all(partial)   # [source_domain, chunk]
output = local_sum(received, dim=0)            # [chunk]
```

After the first sum each rank owns `1 / ranks_per_domain` of the gradient.
The second sum produces its final `1 / fsdp_degree` shard. For one domain, the
second exchange and sum are omitted. Local summation uses fixed source order
and FP32 compensated accumulation; inter-domain partials remain FP32. Results
can differ from NCCL's reduction tree due to floating-point rounding.

## Synchronization and memory

All ranks must follow the same allocation and collective order. The first
barrier protects receive-buffer reuse; the second publishes completed CE writes.
A local stream wait cannot replace this cross-rank protocol. FSDP records the
completion events; these adapters enqueue work on its current communication
stream and return no separate `Work` object, including for async unshard.

Warm up the complete workload and prefetch schedule before graph capture.
Registering an unseen symmetric allocation during capture raises an error.
Reduce-scatter needs an additional receive buffer as large as the full local
gradient bucket, plus FP32 partial buffers when using multiple domains.

Callers can pass a `CEWorkspace` to use bounded exchange storage. Reduce-scatter
then processes the gradient in chunks, preserving source order and FP32
accumulation. All-gather accepts chunks backed by the same workspace. All users
of a shared workspace must enqueue copies, reductions, and copy-out on one
ordered communication stream. GraphTrainer uses this path with one shared
workspace per FSDP group/device, sized by `compile.ce_workspace_size_mib`
(128 MiB by default). The FSDP2 hooks retain their existing allocation behavior.

## Fake process groups

The CE trainer runs under `comm.backend = "fake"` and `"real_pp_fake_spmd"`
for example:

```bash
NGPU=128 COMM_BACKEND=fake MODULE=my_recipes CONFIG=my_ce_recipe ./run_train.sh
```

On a fake FSDP group, buffers still come from the symmetric pool, and the
reduce-scatter scratch, FP32 partials, and local sums are real, so fake memory
reports include CE's allocations. Rendezvous, barriers, and peer copies are
skipped. Outputs match the fake backend's NCCL collectives: all-gather copies
the local shard into every slot, and reduce-scatter returns this rank's own
chunk (divided by the group size for AVG). With the same recipe, CE and NCCL
fake runs produce the same loss. The warm-up-before-capture checks still apply.
Multicast support, NVLink domain layout, and transport are not validated.

## Validate

```bash
pytest torchtitan/experiments/ce_collectives/tests/test_routing.py
pytest torchtitan/experiments/ce_collectives/tests/test_fake.py  # one GPU
torchrun --standalone --nproc-per-node=4 torchtitan/experiments/ce_collectives/tests/test_distributed.py
```

The distributed check exercises BF16/FP16/FP32 reduction, SUM/AVG, uneven chunk
lengths, changing inputs on graph replay, both domain layouts, and FSDP gradients
and SGD updates with prefetch against an unsharded reference. The all-gather
trace must contain one payload copy and two kernels with grid `[1, 1, 1]`.
`nccl_debug`, `ce_debug`, and `ce_two_domain_debug` provide matching 12-step C4
training runs. A logical two-domain test inside one NVLink domain checks routing
and ordering; it does not validate transport between physical NVLink domains.

For GraphTrainer, use `graph_nccl_debug`, `graph_ce_all_gather_debug`, and
`graph_ce_debug` from `torchtitan_recipes.ce_collectives`. These use the same
12-step C4 recipe and compiler settings. All-gather preserves numerics;
reduce-scatter uses a different accumulation order from NCCL.

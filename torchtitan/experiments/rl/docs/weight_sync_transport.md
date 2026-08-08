# Weight-sync transport

This document covers the two per-run dials that steer the trainer->generator
weight sync -- which TorchStore transport carries it, and whether the
generator's GET lands in host memory -- plus how to confirm from the logs what a
run actually did. For the flags alone, see the "Weight-sync transport" section
of `torchtitan/experiments/rl/README.md`.

## Selecting a transport

Which TorchStore transport carries the trainer->generator weight sync is a config
dial, not an env ritual:
`--weight-sync-transport {auto,gloo,monarch_rdma,monarch_rpc,torchcomms}`
(`Controller.Config.weight_sync_transport`).

`auto` leaves torchstore's per-transfer availability cascade in place: SharedMemory
for same-host, then TorchComms, MonarchRDMA, Gloo cross-host. Any other value pins
every transfer to that transport, same-host ones included, which is what lets you
measure or bisect one transport end to end. Pinning is therefore not a production
setting: it gives up SharedMemory for the colocated trainer PUT, so its throughput
is not a perf figure.

Pair it with `--generator.manual-cpu-stage-weight-sync` where registering GPU memory as
the RDMA destination fails: the GET then lands in host memory and is copied H2D, so no
GPU memory is registered at either end. Neither knob is set by any checked-in config;
both are per-run choices.

That second flag is an escape hatch: where the destination buffer lives is torchstore's
decision, since only it knows which transport a transfer resolved to. Gloo already
stages a CUDA destination through host memory internally; MonarchRDMA and TorchComms
register whatever device the tensor is on. So the flag only does something under those
two, or under `auto` when it resolves to one of them -- under a `gloo` or `monarch_rpc`
pin it is pure cost, and the controller warns.

## Confirming what a run did

**Acceptance check.** The controller logs the dials it was given, once, at INFO:

```
[weight-sync] weight_sync_transport=auto manual_cpu_stage_weight_sync=True
```

That is the only record of what was requested. In particular torchstore's span name
reads `cpu_staged` for both generator GET branches -- it is built from torchstore's own
`direct_rdma` argument, a different axis that this path leaves False either way -- so
the span cannot tell you whether staging was on.

For what the transport layer actually did, grep torchstore's own resolution line:

```
[ts-transport] resolved=<Name> (uniflow=..., tc_rdma=..., monarch_rdma=..., gloo=..., shm=...)
```

`<Name>` is `TransportType.name`, so a torchcomms resolution always prints
`resolved=TorchComms` (`TorchCommsRDMA` is an alias of the same enum value). The line
is INFO on logger `torchstore.transport`, which the RL actors quiet by default because
it fires on every op; set `TORCHSTORE_LOG_LEVEL=INFO` to keep it. One line per op, so
dedupe:

```bash
grep -ho 'resolved=[A-Za-z]*' slurm_<jobid>_*.out | sort -u
```

A pinned `torchcomms` raises `RuntimeError("TorchComms transport is not available.")`
when torchcomms is missing rather than sliding to MonarchRDMA, so it cannot yield a
mislabeled data point -- but it can also fail at init rather than at transfer, which
kills the run before any weight sync happens.

## GB300/CoreWeave

Run `auto` plus `--generator.manual-cpu-stage-weight-sync`.
Registering vLLM's live GPU params as the RDMA destination is what fails cross-node
there; staging means that registration never happens, while `auto` still resolves
same-host transfers to SharedMemory and cross-host ones to MonarchRDMA over host
memory. Validated on a 2-node run:

```bash
MONARCH_RDMA_IBVERBS_TARGET=nic:ibp0p0 \
python -m torchtitan.experiments.rl.slurm_launcher \
    --module alphabet_sort --config <config> \
    --generator.manual-cpu-stage-weight-sync
```

The NIC pin is not optional. The fabric has 4 disjoint planes (in `ibp<X>p<Y>`, `X` is
the rail / NIC card and `Y` the plane) and a port reaches only ports in its own plane.
monarch selects a host-memory NIC by hashing each region's own address, which is
plane-blind, so the two ends of a transfer choose independently and mismatch most of the
time, failing with `IBV_WC_RETRY_EXC_ERR` at completion polling rather than at connect.
Pinning every endpoint to one NIC collapses them into a single plane, at the cost of
running on one port instead of spreading across the cards.

`gloo` remains the no-configuration fallback: TCP over the CPU, no NIC pin and no plane
reasoning. Reach for it to get a first run going or to rule the fabric out of a bug.

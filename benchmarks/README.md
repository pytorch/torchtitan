We welcome the community to submit reproducible benchmarking results.

## Collective microbenchmarks

To compare regular NCCL, NCCL symmetric kernels (SymK), and TorchTitan's
custom symmetric-memory all-reduce on one node:

```bash
source .venv/bin/activate
export LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
  benchmarks/collectives/all_reduce_symm_mem.py \
  --sizes 64k 1m 4m 8m 16m 32m \
  --dtype bfloat16
```

The benchmark prints CSV rows from rank 0. NCCL may also print banners or
warnings, so capture the full log and extract the CSV rows if needed:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
  benchmarks/collectives/all_reduce_symm_mem.py \
  2>&1 | tee all_reduce_symm_mem.log

rg '^(# world_size|bytes,|[0-9]+,)' all_reduce_symm_mem.log \
  > all_reduce_symm_mem.csv
```

To verify that NCCL actually selects a symmetric kernel, use a short diagnostic
run with tuning logs enabled (do not use this mode for performance numbers):

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=REG,TUNING \
NCCL_DEBUG_FILE=/tmp/nccl_symk.%h.%p.log \
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
  benchmarks/collectives/all_reduce_symm_mem.py \
  --sizes 4m --warmup 0 --repeats 1 --iterations 1

rg 'AllReduce \[Symmetric\]' /tmp/nccl_symk.*.log
```

## Submission Guidelines

A submission should be a file / files including the following information

1. Entity, which could be your name, GitHub username, company, university, team, etc.
2. The model or theme of benchmarking, e.g. Llama 3.1, Async TP.
3. The hardware setup, including the types of GPUs, interconnections, etc.
4. The actual performance report with training configs, e.g. via
   - Python config files / commandline arguments
   - complete configs, which can be found in the log with [`--print_config`](https://github.com/pytorch/torchtitan/blob/e7c0cae934df78d6e9c2835f42ff1f757dc3fddc/torchtitan/config_manager.py#L47) turned on (preferred as the default value not shown in config files or specified in commandline could change from time to time)
5. The versions and date/time of `torchtitan`, `torch`, `torchao`, or any relevant dependencies.
6. Other notes which could help reproduce the results.

The name of the file should follow the format of
```
[model/theme]_[hardware]_[date/time]_[entity].md
```
For example, `llama3.1_h100_202412_pytorch.md`, `asynctp_256xh100_20250613_alice+bob.md`.

An example can be found at [llama3_h100_202412_torchtitan.md](./llama3_h100_202412_torchtitan.md).

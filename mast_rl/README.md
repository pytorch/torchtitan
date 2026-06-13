# Search-R1 RL on MAST

Submit the torchtitan Search-R1 (multi-turn retrieval GRPO) experiment to MAST,
reusing the Monarch + conda-pack flow from `fbsource//fbcode/pytorch/torchtitan/fb/mast_rl`,
adapted for this OSS checkout. **Nothing needs to land in fbsource** — the conda
env is built from this repo (`pip install .`) and packed to the MAST node.

## What's different from a normal RL job

Search-R1 needs a **dense-retrieval server**. There is no shared retrieval
service reachable from MAST, so we **co-locate** it: `run.sh` starts the
FastAPI retriever on the MAST host (reserving the last GPU for the e5 encoder),
waits for `http://127.0.0.1:8000` to be healthy, then launches training. The
config's `search_url` default (`127.0.0.1:8000`) already points there.

All inputs are read **offline** from the mounted Manifold bucket
`torchtrain_datasets` (mounted at `/mnt/torchtrain_datasets` on MAST). They are
already staged under `tree/yichuan/`:

| Asset | Path on MAST |
|-------|--------------|
| faiss index | `/mnt/torchtrain_datasets/tree/yichuan/search-r1-index/e5_Flat.index` |
| corpus | `/mnt/torchtrain_datasets/tree/yichuan/search-r1-index/wiki-18.jsonl` |
| train/test parquet | `/mnt/torchtrain_datasets/tree/yichuan/Search-R1/data/nq_hotpotqa_train/{train,test}.parquet` |
| e5 model (HF cache) | `/mnt/torchtrain_datasets/tree/yichuan/hf/hub/models--intfloat--e5-base-v2` |
| Qwen3-0.6B | `/mnt/torchtrain_datasets/tree/shuhuay/qwen/Qwen3-0.6B` |

`run.sh` wires these via env (`SEARCH_R1_TRAIN_PARQUET`, `SEARCH_R1_TEST_PARQUET`,
`HF_HOME`, `RETRIEVER_INDEX`, `RETRIEVER_CORPUS`), all overridable. The parquet
paths feed the rollouter through the `SEARCH_R1_*_PARQUET` env hooks added to
`examples/search_r1/rollouter.py`.

## Setup (once)

```bash
conda create -n rlmast python=3.12 -y
conda activate rlmast
bash mast_rl/build_conda.sh        # cu130 torch/vLLM + retriever deps + torchtitan (this checkout) + monarch/torchx fbpkgs
```

## Submit

```bash
conda activate rlmast

# 0.6B smoke (4 train/gen GPUs + 1 retriever GPU). Defaults to Qwen3-0.6B.
bash mast_rl/submit.sh

# 1.7B eval run
bash mast_rl/submit.sh \
    --config rl_grpo_qwen3_1_7b_search_r1 \
    --hf_assets_path /mnt/torchtrain_datasets/tree/shuhuay/qwen/Qwen3-1.7B
```

`submit.sh` reinstalls torchtitan from this checkout (skip with `--no-reinstall`),
then calls `launcher.py`, which conda-packs the active `rlmast` env and ships
`mast_rl/` as the workspace. It prints the MAST job URL.

## Retriever knobs (env, read by run.sh)

| Env | Default | Meaning |
|-----|---------|---------|
| `RETRIEVER_GPU` | last GPU index | GPU for the e5 encoder |
| `RETRIEVER_FAISS_GPU` | `0` | `1` puts the 64GB Flat index on the GPU; `0` keeps it in CPU RAM (the dev-server README's proven mode — encoder still uses the GPU) |
| `RETRIEVER_TOPK` | `3` | passages per query |
| `RETRIEVER_TIMEOUT` | `2400` | seconds to wait for index load (reading 64GB from the bucket is slow) |
| `SEARCH_R1_STAGE_ROOT` | `${MOUNT_POINT}/tree/yichuan` | base dir for staged assets |

## Debugging

* Retriever log on the MAST host: `/tmp/retrieval_server.log` (tail'd to the
  job log on startup failure).
* If the index load from Manifold is slow, raise `RETRIEVER_TIMEOUT` or stage
  the index to local NVMe in `run.sh` before starting the server.

## Local sanity-check (not via this dir)

To validate on the dev server first, follow `torchtitan/experiments/rl/examples/search_r1/README.md`:
start the local retriever, then
`python torchtitan/experiments/rl/train.py --module rl --config rl_grpo_qwen3_0_6b_search_r1 --metrics.no-enable-wandb --num-steps 3`.

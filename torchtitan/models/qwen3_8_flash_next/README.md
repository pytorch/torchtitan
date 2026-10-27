# Qwen3.8-Flash-Next

This package implements the released text and vision backbone of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), following
the `add_model` workflow in [PR #5636](https://github.com/pytorch/torchtitan/pull/5636).
The checkpoint revision is `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
The reference is Transformers' `Qwen4Exp` implementation at
`494470d677d05d44153d235d795e0d01040307ab`.

## Architecture

Compared with the existing Qwen3.8 MoE port, Flash-Next adds four gated residual
streams, per-layer lexical embeddings (PLE), and Qwen Sparse Attention (QSA).
It reuses Qwen3.5's Gated DeltaNet, MoE, multimodal preprocessing, MRoPE, vision
encoder, checkpoint conversion, and training lifecycle.

| Component | Released configuration |
| --- | --- |
| Decoder | 48 layers, dimension 2560, three GDN layers per QSA layer |
| Residual streams | Four, grouped offset RMSNorm, mixing rank 320 |
| QSA | 24 query heads, 2 KV heads, dimension 256, rotary dimension 64 |
| Indexer | Four query heads, one KV head, dimension 128, compress ratio 4 |
| Selection | Top 512 complete blocks (2048 tokens), plus the current partial block |
| MoE | 512 experts, top 10, dimension 640, one gated shared expert |
| GDN | 16 key heads, 48 value heads, key/value dimension 128, sigmoid output gate |
| PLE | Decoder layer 2, bigram/trigram hashes, 16 hash heads, dimension 2560 |
| PLE table | 320001536 rows x 160 features, 128 HF checkpoint chunks |
| Vision | 27 layers, dimension 1152, 16 heads, patch size 16, temporal patch size 2 |
| Vocabulary | 248320, EOS 248044 |

The `125B-A6B` flavor follows the release's size label. Meta construction counts
177392830576 native parameters, including 51200245760 lexical table parameters
and 448931056 vision parameters. The reference excludes the MTP draft head:
the adapter loads all 1627 backbone/vision checkpoint entries and warns when it
encounters the 31 unused `mtp.*` entries. MTP training and speculative decoding
are outside this port's scope.

QSA pools raw indexer keys before normalization and rotary embedding. Selection
respects document boundaries, padding, and causality. Its training objective
uses the technical report's sparse-stage KL target: sum the detached main
attention probabilities over heads, max-pool each complete block, and normalize
over the selected blocks. Heads are summed across TP before max pooling. The
report does not specify its coefficient; the flavor uses `1.0`, independently
of the MoE load-balancing coefficient `0.001`.

PLE hashes reset at EOS and packed document boundaries. Its dilated causal
convolution resets at document boundaries. Original token IDs and document
positions travel to later pipeline stages separately from multimodal rotary
coordinates. The lexical table uses TP across features so exporting the HF row
chunks preserves sharding without gathering the full table on every rank.

## Usage

```python
import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config

config = build_model_config(
    "125B-A6B", seq_len=4096, parallelism=ParallelismConfig()
)
with torch.device("meta"):
    model = config.build()
```

The reduced `debugmodel` retains GDN, QSA, GR, PLE, MoE, and optional vision.
Importable training fixtures live in
`torchtitan_recipes.tests.models.qwen3_8_flash_next`:

| Config | GPUs | Coverage |
| --- | --- | --- |
| `qwen38_flash_next_debugmodel` | 1 | Ten training steps, seed 42, strict deterministic execution |
| `qwen38_flash_next_debugmodel_tp2` | 2 | TP2, EP2, sequence parallelism, SPMD type checking |
| `qwen38_flash_next_debugmodel_all_parallelism` | 8 | FSDP2, TP2, EP4, PP2, two microbatches, PLE on stage 2 |
| `qwen38_flash_next_debugmodel_pp2` | 2 | Original token metadata across a real pipeline boundary |
| `qwen38_flash_next_debugmodel_ac` | 1 | Full activation rematerialization and auxiliary-loss lifecycle |
| `qwen38_flash_next_debugmodel_mm` | 1 | CC12M images, vision, MRoPE, and PLE, sequence length 1024 |

For example:

```bash
MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
CONFIG=qwen38_flash_next_debugmodel_tp2 NGPU=2 ./run_train.sh
```

Use the generic converter for downloaded HF weights:

```bash
python -m scripts.checkpoint_conversion.convert_from_hf \
  /path/to/Qwen3.8-Flash-Next /path/to/native-checkpoint \
  --model_name qwen3_8_flash_next --model_flavor 125B-A6B
```

## Numerical evidence

The reproducible same-weight reference harness is
`scripts.checkpoint_conversion/numerical_tests_qwen3_8_flash_next.py`.
[Its report](../../../tests/assets/numerics/qwen3_8_flash_next_reference.json)
records revisions, input IDs, dtypes, backend, seed, and tensor tolerances.

| Comparison | Maximum absolute error |
| --- | --- |
| GR read/write, input and parameter gradients, FP32 | 0 |
| PLE forward, FP32 | 7.45e-9 |
| PLE input and parameter gradients, FP32 | 3.19e-12 |
| QSA forward, FP32 | 4.10e-8 |
| Vision including merger, FP32 | 3.28e-7 |
| Full eight-layer logits, BF16 | 0.001953125 |

Every layer chooses the same experts as the reference. Logit cosine similarity
is `0.9999963898`, mean per-token KL(reference || native) is `8.10e-8`, and top-1
and top-5 sets agree for all fixture tokens. The tensor check uses BF16
half-epsilon absolute tolerance plus one epsilon relative tolerance; the FP32
checks cover operation-order rounding. New CPU tests also use independent
functional oracles for GR, hashing, PLE convolution, selected attention, KL,
router rounding, gradients, packed inputs, and optimizer/remat resets.

Packed BF16 execution can round differently from separate document invocations:
two near-tied tokens change expert sets in this fixture, giving an unconstrained
maximum logit difference of `0.02587890625`. Replaying only the isolated expert
IDs and recomputing the packed mixing weights reduces it to `0.00390625`.
That replay diagnoses boundary semantics; it is not a claim of bitwise equality
for unconstrained BF16 routing. CPU boundary tests check the math independently.

Local GB300 validation uses PyTorch `2.16.0.dev20261006+cu132`, CUDA 13.2,
seed 42, and strict deterministic execution. Ten-step plain/remat runs match
every full-precision TensorBoard loss and grad_norm. The TransformerBlock
lifecycle change also preserves every value exactly. The TP2/EP2/SP fixture
matches every loss and grad_norm from the remote GB200 repeats. The PP2 and
multimodal fixtures complete ten steps; the image fixture uses sequence length
1024 so image tokens fit and the vision encoder participates in training.

Independent remote launches `qwen38-flash-next-numerics-k-tv4bvsmb` and
`qwen38-flash-next-numerics-l-mcss09v9` recreate the CPU FP32 seed checkpoint,
then run ten TP2/EP2/SP optimizer steps on separate GB200 hosts. Both complete
with identical full-precision loss and grad_norm at every step. GDN's L2-norm
backward autotuner previously chose different FP32 reduction orders across
processes. The shared GDN chunk, CP and recomputed backward paths now pass
`autotune=False` in strict deterministic mode, matching KDA's existing policy.
This intentionally fixes deterministic-mode backward rounding; normal-mode
autotuning and checkpoint layouts are preserved. Reference parity is unchanged,
and the shared GDN GPU forward/backward suite passes all nine tests in strict
mode.

The real eight-GPU MAST run `qwen38-flash-next-parallel-d-pxnh2kqn` completes
ten optimizer steps on four GB200 hosts with FSDP2, TP2, EP4 and PP2 active
together. The stage split puts `layers.0` on stage 1 and `layers.1` through
`layers.7`, the output mixer, and LM head on stage 2. Its output is preserved
at `torchtrain_datasets/tree/shuhuay/qwen38-flash-next/parallel-d` in Manifold.
The H100 CI suite selects the same factory, plus the PP2, remat and multimodal
fixtures. The original MAST PyTorch 2.15 image predates the pipeline
`loss_kwarg_mbs` API used by upstream/main; the validated run uses the same
PyTorch 2.16 build as local validation.

The `b200` suite registers `qwen3_8_flash_next_tp_ep_numerics` against the
[GB200 real-PG golden](../../../tests/assets/losses/real_pg/qwen3_8_flash_next_gb200.txt).
It covers TP2/EP2/SP with SPMD type checking, GDN, QSA block pruning, GR, PLE,
and both auxiliary objectives. The eight-GPU fixture covers combined training;
the golden uses a separate reduced topology. The golden was exported through
the integration runner only after the independent remote repeatability check.
It records the source snapshots, GPU, software, initialization, optimizer,
schedule, data and tokenizer hashes. Replay it on the recorded GB200 environment:

```bash
python -m tests.integration_tests.run_tests outputs/flash-next-replay \
  --test_suite b200 --test_name qwen3_8_flash_next_tp_ep_numerics \
  --execution_mode real_pg --gpu_arch_type cuda --gpu_arch b200 \
  --ngpu 2 --no-parallel
```

The independent MAST replay `qwen38-flash-next-numerics-replay-n-s2744wlm`
imports the committed golden without export and passes the exact loss and
grad_norm assertions for all ten steps. Its persistent output directory is
`torchtrain_datasets/tree/shuhuay/qwen38-flash-next/numerics-replay-n`.
All pre-commit hooks pass; 146 targeted CPU tests and nine shared GDN GPU tests
pass. The CPU coverage includes model/config contracts, the numerical recipe,
checkpoint conversion, shared embedding/normalization/RoPE and CP attention.

A 100-step reduced-model run on the `allenai/c4` English training stream with
the official tokenizer from the pinned checkpoint revision decreases mean loss
from `12.507574` (steps 1-10) to `7.963772` (steps 91-100). These convergence
values come from stdout rounded to five decimal places. Reproduce its config
by deriving from `qwen38_flash_next_debugmodel(seq_len=256)`, setting 100 steps,
pointing `hf_assets_path` at the pinned tokenizer, and selecting
`ConcatThenSplitPackingConfig(dataset=DATASETS["c4"])` on the Grain data loader.

## Execution limits

QSA currently uses bounded indexer/gather chunks and native SDPA MATH. CUDA
graphs are disabled in the fixtures because block selection has dynamic shapes;
context parallelism is rejected explicitly. The complete released checkpoint
schema is checked on meta tensors and reduced checkpoints load strictly, but
full 177B parameter materialization and production-scale convergence have not
been validated. The reduced-model runs establish correctness and training
behavior, and do not establish production throughput.

On the pinned PyTorch nightly, the shared CosSinRoPE eager/compile test has the
same preexisting failure on unchanged upstream/main: one of 65536 BF16 gradient
values differs by `9.54e-7`. The pinned CUTLASS DSL 4.6.2 environment supports
the validated fused GDN backend; the optional cuDNN backend requires DSL 4.7.0.

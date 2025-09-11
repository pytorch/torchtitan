# BlendCorpus Integration with TorchTitan

This document summarizes all code and configuration changes introduced to train on BlendCorpus with TorchTitan, including tokenizer support, dataloader integration, trainer hooks, and configuration schema additions.

⸻

## 1) Tokenizer Support

### 1.1 Added SentencePiece tokenizer adapter

**File**: ``torchtitan/experiments/blendcorpus/dataset/sptoken.py``

**What**: New SPTokenizer wrapper around sentencepiece.SentencePieceProcessor.

**Key features**:
* Attributes mirroring Titan’s expectations:
* n_words (alias of SP vocab size), vocab_size
* bos_id, eos_id, pad_id, unk_id (−1 if absent)
* API compatible with Titan loaders:
    ```python
    ids = tokenizer.encode(text, bos=True, eos=True)
    text = tokenizer.decode(ids)
    ```

### 1.2 Added tokenizer router

**File**: ``torchtitan/experiments/blendcorpus/dataset/build_tokenizer.py``

**What changed**:
* build_tokenizer(job_config) now returns a tokenizer instance, not a function.
* Accepts common aliases:
    - SentencePiece: "sptoken"
    - HuggingFace: "hf"

```python
from torchtitan.experiments.blendcorpus.dataset.sptoken import build_sentencepiece_tokenizer
from torchtitan.components.tokenizer import build_hf_tokenizer

def build_tokenizer(job_config):
    backend = str(getattr(job_config.model, 'tokenizer_backend', '')).strip().lower()
    if backend in {'', 'sptoken', 'sentencepiece', 'sp', 'spm'}:
        print('[Tokenizer] Using backend: sptoken (SentencePiece)')
        return build_sentencepiece_tokenizer(job_config)
    if backend in {'huggingface', 'hf'}:
        print('[Tokenizer] Using backend: huggingface (HF AutoTokenizer)')
        return build_hf_tokenizer(job_config)
    if backend == 'tiktoken':
        raise NotImplementedError("tokenizer_backend='tiktoken' is not supported for this training recipe; use 'sptoken'.")
    raise Exception(f"Unknown tokenizer_backend '{backend}'. Choose 'sptoken' or 'hf'.")
```

In configuration manager, we extended the following option: 
```python
@dataclass
class Model:
    tokenizer_backend: str = "sptoken"
```

## 2) Dataloader Integration for BlendCorpus

### 2.1 New external adapter

**File**: ``torchtitan_ext/datasets/blendcorpus_builder.py`` (new)
**What**: Bridges BlendCorpus datasets to Titan’s expected batch format.

**Responsibilities**:
* Build train/valid/test datasets from a file list (one shard path per line) or directory.
* Collate batches to Titan’s required structure:
* Return a tuple (input_dict, labels), where
   - input_dict["input_ids"] is LongTensor [B, L]
   - (optionally) input_dict["attention_mask"]
   - labels is LongTensor [B, L]
* Ensure the training stream is effectively infinite (no StopIteration killing the loop).

Implementation highlights: ``set_consumed_by_global_step(step, global_batch_size)`` to retarget the data stream after checkpoint restore (see §3.2).

## 3) Trainer Wiring

### 3.1 Selecting BlendCorpus as the dataset

**File**: ``torchtitan/experiments/blendcorpus/train.py``

**What**: At dataloader construction time, detect the dataset and route accordingly.

**Example logic**:
```python
    train_dl, valid_dl, test_dl = self.train_spec.build_dataloader_fn(self.config, global_batch_size)
    self.dataloader = train_dl
    self.eval_dataloader = valid_dl
    self.test_dataloader = test_dl
    utils.logger.info("Using BlendCorpus dataloader (torchtitan_ext).")
else:
    self.dataloader = self.train_spec.build_dataloader_fn(...)
```
### 3.2 Advancing the dataloader after checkpoint restore

**File**: ``torchtitan/experiments/blendcorpus/train.py``

**What**: After ``self.checkpointer.load(...)`` restores self.step, advance BlendCorpus’ dataloader to the correct consumed samples.

**Hook**:
```python
self.checkpointer.load(step=job_config.checkpoint.load_step)
try:
    ds_name = getattr(getattr(self.job_config, "training", object()), "dataset", "").strip().lower()
    if ds_name == "blendcorpus" and hasattr(self.dataloader, "set_consumed_by_global_step"):
        self.dataloader.set_consumed_by_global_step(self.step, self.global_batch_size)
        logger.info(f"BlendCorpus dataloader advanced to consumed={self.step * self.global_batch_size} samples (step={self.step}).")
except Exception as e:
    logger.warning(f"BlendCorpus retarget hook failed: {e}")
logger.info(f"Training starts at step {self.step + 1}.")
```

Why: Guarantees data position matches checkpointed progress:
consumed = step * global_batch_size.


## 4) Config Schema (ConfigManager.py)

### 4.1 New [blendcorpus] TOML section and dataclass

**File**: ``torchtitan/experiments/blendcorpus/job_config.py``

**What**: Introduced a new dataclass and a field on JobConfig to expose BlendCorpus-specific knobs directly in the parsed config.

Dataclass added (verbatim):

```python

@dataclass
class BlendCorpus:
    """Optional settings specific to the BlendCorpus data loader.

    These map to a TOML section named [blendcorpus]. If present, your
    adapter can read them from cfg.blendcorpus.*
    """

    # File list or directory of shards
    data_file_list: str | None = None
    """Path to a text file with one shard path per line, or a directory."""

    # Sequence length and batching (can override training.seq_len / local_batch_size)
    seq_length: int | None = None
    """Optional override for sequence length. If None, use training.seq_len."""

    micro_batch_size: int | None = None
    """Optional override for per-rank batch size. If None, use training.local_batch_size."""

    # Loader behavior
    num_workers: int = 2
    """Number of DataLoader workers."""

    split: str = "98,1,1"
    """Train/valid/test split in percentages."""

    dataloader_type: str = "single"
    """Loader type hint (e.g., 'single', 'repeating')."""

    shuffle_sample_in_corpus: bool = True
    """Whether to shuffle samples within corpus."""

    blend_sample_in_corpus: bool = True
    """Whether to shuffle samples within corpus."""

    append_eod: bool = True
    """Append EOD token at the end of each sample when collating."""

    provide_attention_mask: bool = False
    """Whether the adapter should compute and return attention masks."""

    eod_token_id: int | None = None
    """Optional explicit EOD token id; if None the adapter/tokenizer decides."""

    data_cache_path: str = None
# --- END BlendCorpus dataclass ---
```

## 5) Configuration Examples

Below is a minimal TOML (SentencePiece + BlendCorpus)

```toml
[model]
name = "llama3"
flavor = "debugmodel"               # or your 1B flavor
tokenizer_backend = "sptoken"       # aliases: sentencepiece, sp, spm
tokenizer_path = "/home/USER/blendcorpus/preprocess/tokenizer/tokenizer.model"

[training]
dataset = "blendcorpus"
dataset_path = "/home/USER/datasets/blend/file_list.txt"
seq_len = 8192
local_batch_size = 1
# global_batch_size derives from local_batch_size × replicas

[blendcorpus]
# Optional overrides / extras
num_workers = 2
split = "98,1,1"
dataloader_type = "single"
shuffle = true
append_eod = true
provide_attention_mask = false
# eod_token_id = 128001

[parallelism]
data_parallel_replicate_degree = 4
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
pipeline_parallel_degree = 1
context_parallel_degree = 1
```


## 6) Running
```bash
NGPU=1 CONFIG_FILE="./torchtitan/experiments/blendcorpus/train_configs/debug_model.toml" ./torchtitan/experiments/blendcorpus/run_train.sh 
```

## 7) Experiments Directory

The `torchtitan/experiments/blendcorpus/` folder now contains ready-to-run experiment configurations, logs, and scripts to reproduce BlendCorpus training with TorchTitan.

Typical contents include:
* Example TOML config files for various scales (debug, 1B scale, etc.)
* Run scripts adapted for JLSE/Vast paths to facilitate seamless execution
* Logs and checkpoints for verification and analysis of training runs

Users can start from these provided configs as templates to customize and launch their own BlendCorpus training experiments.
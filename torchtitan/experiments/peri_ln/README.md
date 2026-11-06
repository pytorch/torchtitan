# Peri-LN

Owner: [@prinwinter](https://github.com/prinwinter).

This experiment applies
[Peri-LN (Kim et al., ICML 2025)](https://arxiv.org/abs/2502.02732)
to TorchTitan's Llama 3 model. It normalizes the input and output of each
attention and feed-forward branch, leaving the residual connection outside
the output normalization:

```text
h = x + attention_output_norm(attention(attention_norm(x)))
y = h + ffn_output_norm(feed_forward(ffn_norm(h)))
```

The motivation is to control the scale of each branch's contribution to the
residual stream while preserving the identity gradient path. This is the
placement in Section 3.3 of the paper, using RMSNorm. The optional initial
embedding normalization is omitted in this minimal experiment; Llama's final
RMSNorm is retained. These recipes demonstrate the architecture, not a
reproduction of the paper's model sizes or training hyperparameters.

## Implementation

`model.py` extends `Llama3TransformerBlock` with two independent output norms.
`apply_peri_ln` derives a separate model config from a Llama 3 config, preserving
attention, RoPE, SwiGLU, initialization, and sharding settings. Each new norm
copies its corresponding input norm's epsilon, unit-scale initialization, and
TP/SP placement. With model dimension `D` and `L` blocks, this adds `2 * L * D`
trainable parameters.

The experiment reuses the existing decoder forward, Trainer, optimizer,
scheduler, data loader, loss, checkpointing, and distributed infrastructure.
The output norms declare their activation recomputation dependencies using
the existing `torch_remat` convention. No core model or training code changes
are required, and creating a Peri-LN config does not mutate the baseline.

## Run

Use the repository's supported Linux/CUDA training environment and install
the dependencies described in the [root README](../../../README.md).
Run commands from the repository root. No additional dependencies are needed.

The small recipe uses the checked-in C4 fixture and tokenizer, FlexAttention,
seed 42, deterministic algorithms, and ten training steps. CUDA graphs are
disabled for this smoke run:

```bash
NGPU=1 MODULE=torchtitan.experiments.peri_ln.recipes \
  CONFIG=llama3_debugmodel ./run_train.sh \
  --output-dir ./outputs/peri_ln_debug
```

For an 8B experiment, obtain the tokenizer assets (access to the gated Llama
repository is required):

```bash
python scripts/download_hf_assets.py \
  --repo_id meta-llama/Llama-3.1-8B --assets tokenizer

NGPU=8 MODULE=torchtitan.experiments.peri_ln.recipes \
  CONFIG=llama3_8b ./run_train.sh \
  --output-dir ./outputs/peri_ln_8b
```

The 8B recipe inherits the standard C4 data pipeline and optimizer settings,
uses a context length of 2048, and trains from scratch. GPU memory requirements
depend on the launch configuration; this example is not a measured benchmark.

To customize a run or compare against Pre-LN, place Python recipe functions in
an importable module, for example `peri_ln_comparison.py` at the repository root:

```python
from torchtitan.experiments.peri_ln import apply_peri_ln
from torchtitan_recipes.models.llama3 import llama3_8b


def pre_ln():
    config = llama3_8b(seq_len=2048)
    config.debug.seed = 42
    config.debug.deterministic = True
    config.training.steps = 1000
    return config


def peri_ln():
    config = pre_ln()
    config.model = apply_peri_ln(config.model)
    return config
```

```bash
NGPU=8 MODULE=peri_ln_comparison CONFIG=pre_ln ./run_train.sh \
  --output-dir ./outputs/pre_ln_c4
NGPU=8 MODULE=peri_ln_comparison CONFIG=peri_ln ./run_train.sh \
  --output-dir ./outputs/peri_ln_c4
```

Change training settings in the recipe, following TorchTitan's configuration
API. When changing parallelism, construct the base model with the desired
`ParallelismConfig` before calling `apply_peri_ln`, so its sharding contracts
match the run.

## Checkpoints

Native TorchTitan checkpoints include `layers.N.attention_output_norm.weight`
and `layers.N.ffn_output_norm.weight` along with all original Llama parameters.
Resume with the same Peri-LN model configuration and a configured native
checkpointer. The example recipes inherit `checkpointer=None`; enable a
checkpointer in your recipe to save or resume training.

The experiment disables Llama's Hugging Face state-dict adapter because it
cannot represent the additional norm weights. HF model import/export is not
supported. A Pre-LN checkpoint is not a strict Peri-LN checkpoint, and loading
its shared weights would change the model's behavior. Existing Llama configs
and their checkpoint support remain unchanged.

## Tests and evidence

```bash
python -m pytest torchtitan/experiments/peri_ln/tests/test_model.py -v
python -m pytest torchtitan/experiments/peri_ln/tests/test_recipes.py -v
python -m pytest torchtitan/experiments/peri_ln/tests/test_training.py -v
pre-commit run --all-files
```

The CPU tests cover the residual formula and gradients, initialization,
config isolation, checkpoint state, and recipe selection. An isolated
[CPU workflow](../../../.github/workflows/unit_test_cpu_peri_ln.yaml) runs these
tests for changes to this experiment. The training tests exercise the existing
Trainer on CUDA, including tensor/sequence parallelism and activation
recomputation; they skip when the required GPUs are unavailable.

Unit tests and short training runs establish implementation correctness, not
large-scale convergence or speed. Before proposing upstream acceptance, run
the matched Pre-LN/Peri-LN C4 recipes and attach loss and gradient-norm curves,
peak memory, and tokens per second. Measure at least ten steps and account for
warmup. Peri-LN changes computation, so identical loss to Pre-LN is not expected.
No convergence, throughput, or multi-GPU performance result is claimed here.

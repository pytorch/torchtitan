# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation tests for ``estimate_peak_memory_modified``.

These tests build small *dense* models (llama3 and qwen3 non-MoE), trace the
joint forward/loss/backward graph through ``GraphTrainer``, and validate the
static peak estimate against reality. MoE is intentionally out of scope --
dynamic routing shapes make the static estimate harder and are a follow-up.

What "validation" means here -- three independent signals, weakest to strongest:

1. Scalar est-vs-real (smoke test). Put ``estimate_peak_memory_modified`` next
   to ``torch.cuda.max_memory_allocated()`` and check the ratio. This is only a
   *smoke* test, for two reasons we measured directly:
     - Scale dependence. The estimator models graph *tensors* only. The real
       peak also includes cuBLAS/attention kernel workspaces, allocator block
       rounding, and fragmentation -- fixed-ish overheads that dominate a tiny
       model's footprint. So on debugmodels the estimator legitimately
       *under*-predicts (~0.7-0.85x); at scale (the large tests) tensor memory
       dominates and the ratio approaches 1. A loose band is therefore correct
       for small models -- a tight band would just be testing workspace noise.
     - Direction matters. For a memory *budget* solver, under-prediction is the
       dangerous direction (OOM); over-prediction only wastes memory. We log the
       ratio so regressions are visible even when the band passes.

2. Real-peak stability (regime guard). With cuda graphs the same graph reports
   wildly different ``max_memory_allocated`` per step (eager warmup vs graph
   capture vs replay). We disable the cudagraph pass (``disable_passes=
   ["cudagraph_pass"]``) so the measured peak is the compiled-eager fwd/bwd
   peak -- a single, well-defined quantity -- and assert it is stable across
   steps. An unstable real peak means the comparison target is moving and any
   ratio is meaningless.

3. Analytical per-category (the strong, exact check). Parameter storage is
   exactly ``sum(p.numel() * p.element_size())`` over the model -- independent of
   allocator overhead, workspace, and scale. We assert the estimator's
   parameter-bytes category matches that, and that optimizer state is zero
   (``forward_backward_step`` runs no optimizer). This is the assertion that
   actually pins the estimator's correctness.

Run (no pytest in this venv -- use unittest):

    python -m unittest torchtitan.experiments.graph_trainer.tests.\\
        test_memory_estimator.TestMemoryEstimator -v

Cross-test residue: the regional-inductor compile cache retains a prior model's
params+grads across tests (see ``_free_cuda``), inflating a later test's *real*
peak. The scalar band tolerates it; the stability and analytical checks do not
depend on it. For an exact scalar ratio, run a single test per process.
"""

import gc
import os
import unittest

import torch
import torch.nn as nn
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.graph_trainer.llama3 import (
    model_registry as llama3_registry,
)
from torchtitan.experiments.graph_trainer.memory_estimator import (
    estimate_peak_memory_modified,
)
from torchtitan.experiments.graph_trainer.qwen3 import model_registry as qwen3_registry
from torchtitan.experiments.graph_trainer.tests._trainer_test_utils import (
    build_minimal_trainer,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

DTYPE = torch.bfloat16
BATCH_SIZE = 2
SEQ_LEN = 2048
DEBUGMODEL = "debugmodel"

# Scalar smoke-test band (small models). Deliberately loose: at debugmodel scale
# the real peak is dominated by workspace/fragmentation the estimator does not
# model, so the estimator under-predicts (~0.7-0.85x measured). This catches
# gross breakage only; real accuracy is checked by the analytical + large tests.
SMALL_MIN_RATIO = 0.5
SMALL_MAX_RATIO = 1.5

# Real-peak stability tolerance across steps (cudagraph disabled -> should be
# essentially identical step to step; allow tiny allocator jitter).
STABILITY_TOL = 0.02

# Parameter-bytes match tolerance. Exact in principle; small slack absorbs
# 512B-per-tensor allocator rounding and the (tiny) user-input placeholders that
# share the "input" category with parameters.
PARAM_BYTES_TOL = 0.05

# ---- opt-in large tests: reproduce the PR validation table ----
# Same configs used for the PR's est-vs-real table (default memory policy,
# cudagraph disabled, passes applied): llama3 1B and qwen3 1.7B at batch=8,
# seq=2048. A reviewer running these should see the same graph and a near-1.0
# ratio. Run a SINGLE test per process for an exact ratio (see _free_cuda).
RUN_LARGE = os.environ.get("RUN_LARGE_MEM_TEST", "0") == "1"
LARGE_LLAMA3_FLAVOR = os.environ.get("LARGE_LLAMA3_FLAVOR", "1B")
LARGE_QWEN3_FLAVOR = os.environ.get("LARGE_QWEN3_FLAVOR", "1.7B")
LARGE_BATCH = int(os.environ.get("LARGE_BATCH", "8"))
LARGE_LLAMA3_SEQ = int(os.environ.get("LARGE_LLAMA3_SEQ", "2048"))
LARGE_QWEN3_SEQ = int(os.environ.get("LARGE_QWEN3_SEQ", "2048"))
LARGE_MEMORY_POLICY = os.environ.get("LARGE_MEMORY_POLICY", "default")
LARGE_DATASET = os.environ.get("LARGE_DATASET", "c4_test")
LARGE_TOKENIZER_PATH = os.environ.get(
    "LARGE_TOKENIZER_PATH", "./tests/assets/tokenizer"
)
# forward_backward_step runs no optimizer, so real == params + working set and the
# estimator's peak_bytes should match it closely at scale. Band allows for kernel
# workspace and (when not run single-per-process) cross-test cache residue.
LARGE_MIN_RATIO = 0.90
LARGE_MAX_RATIO = 1.15
MIN_GPU_BYTES_FOR_LARGE = 80 * 1e9


def _set_deterministic() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)


def _build_model(registry, flavor: str, attn_backend: str = "sdpa") -> nn.Module:
    model_spec = registry(flavor, attn_backend=attn_backend)
    with torch.device("meta"):
        model = model_spec.model.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=None)
    model.to(dtype=DTYPE)
    model.train()
    return model


def _build_trainer(
    registry,
    *,
    enable_passes: bool = True,
    flavor: str = DEBUGMODEL,
    memory_policy: str = "default",
) -> GraphTrainer:
    """Build a single-GPU GraphTrainer for estimator validation.

    Graph passes are always applied (``apply_graph_passes`` already skips the
    memory-policy / cpu-offload / activation-remat passes, so this is the
    "original" graph plus cleanup + regional inductor). cudagraph is always
    disabled so ``max_memory_allocated`` measures the compiled-eager fwd/bwd peak
    -- a stable, well-defined target. Without that, the measured peak depends on
    which cudagraph phase (warmup/capture/replay) the step lands in.
    """
    model = _build_model(registry, flavor)
    model_config = registry(flavor, attn_backend="sdpa").model
    trainer = build_minimal_trainer(
        model,
        model_config,
        GraphTrainer,
        activation_checkpoint_mode="selective",
        compile_enable_passes=enable_passes,
        compile_disable_passes=["cudagraph_pass"],  # cudagraph always disabled
    )
    trainer.config.compile.memory_policy = memory_policy
    return trainer


def _analytical_param_bytes(model: nn.Module) -> int:
    """Ground-truth resident parameter+buffer bytes: independent of the allocator
    and of scale. The estimator's parameter storage must match this."""
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.device.type != "cuda":
            continue
        sid = t.untyped_storage()._cdata
        if sid in seen:  # tied weights share storage -- count once
            continue
        seen.add(sid)
        total += t.untyped_storage().nbytes()
    return total


def _make_inputs(batch: int, seq: int):
    # randint high (2048) is < every model's vocab size, so tokens are valid.
    tokens = torch.randint(0, 2048, (batch, seq), device="cuda")
    labels = torch.randint(0, 2048, (batch, seq), device="cuda")
    return tokens, labels


def _real_batch(batch: int, seq: int, dataset: str, tokenizer_path: str):
    """One real (input_dict, labels) batch via the trainer's dataloader, on CUDA.
    Mirrors the run_train.sh data path (real tokenized text, IGNORE_INDEX pad)."""
    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=tokenizer_path)
    dataloader = HuggingFaceTextDataLoader.Config(dataset=dataset).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=seq,
        local_batch_size=batch,
        snapshot_every_n_steps=None,
    )
    input_dict, labels = next(iter(dataloader))
    input_dict = {k: v.to("cuda") for k, v in input_dict.items()}
    labels = labels.to("cuda")
    return input_dict, labels


def _run_step(trainer: GraphTrainer, tokens, labels) -> None:
    model = trainer.model_parts[0]
    model.zero_grad(set_to_none=True)
    global_valid_tokens = torch.tensor(labels.numel(), dtype=torch.float, device="cuda")
    positions = (
        torch.arange(tokens.shape[1], device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .expand(tokens.shape[0], tokens.shape[1])
    )
    trainer.forward_backward_step(
        input_dict={"input": tokens, "positions": positions},
        labels=labels,
        global_valid_tokens=global_valid_tokens,
    )


def _run_step_dict(trainer: GraphTrainer, input_dict, labels) -> None:
    """Run one fwd+bwd step from a real dataloader batch (input_dict carries the
    real per-document 'positions'); valid-token count excludes IGNORE_INDEX."""
    model = trainer.model_parts[0]
    model.zero_grad(set_to_none=True)
    global_valid_tokens = (labels != IGNORE_INDEX).sum().to(torch.float)
    trainer.forward_backward_step(
        input_dict=input_dict,
        labels=labels,
        global_valid_tokens=global_valid_tokens,
    )


def _measure_real_peaks(run_step, trainer, *args, warmup: int = 1, measure: int = 3):
    """Run ``warmup`` steps to absorb tracing/autotune, then ``measure`` clean
    steps, returning the per-step ``max_memory_allocated`` (one reset per step).

    Returning every measured step (not just one) lets callers assert the real
    peak is *stable* -- the regime guard. With cuda graphs disabled these should
    be essentially identical.
    """
    for _ in range(warmup):
        run_step(trainer, *args)
    torch.cuda.synchronize()
    # Drop residue from a prior test (another model's weights/grads) so this
    # step's peak is not inflated by unrelated residency.
    gc.collect()
    torch.cuda.empty_cache()

    peaks = []
    for _ in range(measure):
        torch.cuda.reset_peak_memory_stats()
        run_step(trainer, *args)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    return peaks


def _free_cuda() -> None:
    """Best-effort release of the just-finished test's allocations. Does NOT
    fully isolate tests: the regional-inductor compile pipeline retains the prior
    model's params+grads in a global cache that survives reset+gc+empty_cache, so
    a later test's measured ``real`` is inflated by ~the earlier model's
    footprint. Run a single test per process for an exact est/real ratio."""
    try:
        import torch._dynamo as dynamo

        dynamo.reset()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMemoryEstimator(unittest.TestCase):
    def setUp(self):
        _set_deterministic()
        self.tokens = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")
        self.labels = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")

    def tearDown(self):
        torch.use_deterministic_algorithms(False)
        _free_cuda()

    # ---- signal 1: scalar smoke test (loose; workspace dominates at this scale)
    def _smoke(self, name, trainer):
        peaks = _measure_real_peaks(_run_step, trainer, self.tokens, self.labels)
        real = peaks[-1]
        est = estimate_peak_memory_modified(trainer._traced_step.gm)
        ratio = est.peak_bytes / real if real else float("inf")
        print(
            f"\n[{name}] estimate={est.peak_bytes / 1e9:.3f} GB  "
            f"real={real / 1e9:.3f} GB  ratio=est/real={ratio:.3f}"
        )
        print(
            "    per-category at peak: "
            + ", ".join(
                f"{k}={v / 1e9:.3f}GB" for k, v in est.per_category_at_peak.items()
            )
        )
        self.assertGreater(est.peak_bytes, 0, f"{name}: estimate should be > 0")
        self.assertGreater(real, 0, f"{name}: real peak should be > 0")
        self.assertGreater(
            ratio, SMALL_MIN_RATIO, f"{name}: estimate grossly below real ({ratio:.3f})"
        )
        self.assertLess(
            ratio, SMALL_MAX_RATIO, f"{name}: estimate grossly above real ({ratio:.3f})"
        )

    def test_llama3_dense_smoke(self):
        self._smoke("llama3", _build_trainer(llama3_registry))

    def test_qwen3_dense_smoke(self):
        self._smoke("qwen3", _build_trainer(qwen3_registry))

    # ---- signal 2: regime guard -- real peak must be stable across steps
    def test_real_peak_stable_across_steps(self):
        trainer = _build_trainer(llama3_registry, enable_passes=True)
        peaks = _measure_real_peaks(
            _run_step, trainer, self.tokens, self.labels, warmup=2, measure=4
        )
        lo, hi = min(peaks), max(peaks)
        spread = (hi - lo) / hi if hi else 0.0
        print(
            "\n[stability] per-step real peaks (GB): "
            + ", ".join(f"{p / 1e9:.3f}" for p in peaks)
            + f"  spread={spread:.4f}"
        )
        self.assertLess(
            spread,
            STABILITY_TOL,
            "real peak is not stable across steps -- the comparison target is "
            f"moving (cudagraph regime leaking?): {[p / 1e9 for p in peaks]}",
        )

    # ---- signal 3: analytical per-category (exact; the real validation)
    def _check_param_category(self, name, trainer):
        # Run one step so the graph is traced, then read category totals.
        _run_step(trainer, self.tokens, self.labels)
        torch.cuda.synchronize()
        result = estimate_peak_memory_modified(trainer._traced_step.gm)
        totals = result.category_totals
        model = trainer.model_parts[0]
        analytical = _analytical_param_bytes(model)
        # Parameters arrive as graph placeholders, so they land in "parameter"
        # and/or "input"; user inputs (tokens/positions) are tiny and share
        # "input". Compare the combined param-like total against ground truth.
        est_param = totals.get("parameter", 0) + totals.get("input", 0)
        ratio = est_param / analytical if analytical else float("inf")
        print(
            f"\n[{name} params] estimator={est_param / 1e9:.3f} GB  "
            f"analytical={analytical / 1e9:.3f} GB  ratio={ratio:.4f}"
        )
        self.assertAlmostEqual(
            ratio,
            1.0,
            delta=PARAM_BYTES_TOL,
            msg=f"{name}: param bytes off by >{PARAM_BYTES_TOL:.0%} (got {ratio:.4f})",
        )
        # No optimizer is run in forward_backward_step, so optimizer state is 0.
        self.assertEqual(totals.get("optimizer_state", 0), 0)
        # The joint graph must expose activations and gradients.
        self.assertGreater(totals.get("activation", 0), 0, f"{name}: no activations")
        self.assertGreater(totals.get("gradient", 0), 0, f"{name}: no gradients")

    def test_llama3_param_category(self):
        self._check_param_category(
            "llama3", _build_trainer(llama3_registry, enable_passes=True)
        )

    def test_qwen3_param_category(self):
        self._check_param_category(
            "qwen3", _build_trainer(qwen3_registry, enable_passes=True)
        )

    # ---- milestone deliverable: peak + schedule point + per-category report
    def test_llama3_report_complete(self):
        """The estimator must report all three deliverables, self-consistently:
        peak memory, the schedule point (node index + name), and the
        per-category breakdown. Validates llama (the milestone's starting model).
        """
        trainer = _build_trainer(llama3_registry, enable_passes=True)
        _run_step(trainer, self.tokens, self.labels)
        torch.cuda.synchronize()
        gm = trainer._traced_step.gm
        result = estimate_peak_memory_modified(gm)
        print("\n[llama3 report]\n" + result.summary())

        # (1) peak memory is reported and positive.
        self.assertGreater(result.peak_bytes, 0)

        # (2) schedule point is a valid node in execution order.
        nodes = list(gm.graph.nodes)
        self.assertTrue(0 <= result.peak_node_index < len(nodes))
        self.assertEqual(nodes[result.peak_node_index].name, result.peak_node_name)

        # (3) per-category breakdown is present, drawn from the known categories,
        #     and sums exactly to the reported peak (internal consistency).
        known = {"parameter", "input", "gradient", "activation", "temporary", "buffer"}
        self.assertTrue(result.per_category_at_peak)
        self.assertTrue(set(result.per_category_at_peak).issubset(known))
        self.assertEqual(
            sum(result.per_category_at_peak.values()),
            result.peak_bytes,
            "per-category breakdown must sum to the reported peak",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(
    RUN_LARGE, "set RUN_LARGE_MEM_TEST=1 to run the large reproduction tests"
)
class TestMemoryEstimatorLarge(unittest.TestCase):
    """Opt-in large-footprint validation that reproduces the PR's est-vs-real
    table: llama3 1B and qwen3 1.7B at batch=8, seq=2048, default memory policy,
    cudagraph disabled, passes applied -- the same configs a reviewer can run to
    get the same graph and a near-1.0 ratio.

    At this scale tensor memory dominates the real peak, so the scalar est/real
    ratio is a *meaningful* accuracy check (unlike the debugmodel smoke tests).
    Uses a real C4 batch (the run_train.sh data path). Run a single test per
    process for an exact ratio (see _free_cuda).

    The reproducible claim is the ~1.0 est/real ratio. Absolute GB differ from
    the run_train.sh table because this harness builds bf16 models, whereas
    run_train.sh keeps f32 masters and casts per use -- the estimator tracks both
    correctly (it is the ratio, not the GB, that is hardware/setup independent).

        RUN_LARGE_MEM_TEST=1 python -m unittest \\
            torchtitan.experiments.graph_trainer.tests.test_memory_estimator.\\
            TestMemoryEstimatorLarge.test_qwen3_large -v
    """

    def setUp(self):
        total = torch.cuda.get_device_properties(0).total_memory
        if total < MIN_GPU_BYTES_FOR_LARGE:
            self.skipTest(
                f"GPU has {total / 1e9:.0f} GB; large tests need "
                f">= {MIN_GPU_BYTES_FOR_LARGE / 1e9:.0f} GB"
            )
        _set_deterministic()

    def tearDown(self):
        torch.use_deterministic_algorithms(False)
        _free_cuda()

    def _check(self, name, trainer, seq):
        input_dict, labels = _real_batch(
            LARGE_BATCH, seq, LARGE_DATASET, LARGE_TOKENIZER_PATH
        )
        peaks = _measure_real_peaks(_run_step_dict, trainer, input_dict, labels)
        real = peaks[-1]
        est = estimate_peak_memory_modified(trainer._traced_step.gm)
        ratio = est.peak_bytes / real if real else float("inf")
        print(
            f"\n[{name}] batch={LARGE_BATCH} seq={seq} "
            f"dataset={LARGE_DATASET} policy={LARGE_MEMORY_POLICY}\n"
            f"    estimate={est.peak_bytes / 1e9:.3f} GB  real={real / 1e9:.3f} GB  "
            f"ratio=est/real={ratio:.3f}"
        )
        print(
            "    per-category at peak: "
            + ", ".join(
                f"{k}={v / 1e9:.3f}GB" for k, v in est.per_category_at_peak.items()
            )
        )
        self.assertGreater(est.peak_bytes, 0, f"{name}: estimate should be > 0")
        self.assertGreater(real, 0, f"{name}: real peak should be > 0")
        self.assertGreater(
            ratio, LARGE_MIN_RATIO, f"{name}: estimate below real ({ratio:.3f})"
        )
        self.assertLess(
            ratio, LARGE_MAX_RATIO, f"{name}: estimate above real ({ratio:.3f})"
        )

    def test_llama3_large(self):
        trainer = _build_trainer(
            llama3_registry,
            flavor=LARGE_LLAMA3_FLAVOR,
            memory_policy=LARGE_MEMORY_POLICY,
        )
        self._check(f"llama3 {LARGE_LLAMA3_FLAVOR} large", trainer, LARGE_LLAMA3_SEQ)

    def test_qwen3_large(self):
        trainer = _build_trainer(
            qwen3_registry,
            flavor=LARGE_QWEN3_FLAVOR,
            memory_policy=LARGE_MEMORY_POLICY,
        )
        self._check(f"qwen3 {LARGE_QWEN3_FLAVOR} large", trainer, LARGE_QWEN3_SEQ)


if __name__ == "__main__":
    unittest.main()

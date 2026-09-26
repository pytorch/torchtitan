# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import OverrideConfig
from torchtitan.distributed.utils import (
    is_in_batch_invariant_mode,
    set_batch_invariance,
)
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.rl.model import gdn, vllm_registry as registry
from torchtitan.rl.model.batch_invariance import force_logprobs_fn_for_batch_invariance
from torchtitan.rl.model.gdn_backend import (
    GDNExecutionPath,
    TorchTitanGDNAttentionMetadata,
)
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)

MODEL_ENV = "TORCHTITAN_QWEN3_5_0_8B_HF_PATH"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "batch_invariant", [False, True], ids=["regular", "batch-invariant"]
)
def test_gdn_full_runner_matches_eager(tmp_path: Path, batch_invariant: bool) -> None:
    model = os.environ.get(MODEL_ENV)
    if not model:
        pytest.skip(f"set {MODEL_ENV} to a local Qwen3.5-0.8B checkpoint")
    assert Path(model).is_dir(), model
    root = Path(__file__).resolve().parents[3]
    # Set import-time options before the child interpreter starts.
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        MODEL_ENV: str(Path(model).resolve()),
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "HF_HUB_OFFLINE": "1",
    }
    results = []
    for mode in ("eager", "full"):
        output = tmp_path / f"{mode}.json"
        command = ["timeout", "--kill-after=5s", "180s", sys.executable]
        command += "-m torch.distributed.run --standalone --nproc-per-node=1".split()
        command += [
            str(Path(__file__).resolve()),
            mode,
            str(output),
            str(int(batch_invariant)),
        ]
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                timeout=195,
                start_new_session=True,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        results.append(json.loads(output.read_text()))
    eager, full = results
    # Real packed recurrence ran eagerly and was captured for FULL replay.
    assert [False, "PACKED"] in eager["recurrent_calls"]
    assert [True, "PACKED"] in full["recurrent_calls"]
    assert eager["outputs"] == full["outputs"]
    assert eager["states"] == full["states"]
    assert all(step["mode"] == "NONE" for step in eager["dispatch"])
    replayed = [step for step in full["dispatch"] if step["mode"] == "FULL"]
    assert {step["kind"] for step in replayed} == {"prefill", "mixed", "decode"}
    assert {
        step["uniform"] for step in replayed if step["num_tokens"] > step["actual"]
    } == {False, True}
    assert any(
        step["mode"] == "NONE" and step["actual"] > 128 for step in full["dispatch"]
    )
    assert len({tuple(step["slots"]) for step in replayed}) > 1
    single_token = [step for step in replayed if step["path"] == "SINGLE_TOKEN"]
    assert any(step["initial"] == [False] for step in single_token)
    assert any(set(step["initial"]) == {False, True} for step in single_token)


@pytest.mark.parametrize(
    "batch_invariant", [False, True], ids=["regular", "batch-invariant"]
)
def test_single_token_reused_state_capture(monkeypatch, batch_invariant: bool) -> None:
    monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: batch_invariant)
    torch.manual_seed(42)
    device = torch.device("cuda")
    # Qwen3.5-0.8B's local TP=1 shapes, with padding between physical slots.
    shapes = ((3, 6144), (16, 128, 128))
    state_indices = torch.zeros(4, device=device, dtype=torch.int32)
    has_initial_state = torch.zeros(4, device=device, dtype=torch.bool)
    metadata = TorchTitanGDNAttentionMetadata(
        execution_path=GDNExecutionPath.SINGLE_TOKEN,
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=4,
        num_decode_tokens=4,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=4,
        non_spec_query_start_loc=torch.arange(5, device=device, dtype=torch.int32),
        non_spec_state_indices_tensor=state_indices,
        has_initial_state=has_initial_state,
    )
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )
    layer = gdn.VLLMInnerGatedDeltaNet.__new__(gdn.VLLMInnerGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.prefix = "gdn"
    layer.local_num_k_heads = layer.local_num_v_heads = 16
    layer.head_k_dim = layer.head_v_dim = 128
    layer.local_key_dim = 2048
    pools = [
        torch.randn(6, torch.Size(shape).numel() + 128, device=device, dtype=dtype)
        for shape, dtype in zip(shapes, (torch.bfloat16, torch.float32))
    ]

    def cache_views(storage):
        return tuple(
            pool[:, : torch.Size(shape).numel()].view(6, *shape)
            for pool, shape in zip(storage, shapes)
        )

    layer.kv_cache = cache_views(pools)
    inputs = torch.randn(4, 6144, device=device, dtype=torch.bfloat16)
    weight = torch.randn(6144, 4, device=device, dtype=torch.bfloat16)
    a, b = torch.randn(2, 4, 16, device=device)
    A_log, bias = torch.randn(2, 16, device=device)
    output = torch.zeros(4, 16, 128, device=device, dtype=torch.bfloat16)

    def forward():
        layer._forward(inputs, a, b, weight, None, A_log, bias, output)

    def without_mask(kernel):
        def resumed(*args, has_initial_state=None, **kwargs):
            return kernel(*args, **kwargs)

        return resumed

    with torch.inference_mode(), override_forward_context(context):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        # The first replay occupies both slots. Later requests reuse dirty slots,
        # alone and beside a continuing request, without changing the graph key.
        for slots, initial in (
            ([1, 2], [True, True]),
            ([2], [False]),
            ([1, 2], [True, False]),
            ([1, 2], [False, True]),
        ):
            before = [pool.clone() for pool in pools]
            reference = [pool.clone() for pool in pools]
            for slot, has_state in zip(slots, initial):
                if not has_state:
                    for state, clean in zip(layer.kv_cache, cache_views(reference)):
                        state[slot].fill_(float("nan"))
                        clean[slot].zero_()
            layer.kv_cache = cache_views(reference)
            # Explicitly clean history is the initialization oracle, independent
            # of both the mask implementation and eager-vs-graph agreement.
            state_indices.copy_(
                torch.tensor(slots + [0] * (4 - len(slots)), device=device)
            )
            # Bypass mask handling in the already-clean reference, so an
            # accidental reset of continuing slots cannot affect both sides.
            with monkeypatch.context() as unmasked:
                for name in (
                    "causal_conv1d_decode",
                    "recurrent_gdn_decode",
                    "recurrent_gdn",
                ):
                    unmasked.setattr(gdn, name, without_mask(getattr(gdn, name)))
                forward()
            expected = output.clone()
            layer.kv_cache = cache_views(pools)
            has_initial_state.copy_(
                torch.tensor(initial + [False] * (4 - len(slots)), device=device)
            )
            # Both repeated null slots and a negative sentinel must skip writes.
            state_indices[3] = -1
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert not output[len(slots) :].count_nonzero()
            for actual, clean, untouched in zip(pools, reference, before):
                assert torch.equal(actual.view(torch.uint8), clean.view(torch.uint8))
                unused = [slot for slot in range(6) if slot not in slots]
                assert torch.equal(actual[unused], untouched[unused])
                assert torch.equal(actual[:, -128:], untouched[:, -128:])


def run_engine(mode: str, output: Path, batch_invariant: bool) -> None:
    assert int(os.environ["WORLD_SIZE"]) == 1
    set_batch_invariance(batch_invariant)
    if batch_invariant:
        force_logprobs_fn_for_batch_invariance()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(42)
    recurrent_calls = []
    kernel_name = "recurrent_gdn" if batch_invariant else "paged_chunk_gdn"
    original_recurrent = getattr(gdn, kernel_name)

    def recurrent(*args, **kwargs):
        metadata = next(
            metadata
            for metadata in get_forward_context().attn_metadata.values()
            if isinstance(metadata, gdn.TorchTitanGDNAttentionMetadata)
        )
        recurrent_calls.append(
            [torch.cuda.is_current_stream_capturing(), metadata.execution_path.name]
        )
        return original_recurrent(*args, **kwargs)

    setattr(gdn, kernel_name, recurrent)
    model = os.environ[MODEL_ENV]
    registry.register_to_vllm(
        model_registry("0.8B", enable_sp=True, seq_len=256, attn_backend="varlen"),
        parallelism=registry.InferenceParallelismConfig(tensor_parallel_degree=1),
        compile_config=None,
        checkpointer_config=CheckpointManager.Config(
            initial_load_in_hf=True, initial_load_path=model
        ),
        override=OverrideConfig(),
    )
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=model,
            config_format=registry.TORCHTITAN_CONFIG_FORMAT,
            dtype="bfloat16",
            worker_cls=registry.TORCHTITAN_WORKER_CLS,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=1,
            enforce_eager=mode == "eager",
            attention_config={"backend": "CUSTOM"},
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL" if mode == "full" else "NONE",
                "cudagraph_capture_sizes": [1, 2, 4, 64, 128],
                "cudagraph_num_of_warmups": 1,
            },
            max_model_len=256,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            num_gpu_blocks_override=32,
            gpu_memory_utilization=0.25,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            seed=42,
            disable_log_stats=True,
        )
    )
    result = {
        "outputs": {},
        "states": [],
        "dispatch": [],
        "recurrent_calls": recurrent_calls,
    }

    def instrument(worker):
        runner = worker.model_runner
        assert is_in_batch_invariant_mode() == batch_invariant
        layers = {
            name: layer
            for name, layer in runner.compilation_config.static_forward_context.items()
            if isinstance(layer, gdn.VLLMInnerGatedDeltaNet)
        }
        assert layers
        with torch.inference_mode():
            for layer in layers.values():
                for state in layer.kv_cache:
                    state.zero_()
        replays = []
        if mode == "full":
            entries = runner.model.concrete_cudagraph_entries
            for entry in entries.values():
                assert entry.cudagraph is not None

                def replay(original=entry.cudagraph.replay):
                    replays.append(get_forward_context().batch_descriptor.uniform)
                    return original()

                entry.cudagraph.replay = replay
        original_forward = runner._model_forward

        def forward(*args, **kwargs):
            context = get_forward_context()
            requests = list(runner.input_batch.req_ids)
            metadata = context.attn_metadata
            first = metadata[next(iter(layers))]
            offsets = first.non_spec_query_start_loc.cpu().tolist()[: len(requests) + 1]
            prefills = [end - start > 1 for start, end in zip(offsets, offsets[1:])]
            kind = "prefill" if all(prefills) else "decode"
            if any(prefills) and not all(prefills):
                kind = "mixed"
            runtime = context.cudagraph_runtime_mode.name
            assert runtime in ("NONE", "FULL")
            replays.clear()
            value = original_forward(*args, **kwargs)
            uniform = context.batch_descriptor.uniform
            assert replays == ([uniform] if runtime == "FULL" else [])
            hashes = {}
            for name, layer in layers.items():
                slots = (
                    metadata[name]
                    .non_spec_state_indices_tensor.cpu()
                    .tolist()[: len(requests)]
                )
                assert all(slot > 0 for slot in slots)
                hashes[name] = []
                for state in layer.kv_cache:
                    raw = state[slots].view(torch.uint8).cpu().numpy().tobytes()
                    hashes[name].append(hashlib.sha256(raw).hexdigest())
            result["states"].append([requests, offsets, hashes])
            record = asdict(context.batch_descriptor)
            assert record["num_tokens"] == kwargs["input_ids"].shape[0]
            record.update(
                mode=runtime,
                slots=slots,
                kind=kind,
                actual=offsets[-1],
                path=first.execution_path.name,
                initial=first.has_initial_state.cpu().tolist()[: len(requests)],
            )
            result["dispatch"].append(record)
            return value

        runner._model_forward = forward

    engine.collective_rpc(instrument)
    sampling = SamplingParams(
        temperature=0.0, max_tokens=8, ignore_eos=True, seed=42, logprobs=5
    )
    base = engine.renderer.tokenizer.encode(
        "Explain physics.", add_special_tokens=False
    )
    for wave, (lengths, initial_count) in enumerate(
        (([11, 17, 23, 37], 4), ([129], 1), ([9, 13, 29, 41], 3), ([1], 1), ([1, 1], 1))
    ):
        prompts = engine.renderer.render_cmpl(
            [{"prompt_token_ids": (base * length)[:length]} for length in lengths]
        )
        for index, prompt in enumerate(prompts[:initial_count]):
            engine.add_request(f"{wave}-{index}", prompt, sampling)
        step = 0
        while engine.has_unfinished_requests():
            if step == 2:
                for index in range(initial_count, len(prompts)):
                    engine.add_request(f"{wave}-{index}", prompts[index], sampling)
            for request in engine.step():
                if request.finished:
                    completion = request.outputs[0]
                    assert len(completion.token_ids) == 8
                    assert completion.logprobs is not None
                    result["outputs"][request.request_id] = asdict(completion)
            step += 1
    assert len(result["outputs"]) == 12
    output.write_text(json.dumps(result, allow_nan=False))
    engine.model_executor.shutdown()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_engine(sys.argv[1], Path(sys.argv[2]), bool(int(sys.argv[3])))

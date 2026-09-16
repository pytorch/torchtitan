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

MODEL_ENV = "TORCHTITAN_QWEN3_5_0_8B_HF_PATH"


def test_gdn_full_runner_matches_eager(tmp_path: Path) -> None:
    model = os.environ.get(MODEL_ENV)
    if not model:
        pytest.skip(f"set {MODEL_ENV} to a local Qwen3.5-0.8B checkpoint")
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert Path(model).is_dir(), model
    root = Path(__file__).resolve().parents[4]
    env = {**os.environ, "PYTHONPATH": str(root), MODEL_ENV: str(Path(model).resolve())}
    results = []
    for mode in ("eager", "full"):
        output = tmp_path / f"{mode}.json"
        command = ["timeout", "--kill-after=5s", "180s", sys.executable]
        command += "-m torch.distributed.run --standalone --nproc-per-node=1".split()
        command += [str(Path(__file__).resolve()), mode, str(output)]
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


def run_engine(mode: str, output: Path) -> None:
    os.environ.update(
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_USE_BREAKABLE_CUDAGRAPH="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1",
        HF_HUB_OFFLINE="1",
    )
    import torch
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.forward_context import get_forward_context

    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.config import CompileConfig, OverrideConfig
    from torchtitan.experiments.rl.models import vllm_registry as registry
    from torchtitan.experiments.rl.models.gdn import VLLMInnerGatedDeltaNet
    from torchtitan.experiments.rl.models.vllm_worker import TorchTitanGPUModelRunner
    from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper
    from torchtitan.models.qwen3_5 import model_registry

    assert int(os.environ["WORLD_SIZE"]) == 1
    torch.manual_seed(42)
    model = os.environ[MODEL_ENV]
    registry.register_to_vllm(
        model_registry("0.8B", seq_len=256, attn_backend="varlen"),
        parallelism=registry.InferenceParallelismConfig(tensor_parallel_degree=1),
        compile_config=CompileConfig(enable=False),
        checkpoint_config=CheckpointManager.Config(
            enable=True, initial_load_in_hf=True, initial_load_path=model
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
    result = {"outputs": {}, "states": [], "dispatch": []}

    def instrument(worker):
        runner = worker.model_runner
        assert type(runner) is TorchTitanGPUModelRunner
        layers = {
            name: layer
            for name, layer in runner.compilation_config.static_forward_context.items()
            if isinstance(layer, VLLMInnerGatedDeltaNet)
        }
        assert layers
        with torch.inference_mode():
            for layer in layers.values():
                for state in layer.kv_cache:
                    state.zero_()
        replays = []
        if mode == "full":
            assert type(runner.model) is CUDAGraphWrapper
            assert type(runner.model.runnable) is BreakableCUDAGraphWrapper
            assert isinstance(runner.model.unwrap(), VLLMModelWrapper)
            entries = runner.model.concrete_cudagraph_entries
            assert {descriptor.uniform for descriptor in entries} == {False, True}
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
            record.update(mode=runtime, slots=slots, kind=kind, actual=offsets[-1])
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
    for wave, lengths in enumerate(([11, 17, 23, 37], [129], [9, 13, 29, 41])):
        prompts = engine.renderer.render_cmpl(
            [{"prompt_token_ids": (base * length)[:length]} for length in lengths]
        )
        for index, prompt in enumerate(prompts):
            if wave != 2 or index < 3:
                engine.add_request(f"{wave}-{index}", prompt, sampling)
        step = 0
        while engine.has_unfinished_requests():
            if wave == 2 and step == 2:
                engine.add_request("2-3", prompts[3], sampling)
            for request in engine.step():
                if request.finished:
                    completion = request.outputs[0]
                    assert len(completion.token_ids) == 8
                    assert completion.logprobs is not None
                    result["outputs"][request.request_id] = asdict(completion)
            step += 1
    assert len(result["outputs"]) == 9
    if torch.distributed.get_rank() == 0:
        output.write_text(json.dumps(result, allow_nan=False))
    engine.model_executor.shutdown()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_engine(sys.argv[1], Path(sys.argv[2]))

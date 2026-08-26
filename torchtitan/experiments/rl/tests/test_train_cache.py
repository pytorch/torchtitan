# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan.experiments.rl import train


def test_configure_local_compile_cache(monkeypatch) -> None:
    created_dirs: list[tuple[str, bool]] = []
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("TORCHTITAN_RL_ROLE", "generator")
    monkeypatch.setattr(train.os, "getuid", lambda: 1000)
    monkeypatch.setattr(train.os, "getpid", lambda: 5678)
    monkeypatch.setattr(
        train.os, "uname", lambda: SimpleNamespace(nodename="compute-01")
    )
    monkeypatch.setattr(
        train.os,
        "makedirs",
        lambda path, *, exist_ok: created_dirs.append((path, exist_ok)),
    )

    train._configure_local_compile_cache()

    cache_root = "/tmp/torchtitan-rl-1000/1234/generator-compute-01-5678"
    expected = {
        "TRITON_CACHE_DIR": f"{cache_root}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache_root}/torchinductor",
        "VLLM_CACHE_ROOT": f"{cache_root}/vllm",
    }
    assert {name: train.os.environ[name] for name in expected} == expected
    assert created_dirs == [(path, True) for path in expected.values()]


def test_native_vllm_does_not_enable_wrapper_breakable_cudagraph() -> None:
    generator_config = SimpleNamespace(
        use_native_vllm_model=True,
        cudagraph=SimpleNamespace(enable=True, mode="FULL_AND_PIECEWISE"),
    )

    assert train.breakable_cudagraph_env(generator_config) == {}

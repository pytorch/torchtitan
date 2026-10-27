# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Installed supervisor/service plus real four-rank CPU checkpoint publication.

This isolates the save protocol and cursor restore from native TorchFT quorum.
It is not an NPU training or native fault-recovery acceptance test.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def sample_digest():
    import numpy as np
    import torch

    value = dict(
        python=random.random(),
        numpy=np.random.random(4).tolist(),
        torch=torch.rand(4).tolist(),
    )
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


class MappingState:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return copy.deepcopy(self.value)

    def load_state_dict(self, value):
        self.value = copy.deepcopy(value)


def worker(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torchtitan.components.optim import AdamW, LRSchedulersContainer
    from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer
    from torchtitan.last_words.checkpoint import LastWordsCheckpointManager
    from torchtitan.last_words.control import LastWordsConfig, LastWordsCoordinator
    from torchtitan.last_words.rng import RNG_STATE, RNGCheckpointState

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    control = manager = None
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        optimizers = TorchFTOptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=".*", fused=False)]
        ).build(model_parts=[model])
        optimizers.init_cache_state_dict()
        schedulers = LRSchedulersContainer.Config(warmup_steps=0).build(
            optimizers=optimizers, training_steps=10
        )
        loss = model(torch.tensor([[1.0, -0.5]])).square().sum()
        loss.backward()
        optimizers.step()
        schedulers.step()
        progress = dict(step=1, ntokens_seen=8, last_words_num_updates=1)
        cursor = MappingState(
            {f"rank_{rank}": dict(position=11 + args.replica * 2 + rank)}
        )
        control = LastWordsCoordinator(
            LastWordsConfig(
                enable=True,
                run_id=args.run_id,
                resume_path=args.resume_path,
                coordination_backend="file" if args.restore else "service",
                timeout_seconds=45,
                heartbeat_timeout_seconds=30,
            ),
            folder=str(args.root),
            replica_id=args.replica,
            num_replicas=2,
            rank=rank,
            world_size=2,
        )
        manager = LastWordsCheckpointManager.Config(
            folder=str(args.root),
            interval=10,
            keep_latest_k=0,
            last_save_model_only=False,
            enable_ft_dataloader_checkpoints=False,
        ).build(
            model_parts=[model],
            optimizers=optimizers,
            lr_schedulers=schedulers,
            ema=None,
            extra_states={"train_state": MappingState(progress)},
            sd_adapter=None,
            ft_manager=None,
        )
        # Exercise real checkpoint/service code without emulating native quorum.
        manager.last_words = control
        manager.last_words_signature = "cpu-service-dcp-rng-schema1"
        manager.last_words_state = dict(progress)
        manager.ft_replica_id = args.replica
        manager.enable_ft_dataloader_checkpoints = True
        manager.pg = dist.new_group(backend="gloo")
        manager._last_words_control_pg = dist.new_group(backend="gloo")
        random.seed(1000 + args.replica * 2 + rank)
        np.random.seed(2000 + args.replica * 2 + rank)
        torch.manual_seed(3000 + args.replica * 2 + rank)
        rng = RNGCheckpointState(torch.device("cpu"), rank)
        rng.capture()
        expected_rng = sample_digest()
        manager.rng_state = rng
        manager.ft_states = {"dataloader": cursor, RNG_STATE: rng}
        expected_model = model_digest(model)
        expected_cursor = cursor.state_dict()
        if args.restore:
            saved = json.loads(
                (
                    args.output / f"saved-replica-{args.replica}-rank-{rank}.json"
                ).read_text()
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            cursor.value = {f"rank_{rank}": dict(position=999)}
            random.seed(7000 + args.replica * 2 + rank)
            np.random.seed(8000 + args.replica * 2 + rank)
            torch.manual_seed(9000 + args.replica * 2 + rank)
            rng.capture()
            assert manager.load(1)
            assert model_digest(model) == saved["model_sha256"]
            assert cursor.state_dict() == saved["cursor"]
            actual_rng = sample_digest()
            assert actual_rng == saved["expected_rng"]
            (
                args.output / f"restore-replica-{args.replica}-rank-{rank}.json"
            ).write_text(
                json.dumps(
                    dict(
                        replica=args.replica,
                        rank=rank,
                        pid=os.getpid(),
                        model_equal=True,
                        cursor_equal=True,
                        rng_equal=True,
                        expected_rng=saved["expected_rng"],
                        actual_rng=actual_rng,
                        target=args.resume_path,
                    )
                )
            )
            return
        control.publish("completed", state=progress)
        manager._ft_save(1)
        manager.maybe_wait_for_saving()
        deadline = time.monotonic() + 45
        while not all(
            manager._cursor_candidates(replica, progress) for replica in range(2)
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Other logical replica did not publish its complete cursor"
                )
            time.sleep(0.05)
        control.request("CPU service checkpoint integration")
        target = manager.save_last_words(progress)
        (args.output / f"saved-replica-{args.replica}-rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    replica=args.replica,
                    rank=rank,
                    pid=os.getpid(),
                    model_sha256=expected_model,
                    cursor=expected_cursor,
                    expected_rng=expected_rng,
                    target=str(target),
                )
            )
        )
    finally:
        if control is not None:
            control.close()
        if manager is not None:
            manager.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--replica", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--resume-path", default="")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    checkpoint = output / "checkpoint"
    commands = [
        [
            sys.executable,
            "-I",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--worker",
            "--output",
            str(output),
            "--root",
            str(checkpoint),
            "--replica",
            str(replica),
            "--run-id",
            "{run_id}",
            "--resume-path",
            "{resume_path}",
        ]
        for replica in range(2)
    ]
    spec = dict(
        mode="last_words",
        coordination_backend="service",
        commands=commands,
        checkpoint_root=str(checkpoint),
        world_size=2,
        required_free_bytes=16 * 1024**2,
        max_attempts=1,
        max_attempt_seconds=150,
        startup_timeout_seconds=80,
        fault_timeout_seconds=45,
        heartbeat_timeout_seconds=30,
        no_progress_seconds=90,
        cwd="/var/tmp",
    )
    (output / "spec.json").write_text(json.dumps(spec, indent=2))
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-I",
        "-m",
        "torchtitan.last_words.job",
        "run",
        "--spec",
        str(output / "spec.json"),
        "--state-dir",
        str(output / "supervisor"),
    ]
    with (output / "supervisor.log").open("w") as log:
        process = subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    assert process.returncode == 0, f"Supervisor failed: inspect {output}"
    manifests = list(checkpoint.glob("last_words/launch-*/replica-*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["rng_schema"] == 1 and manifest["rng_shards"] == 2
    assert set(manifest["cursors"]) == {"0", "1"}
    # A terminal save exits its old generation. Restore in fresh processes,
    # outside the supervisor's bounded terminal-shutdown window.
    for replica in range(2):
        command = [
            argument.replace("{run_id}", f"restore-{replica}").replace(
                "{resume_path}", str(manifests[0].parent)
            )
            for argument in commands[replica]
        ] + ["--restore"]
        with (output / f"restore-{replica}.log").open("w") as log:
            restored = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        assert restored.returncode == 0, f"Fresh replica {replica} restore failed"
    records = [
        json.loads((output / f"restore-replica-{replica}-rank-{rank}.json").read_text())
        for replica in range(2)
        for rank in range(2)
    ]
    assert len({entry["actual_rng"] for entry in records}) == 4
    (output / "result.json").write_text(
        json.dumps(
            dict(
                scope="installed supervisor/service and real CPU DCP; no native TorchFT quorum or NPU",
                world_size=4,
                success=True,
                manifest=str(manifests[0]),
                restores=records,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

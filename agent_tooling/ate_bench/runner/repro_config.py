"""TorchTitan-specific settings for the ATE-Bench reproduction.

Encodes the paper's fixed evaluation config (Appendix B): ``PP=4, EP=2, DP=1``,
seq_len 2048, global batch 1024, BF16, 8 GPUs — mapped onto TorchTitan's real
infrastructure:

- launcher: ``run_train.sh`` (``MODULE``/``CONFIG`` env + CLI overrides)
- parallelism flags: ``--parallelism.{pipeline,expert}_parallel_degree`` etc.
- model: an MoE model (paper used Qwen3-30B-A3B / DeepSeek-V2-Lite / GPT-OSS-20B;
  TorchTitan analogs: ``deepseek_v3``, ``qwen3`` (``debugmodel_moe``), ``llama4``,
  ``gpt_oss``)
- dataset: C4 (TorchTitan native; the paper used DCLM)
- checkpoints: ``scripts/checkpoint_conversion/convert_{to,from}_hf.py``

The ``*_debugmodel`` configs are tiny and meant for harness validation / smoke
runs; reproducing paper-scale numbers needs a full MoE config and a real
checkpoint on 8 GPUs. Mesh degrees are overridable (env in setup/train.sh, or
fields here) so the harness can run a 1-GPU smoke before the full 8-GPU config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# .../agent_tooling/ate_bench/runner/repro_config.py
RUNNER_DIR = Path(__file__).resolve().parent
ATE_ROOT = RUNNER_DIR.parent
REPO_ROOT = ATE_ROOT.parents[1]  # agent_tooling/.. -> repo root
SETUP_DIR = ATE_ROOT / "setup"


@dataclass
class ReproConfig:
    # Which framework/model is under test.
    module: str = "deepseek_v3"
    config: str = "deepseek_v3_debugmodel"
    label: str = "titan"  # used for workspace/<label>/... and results/<label>/

    # Fixed mesh (paper Appendix B): PP=4, EP=2, DP=1.
    # NOTE: the paper's notation is multiplicative (PP*EP*DP = 8), but TorchTitan's
    # mesh is dp_replicate*dp_shard*cp*tp*pp == world_size with EP *carved from* the
    # data axis (not multiplied on top). So on 8 GPUs we set pp=4 and leave
    # dp_shard=-1 (auto -> 8/(1*1*1*4)=2); EP=2 then shards experts across that data
    # axis of 2. dp_shard=1 would fail validation (1*1*1*1*4 != 8).
    ngpu: int = 8
    pipeline_parallel_degree: int = 4
    expert_parallel_degree: int = 2
    data_parallel_shard_degree: int = -1  # auto-fill the mesh; EP overlaps this axis

    # Fixed training shape.
    seq_len: int = 2048
    global_batch_size: int = 1024
    dataset: str = "c4"  # paper used DCLM; TorchTitan native is C4

    workspace: Path = field(default_factory=lambda: ATE_ROOT / "workspace")

    def parallelism_flags(self) -> list[str]:
        return [
            f"--parallelism.pipeline_parallel_degree={self.pipeline_parallel_degree}",
            f"--parallelism.expert_parallel_degree={self.expert_parallel_degree}",
            f"--parallelism.data_parallel_shard_degree={self.data_parallel_shard_degree}",
        ]

    def workspace_dir(self) -> Path:
        return self.workspace / self.label

    def routing_traces_dir(self) -> Path:
        return self.workspace_dir() / "routing-traces"

    def heavy_kernels_dir(self) -> Path:
        return self.workspace_dir() / "heavy-kernels"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["workspace"] = str(self.workspace)
        return d


    def render(self, text: str, task_id: str = "", workspace_root: Path | None = None) -> str:
        """Substitute {{PLACEHOLDER}} tokens in a task prompt with config values.

        Script paths are repo-root-relative so they resolve whether the agent runs
        in the main checkout (operate tasks) or a worktree (new-feature). Pass an
        absolute ``workspace_root`` (the main checkout's workspace) so worktree
        runs still write artifacts where the checks read them.
        """
        train_sh = "agent_tooling/ate_bench/setup/train.sh"
        evaluate_sh = "agent_tooling/ate_bench/setup/evaluate.sh"
        ws = str(workspace_root) if workspace_root else "agent_tooling/ate_bench/workspace"
        repl = {
            "{{NGPU}}": str(self.ngpu),
            "{{MODULE}}": self.module,
            "{{CONFIG}}": self.config,
            "{{LABEL}}": self.label,
            "{{SEQ_LEN}}": str(self.seq_len),
            "{{GLOBAL_BATCH_SIZE}}": str(self.global_batch_size),
            "{{MESH}}": (
                f"PP={self.pipeline_parallel_degree}, EP={self.expert_parallel_degree}, "
                f"DP={self.data_parallel_shard_degree}"
            ),
            "{{DATASET}}": self.dataset,
            "{{TRAIN_SH}}": train_sh,
            "{{EVALUATE_SH}}": evaluate_sh,
            "{{WORKSPACE}}": ws,
            "{{ROUTING_TRACES_DIR}}": f"{ws}/{self.label}/routing-traces",
            "{{HEAVY_KERNELS_DIR}}": f"{ws}/{self.label}/heavy-kernels",
            "{{TASK_ID}}": task_id,
        }
        for k, v in repl.items():
            text = text.replace(k, v)
        return text


# Singleton default; runners may build their own with overrides.
DEFAULT = ReproConfig()

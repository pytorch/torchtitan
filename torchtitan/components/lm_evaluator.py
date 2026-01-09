# Copyright (c) Nous Research.
# All rights reserved.

"""
LM-Evaluation-Harness integration for torchtitan.

This module provides automatic evaluation during training using
lm-evaluation-harness with full reproducibility through seed control
and configuration logging.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.config.job_config import JobConfig, LMEvalConfig


class LMEvaluator:
    """
    Evaluator component for running lm-evaluation-harness during training.

    Supports three execution modes:
    - inline: Run in same process (blocks training)
    - subprocess: Run in background subprocess (non-blocking)
    - slurm: Submit as SLURM job (fully async)

    All modes support full reproducibility through:
    - Seed control (4 independent seeds)
    - Configuration logging
    - Generated SLURM scripts for audit trail
    """

    def __init__(self, job_config: JobConfig, rank: int = 0) -> None:
        """
        Initialize the LM Evaluator.

        Args:
            job_config: The job configuration containing lm_eval settings
            rank: Current distributed rank (eval only runs on rank 0)
        """
        self.job_config = job_config
        self.lm_eval_config = job_config.lm_eval
        self.rank = rank

        # Only rank 0 runs evaluations
        self.enabled = self.lm_eval_config.enable and rank == 0

        if not self.enabled:
            return

        # Setup directories
        self.dump_folder = Path(job_config.job.dump_folder)
        self.output_dir = self.dump_folder / self.lm_eval_config.output_dir
        self.slurm_script_dir = self.dump_folder / self.lm_eval_config.slurm_script_dir
        self.slurm_logs_dir = self.dump_folder / self.lm_eval_config.slurm_logs_dir

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.lm_eval_config.mode == "slurm":
            self.slurm_script_dir.mkdir(parents=True, exist_ok=True)
            self.slurm_logs_dir.mkdir(parents=True, exist_ok=True)

        # Resolve paths
        self.torchtitan_path = self._resolve_torchtitan_path()
        self.lm_eval_path = self.lm_eval_config.lm_eval_path

        # Track running jobs
        self.running_jobs: dict[int, dict[str, Any]] = {}

        logger.info(
            f"LMEvaluator initialized: mode={self.lm_eval_config.mode}, "
            f"tasks={self.lm_eval_config.tasks}, interval={self.lm_eval_config.eval_interval}"
        )

    def _resolve_torchtitan_path(self) -> str:
        """Resolve the torchtitan installation path."""
        if self.lm_eval_config.torchtitan_path:
            return self.lm_eval_config.torchtitan_path

        # Try to find torchtitan from current module path
        possible_paths = [
            Path(__file__).parent.parent.parent,  # From components/lm_evaluator.py
            Path("/home/phuc/workspace/moe/online_evals/torchtitan"),
        ]

        for path in possible_paths:
            if (path / "torchtitan").exists():
                return str(path)

        # Fallback to assuming it's in PYTHONPATH
        return ""

    def should_evaluate(self, step: int) -> bool:
        """
        Check if evaluation should run at this step.

        Args:
            step: Current training step

        Returns:
            True if evaluation should run
        """
        if not self.enabled:
            return False

        if step == 0:
            return False

        return step % self.lm_eval_config.eval_interval == 0

    def run_evaluation(
        self,
        step: int,
        checkpoint_path: str,
    ) -> dict[str, Any] | None:
        """
        Run evaluation for the given checkpoint.

        Args:
            step: Current training step
            checkpoint_path: Path to the checkpoint to evaluate

        Returns:
            Evaluation results dict (inline mode) or job info (subprocess/slurm mode)
        """
        if not self.enabled:
            return None

        logger.info(f"Starting evaluation at step {step}")

        # Create step-specific output directory
        step_output_dir = self.output_dir / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation config for reproducibility
        self._save_eval_config(step, checkpoint_path, step_output_dir)

        if self.lm_eval_config.mode == "inline":
            return self._run_inline(step, checkpoint_path, step_output_dir)
        elif self.lm_eval_config.mode == "subprocess":
            return self._run_subprocess(step, checkpoint_path, step_output_dir)
        elif self.lm_eval_config.mode == "slurm":
            return self._run_slurm(step, checkpoint_path, step_output_dir)
        else:
            raise ValueError(f"Unknown eval mode: {self.lm_eval_config.mode}")

    def _save_eval_config(
        self,
        step: int,
        checkpoint_path: str,
        output_dir: Path,
    ) -> None:
        """Save evaluation configuration for reproducibility."""
        eval_info = {
            "step": step,
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "lm_eval_config": asdict(self.lm_eval_config),
            "model_config": {
                "name": self.job_config.model.name,
                "flavor": self.job_config.model.flavor,
            },
            "seeds": {
                "random_seed": self.lm_eval_config.get_seeds()[0],
                "numpy_seed": self.lm_eval_config.get_seeds()[1],
                "torch_seed": self.lm_eval_config.get_seeds()[2],
                "fewshot_seed": self.lm_eval_config.get_seeds()[3],
            },
        }

        config_path = output_dir / "eval_config.json"
        with open(config_path, "w") as f:
            json.dump(eval_info, f, indent=2, default=str)

        logger.info(f"Saved eval config to {config_path}")

    def _get_eval_command(
        self,
        checkpoint_path: str,
        output_dir: Path,
    ) -> list[str]:
        """Build the lm_eval command line arguments."""
        cfg = self.lm_eval_config
        model_cfg = self.job_config.model

        # For DCP checkpoints, tokenizer is in hf_assets_path
        # For HF checkpoints, tokenizer is in checkpoint_path
        tokenizer_path = model_cfg.hf_assets_path

        # Build model_args string
        model_args = (
            f"pretrained={checkpoint_path},"
            f"tokenizer_path={tokenizer_path},"
            f"model_name={model_cfg.name},"
            f"model_flavor={model_cfg.flavor},"
            f"dtype=bfloat16,"
            f"max_seq_len={cfg.max_seq_len}"
        )

        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "torchtitan",
            "--model_args",
            model_args,
            "--tasks",
            cfg.tasks,
            "--num_fewshot",
            str(cfg.num_fewshot),
            "--batch_size",
            str(cfg.batch_size),
            "--seed",
            cfg.get_seed_string(),
            "--output_path",
            str(output_dir),
        ]

        if cfg.limit is not None:
            cmd.extend(["--limit", str(cfg.limit)])

        if cfg.log_samples:
            cmd.append("--log_samples")

        return cmd

    def _run_inline(
        self,
        step: int,
        checkpoint_path: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Run evaluation inline (blocking)."""
        logger.info(f"Running inline evaluation for step {step}")

        try:
            import lm_eval

            cfg = self.lm_eval_config
            model_cfg = self.job_config.model

            # Set seeds for reproducibility
            seeds = cfg.get_seeds()
            import random

            import numpy as np
            import torch

            random.seed(seeds[0])
            np.random.seed(seeds[1])
            torch.manual_seed(seeds[2])

            # For DCP checkpoints, tokenizer is in hf_assets_path
            tokenizer_path = model_cfg.hf_assets_path

            model_args = (
                f"pretrained={checkpoint_path},"
                f"tokenizer_path={tokenizer_path},"
                f"model_name={model_cfg.name},"
                f"model_flavor={model_cfg.flavor},"
                f"dtype=bfloat16,"
                f"max_seq_len={cfg.max_seq_len}"
            )

            results = lm_eval.simple_evaluate(
                model="torchtitan",
                model_args=model_args,
                tasks=cfg.tasks.split(","),
                num_fewshot=cfg.num_fewshot,
                limit=cfg.limit,
                batch_size=cfg.batch_size,
                log_samples=cfg.log_samples,
            )

            # Save results
            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Log summary
            self._log_results_summary(step, results)

            return results

        except Exception as e:
            logger.error(f"Inline evaluation failed: {e}")
            error_info = {"error": str(e), "step": step}
            error_path = output_dir / "error.json"
            with open(error_path, "w") as f:
                json.dump(error_info, f, indent=2)
            raise

    def _run_subprocess(
        self,
        step: int,
        checkpoint_path: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Run evaluation in background subprocess (non-blocking)."""
        logger.info(f"Starting subprocess evaluation for step {step}")

        # Generate eval script
        script_path = output_dir / "run_eval.py"
        self._generate_eval_script(checkpoint_path, output_dir, script_path)

        # Build environment
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        if self.torchtitan_path:
            pythonpath = f"{self.torchtitan_path}:{pythonpath}"
        if self.lm_eval_path:
            pythonpath = f"{self.lm_eval_path}:{pythonpath}"
        env["PYTHONPATH"] = pythonpath

        # Start subprocess
        log_path = output_dir / "eval.log"
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        job_info = {
            "mode": "subprocess",
            "step": step,
            "pid": process.pid,
            "script_path": str(script_path),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
        }

        self.running_jobs[step] = job_info
        logger.info(f"Subprocess evaluation started: PID={process.pid}")

        return job_info

    def _run_slurm(
        self,
        step: int,
        checkpoint_path: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Submit evaluation as SLURM job (fully async)."""
        logger.info(f"Submitting SLURM evaluation job for step {step}")

        # Generate eval script
        script_path = output_dir / "run_eval.py"
        self._generate_eval_script(checkpoint_path, output_dir, script_path)

        # Generate SLURM script
        slurm_script_path = self.slurm_script_dir / f"eval_step_{step}.sh"
        self._generate_slurm_script(step, script_path, output_dir, slurm_script_path)

        # Submit job
        result = subprocess.run(
            ["sbatch", str(slurm_script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"SLURM submission failed: {result.stderr}")
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")

        # Parse job ID from "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]

        job_info = {
            "mode": "slurm",
            "step": step,
            "job_id": job_id,
            "script_path": str(script_path),
            "slurm_script_path": str(slurm_script_path),
            "output_dir": str(output_dir),
        }

        self.running_jobs[step] = job_info
        logger.info(f"SLURM job submitted: {job_id}")

        # Save job info
        job_info_path = output_dir / "slurm_job_info.json"
        with open(job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)

        return job_info

    def _generate_eval_script(
        self,
        checkpoint_path: str,
        output_dir: Path,
        script_path: Path,
    ) -> None:
        """Generate a standalone evaluation script for reproducibility."""
        cfg = self.lm_eval_config
        model_cfg = self.job_config.model
        seeds = cfg.get_seeds()

        script_content = f'''#!/usr/bin/env python
"""
Auto-generated evaluation script for reproducibility.
Generated at: {datetime.now().isoformat()}

To rerun this evaluation:
    python {script_path}
"""

import sys
import json
import random
import numpy as np
import torch

# Add torchtitan to path if needed
torchtitan_path = "{self.torchtitan_path}"
if torchtitan_path and torchtitan_path not in sys.path:
    sys.path.insert(0, torchtitan_path)

lm_eval_path = "{self.lm_eval_path or ''}"
if lm_eval_path and lm_eval_path not in sys.path:
    sys.path.insert(0, lm_eval_path)

import lm_eval

# Set seeds for reproducibility
RANDOM_SEED = {seeds[0]}
NUMPY_SEED = {seeds[1]}
TORCH_SEED = {seeds[2]}
FEWSHOT_SEED = {seeds[3]}

random.seed(RANDOM_SEED)
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)

# Evaluation configuration
CHECKPOINT_PATH = "{checkpoint_path}"
TOKENIZER_PATH = "{model_cfg.hf_assets_path}"
OUTPUT_DIR = "{output_dir}"
MODEL_NAME = "{model_cfg.name}"
MODEL_FLAVOR = "{model_cfg.flavor}"
TASKS = "{cfg.tasks}"
NUM_FEWSHOT = {cfg.num_fewshot}
BATCH_SIZE = {cfg.batch_size}
MAX_SEQ_LEN = {cfg.max_seq_len}
LIMIT = {cfg.limit if cfg.limit is not None else "None"}
LOG_SAMPLES = {cfg.log_samples}

print("=" * 60)
print("LM-EVALUATION-HARNESS")
print("=" * 60)
print(f"Checkpoint: {{CHECKPOINT_PATH}}")
print(f"Model: {{MODEL_NAME}} ({{MODEL_FLAVOR}})")
print(f"Tasks: {{TASKS}}")
print(f"Seeds: random={{RANDOM_SEED}}, numpy={{NUMPY_SEED}}, torch={{TORCH_SEED}}, fewshot={{FEWSHOT_SEED}}")
print("=" * 60)

model_args = (
    f"pretrained={{CHECKPOINT_PATH}},"
    f"tokenizer_path={{TOKENIZER_PATH}},"
    f"model_name={{MODEL_NAME}},"
    f"model_flavor={{MODEL_FLAVOR}},"
    f"dtype=bfloat16,"
    f"max_seq_len={{MAX_SEQ_LEN}}"
)

results = lm_eval.simple_evaluate(
    model="torchtitan",
    model_args=model_args,
    tasks=TASKS.split(","),
    num_fewshot=NUM_FEWSHOT,
    limit=LIMIT,
    batch_size=BATCH_SIZE,
    log_samples=LOG_SAMPLES,
)

# Save results
results_path = f"{{OUTPUT_DIR}}/results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\\nResults saved to: {{results_path}}")

# Print summary
print("\\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for task_name, task_results in results.get("results", {{}}).items():
    print(f"\\n{{task_name}}:")
    for metric, value in task_results.items():
        if isinstance(value, (int, float)):
            print(f"  {{metric}}: {{value:.4f}}")
'''

        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

        logger.info(f"Generated eval script: {script_path}")

    def _generate_slurm_script(
        self,
        step: int,
        eval_script_path: Path,
        output_dir: Path,
        slurm_script_path: Path,
    ) -> None:
        """Generate SLURM submission script for evaluation."""
        slurm_cfg = self.lm_eval_config.slurm
        job_name = f"lm_eval_step_{step}"

        # Build PYTHONPATH
        pythonpath_parts = []
        if self.torchtitan_path:
            pythonpath_parts.append(self.torchtitan_path)
        if self.lm_eval_path:
            pythonpath_parts.append(self.lm_eval_path)
        pythonpath = ":".join(pythonpath_parts) if pythonpath_parts else ""

        # Build extra sbatch args
        extra_args = slurm_cfg.extra_sbatch_args

        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task={slurm_cfg.gpus_per_node}
#SBATCH --cpus-per-task={slurm_cfg.cpus_per_task}
#SBATCH --time={slurm_cfg.time}
#SBATCH --partition={slurm_cfg.partition}
#SBATCH --output={self.slurm_logs_dir}/{job_name}_%j.out
#SBATCH --error={self.slurm_logs_dir}/{job_name}_%j.err
"""

        if slurm_cfg.qos:
            script_content += f"#SBATCH --qos={slurm_cfg.qos}\n"
        if slurm_cfg.account:
            script_content += f"#SBATCH --account={slurm_cfg.account}\n"
        if slurm_cfg.reservation:
            script_content += f"#SBATCH --reservation={slurm_cfg.reservation}\n"
        if extra_args:
            script_content += f"#SBATCH {extra_args}\n"

        script_content += f"""
# Auto-generated SLURM script for lm-evaluation-harness
# Generated at: {datetime.now().isoformat()}
# Step: {step}
# To resubmit: sbatch {slurm_script_path}

set -e

echo "=========================================="
echo "LM-Evaluation-Harness SLURM Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Step: {step}"
echo "=========================================="

# Environment setup
export LOGLEVEL=INFO
export FI_PROVIDER="efa"
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export HF_HOME="{slurm_cfg.hf_cache}"
export TRANSFORMERS_CACHE="{slurm_cfg.hf_cache}"
"""

        if pythonpath:
            script_content += f'export PYTHONPATH="{pythonpath}:$PYTHONPATH"\n'

        # Activate virtual environment if specified
        if slurm_cfg.venv_path:
            script_content += f"""
# Activate Python virtual environment
export PATH="{slurm_cfg.venv_path}/bin:$PATH"
export CONDA_PREFIX="{slurm_cfg.venv_path}"
echo "Activated venv: {slurm_cfg.venv_path}"
"""
        elif slurm_cfg.conda_env:
            script_content += f"""
# Activate conda environment
conda activate {slurm_cfg.conda_env}
"""

        script_content += f"""
# Verify python is available
which python || echo "ERROR: python not found in PATH"

# Record start time
START_TIME=$(date +%s)

# Run evaluation
echo "Starting evaluation..."
python {eval_script_path}

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Save status
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Evaluation completed (Duration: ${{DURATION}}s)"
else
    echo "FAILED: Evaluation failed with exit code $EXIT_CODE"
fi

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: {output_dir}"
echo "Duration: ${{DURATION}} seconds"
echo "=========================================="

exit $EXIT_CODE
"""

        with open(slurm_script_path, "w") as f:
            f.write(script_content)

        # Make executable
        slurm_script_path.chmod(0o755)

        logger.info(f"Generated SLURM script: {slurm_script_path}")

    def _log_results_summary(self, step: int, results: dict[str, Any]) -> None:
        """Log a summary of evaluation results."""
        logger.info(f"Evaluation results at step {step}:")

        for task_name, task_results in results.get("results", {}).items():
            metrics_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in task_results.items()
                if isinstance(v, (int, float))
            )
            logger.info(f"  {task_name}: {metrics_str}")

    def check_running_jobs(self) -> dict[int, str]:
        """
        Check status of running evaluation jobs.

        Returns:
            Dict mapping step to status ('running', 'completed', 'failed')
        """
        statuses = {}

        for step, job_info in list(self.running_jobs.items()):
            if job_info["mode"] == "subprocess":
                # Check if process is still running
                try:
                    pid = job_info["pid"]
                    os.kill(pid, 0)  # Check if process exists
                    statuses[step] = "running"
                except ProcessLookupError:
                    # Process finished, check for results
                    results_path = Path(job_info["output_dir"]) / "results.json"
                    if results_path.exists():
                        statuses[step] = "completed"
                        # Load and log results
                        with open(results_path) as f:
                            results = json.load(f)
                        self._log_results_summary(step, results)
                    else:
                        statuses[step] = "failed"
                    del self.running_jobs[step]

            elif job_info["mode"] == "slurm":
                # Check SLURM job status
                job_id = job_info["job_id"]
                result = subprocess.run(
                    ["squeue", "-j", job_id, "-h", "-o", "%T"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0 or not result.stdout.strip():
                    # Job not in queue, check if completed
                    results_path = Path(job_info["output_dir"]) / "results.json"
                    if results_path.exists():
                        statuses[step] = "completed"
                        with open(results_path) as f:
                            results = json.load(f)
                        self._log_results_summary(step, results)
                    else:
                        statuses[step] = "failed"
                    del self.running_jobs[step]
                else:
                    status = result.stdout.strip()
                    if status in ("PENDING", "RUNNING", "CONFIGURING"):
                        statuses[step] = "running"
                    elif status == "COMPLETED":
                        statuses[step] = "completed"
                    else:
                        statuses[step] = "failed"

        return statuses

    def get_latest_results(self) -> dict[int, dict[str, Any]]:
        """
        Get all available evaluation results.

        Returns:
            Dict mapping step to results dict
        """
        results = {}

        for step_dir in self.output_dir.iterdir():
            if step_dir.is_dir() and step_dir.name.startswith("step_"):
                results_path = step_dir / "results.json"
                if results_path.exists():
                    step = int(step_dir.name.replace("step_", ""))
                    with open(results_path) as f:
                        results[step] = json.load(f)

        return results

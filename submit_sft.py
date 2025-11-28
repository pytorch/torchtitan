#!/usr/bin/env python3
"""
Submit SFT training jobs to SLURM, mirroring the axolotl submit workflow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
from rich.console import Console
from rich.panel import Panel


def _format_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def submit(
    config_file: str,
    slurm_script: str = "sft.slurm",
    job_name: str = "sft-train",
    n_nodes: int = 1,
    partition: str = "batch",
    time: str = "72:00:00",
    gpus_per_node: int = 8,
    cpus_per_task: int = 224,
    conda: str | None = "torchtitan",
    venv: str | None = None,
    dry_run: bool = False,
    # WandB configuration
    wandb_team: str | None = "nous_research",
    wandb_project: str | None = "hillclimb",
    wandb_run_name: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Submit an SFT job with environment exports similar to axolotl's submit helper.

    Args:
        config_file: Path to the Torchtitan config YAML.
        slurm_script: SLURM script to submit (default: sft.slurm).
        job_name: SLURM job name.
        n_nodes: Number of training nodes to export to the job.
        partition: SLURM partition to target.
        time: Requested wall-clock limit.
        gpus_per_node: GPUs per node (passed to sbatch for convenience).
        cpus_per_task: CPUs per task.
        conda: Conda environment name to activate (set to "none" to skip).
        venv: Path to a Python virtual environment to activate instead of Conda.
        dry_run: If True, only print the sbatch command without executing.
        wandb_team: WandB team/entity name (default: nous_research).
        wandb_project: WandB project name (default: hillclimb).
        wandb_run_name: WandB run name (default: auto-generated from model+date).
        **kwargs: Additional key=value pairs.
            - Keys starting with ``slurm_`` become extra ``--<flag>=value`` args.
            - Other keys are exported as environment variables (upper-cased).
    """

    console = Console()

    config_path = Path(config_file).expanduser().resolve()
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file '{config_file}' not found.")
        sys.exit(1)

    script_path = Path(slurm_script).expanduser().resolve()
    if not script_path.exists():
        console.print(f"[red]Error:[/red] SLURM script '{slurm_script}' not found.")
        sys.exit(1)

    exports = ["ALL", f"CONFIG_FILE={config_path}"]
    if n_nodes:
        exports.append(f"NUM_TRAINING_NODES={n_nodes}")

    # WandB exports
    if wandb_team:
        exports.append(f"WANDB_TEAM={wandb_team}")
    if wandb_project:
        exports.append(f"WANDB_PROJECT={wandb_project}")
    if wandb_run_name:
        exports.append(f"WANDB_RUN_NAME={wandb_run_name}")

    # Conda / venv handling modeled after submit_multinode.py
    if conda and conda.lower() != "none":
        try:
            conda_base = (
                subprocess.run(
                    ["conda", "info", "--base"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            exports.extend([f"CONDA_ENV={conda}", f"CONDA_BASE={conda_base}"])
            console.print(
                f"[green]Using conda environment:[/green] {conda} (base: {conda_base})"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print(
                f"[yellow]Warning:[/yellow] 'conda' command not found; exporting CONDA_ENV={conda} only."
            )
            exports.append(f"CONDA_ENV={conda}")
    elif venv and venv.lower() != "none":
        venv_path = _format_path(venv)
        exports.append(f"VENV_ENV={venv_path}")
        console.print(f"[green]Using venv:[/green] {venv_path}")

    extra_slurm_args: list[str] = []
    for key, value in kwargs.items():
        if key.startswith("slurm_"):
            param = key[6:].replace("_", "-")
            extra_slurm_args.append(f"--{param}={value}")
        else:
            exports.append(f"{key.upper()}={value}")

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--nodes={n_nodes}",
        f"--gpus-per-node={gpus_per_node}",
        f"--cpus-per-task={cpus_per_task}",
        f"--partition={partition}",
        f"--time={time}",
        f"--export={','.join(exports)}",
        *extra_slurm_args,
        str(script_path),
    ]

    panel_text = "\n".join(
        [
            f"Config: {config_path}",
            f"SLURM script: {script_path}",
            f"Nodes: {n_nodes}",
            f"GPUs per node: {gpus_per_node}",
            f"Partition: {partition}",
            f"Time: {time}",
            f"Job name: {job_name}",
            f"Exports: {exports}",
        ]
    )
    console.print(Panel(panel_text, title="SFT Job Submission"))
    console.print("Command:", " ".join(cmd))

    if dry_run:
        console.print("[cyan]Dry run; not submitting.[/cyan]")
        return

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        console.print("[green]Job submitted successfully![/green]")
        if result.stdout:
            console.print(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Error submitting job:[/red] {exc}")
        if exc.stderr:
            console.print(exc.stderr.strip())
        sys.exit(exc.returncode or 1)


def main() -> None:
    fire.Fire(submit)


if __name__ == "__main__":
    main()

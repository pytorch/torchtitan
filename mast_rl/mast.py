"""
Build a single-host MAST AppDef for Search-R1 RL training.

Adapted from ``fbsource//fbcode/pytorch/torchtitan/fb/mast_rl/mast.py``. The
role's bash command execs ``${WORKSPACE_DIR}/mast_rl/run.sh``, which starts the
dense-retrieval server, then runs the training command. Trainer + generator
share the host's GPUs via ``CUDA_VISIBLE_DEVICES`` partitioning inside
``train.main``; ``run.sh`` reserves the last GPU for the retriever's e5 encoder.
"""

import re

import torchx.specs as specs
from torchx.specs.fb.component_helpers import run_as
from torchx.specs.fb.named_resources import MAST_WHOLE_HOST_FEATURE


# Placeholder image; ``mast_conda`` swaps ``role.image[0]`` for an ephemeral
# fbpkg of the conda-packed env + workspace, keeping the name ``monarch_conda``
# (a CAF namespace we have write access to). The trailing entries are real,
# pre-built fbpkgs MAST downloads as-is: OilFS + Manifold mounts.
_DEFAULT_IMAGE: str = (
    "monarch_conda:latest_conveyor_build"
    ";oil.oilfs:stable"
    ";manifold.manifoldfs:prod"
)


def build_single_host_appdef(
    *,
    script_args: tuple[str, ...],
    job_name: str,
    host_type: str,
    wandb_run_name: str | None = None,
    oilfs_workspace: str = "",
) -> specs.AppDef:
    """Build a single-host MAST AppDef that runs the RL script.

    Args:
        script_args: argv passed to ``run.sh`` (e.g.,
            ``("mast_rl/main.py", "--module", "rl", "--config", ...)``). The
            first element is interpreted as a python script (if it ends in .py)
            or a module name (-m otherwise).
        job_name: MAST job name (must be unique). Caller sanitizes.
        host_type: MAST host type (e.g., ``grandteton_80g_roce``).
        wandb_run_name: human-friendly wandb run name. Defaults to ``job_name``.
        oilfs_workspace: when set, the role mounts ``ws://<oilfs_workspace>`` at
            ``/mnt/$(basename ...)``. When empty, falls back to the default
            Manifold bucket mount (``torchtrain_datasets``).
    """
    assert len(script_args) > 0, "must provide a script to run"

    original_script = script_args[0]
    if not original_script.endswith(".py"):
        full_script_args: tuple[str, ...] = ("python", "-m", *script_args)
    else:
        full_script_args = ("python", *script_args)

    # ``${WORKSPACE_DIR}`` is set on the role by
    # ``monarch.tools.config.Workspace.set_env_vars`` to ``${img_root}/workspace``
    # (substituted to ``/packages/monarch_conda/workspace`` at runtime). The
    # launcher ships this directory under dst ``mast_rl``, so ``run.sh`` lives at
    # ``${WORKSPACE_DIR}/mast_rl/run.sh``.
    bash_cmd = 'exec "${WORKSPACE_DIR}/mast_rl/run.sh" "$@"'

    # wandb: OFFLINE on MAST (no runtime egress dependency -> can't hang training).
    # Entity is set to the user's own (yichuan) via WANDB_TEAM (torchtitan's
    # WandBLogger reads entity from WANDB_TEAM). The offline run is written to the
    # bucket; wandb_autosync.sh on the devvm pushes it to
    # meta.wandb.io/yichuan/torchtitan every few minutes.
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "offline",
        "WANDB_TEAM": "yichuan",
        "WANDB_PROJECT": "torchtitan",
        "WANDB_RUN_NAME": wandb_run_name or job_name,
        "TORCH_CPP_LOG_LEVEL": "ERROR",
        "LOGLEVEL": "ERROR",
    }
    # File-service mount on the MAST node: run.sh mounts OilFS at
    # ws://<oilfs_workspace> if set, otherwise the default Manifold bucket
    # (where the Search-R1 index/corpus/parquet + checkpoints are staged).
    if oilfs_workspace:
        env["OILFS_WORKSPACE"] = oilfs_workspace
    else:
        env["MANIFUSE_BUCKET"] = "torchtrain_datasets"

    appdef = specs.AppDef(name=job_name)
    role = specs.Role(
        name="script",
        image=_DEFAULT_IMAGE,
        entrypoint="/bin/bash",
        # ``--`` becomes ``$0`` and the rest becomes ``$@`` for bash_cmd.
        args=["-c", bash_cmd, "--", *full_script_args],
        num_replicas=1,
        resource=specs.resource(h=host_type),
        env=env,
    )
    # Pin the entire host so Monarch can spawn workers across all GPUs and the
    # retriever can claim a spare GPU.
    role.resource.capabilities[MAST_WHOLE_HOST_FEATURE] = True
    # Run as root so the bash command can mkdir under /mnt for the OilFS /
    # Manifold mount (default unix user is ``nobody``).
    run_as(role, root_user=True)
    appdef.roles.append(role)
    appdef.metadata["tags"] = ",".join(
        [
            "torchtitan-rl",
            original_script,
            # monarch-specific tags used by the monarch team for job attribution.
            "monarch:torchtitan-rl",
            "monarch",
        ]
    )
    return appdef


def sanitize_job_name(raw: str) -> str:
    """MAST job names must match ``[a-zA-Z0-9_-]+``."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)

"""Submit a single-host Search-R1 RL training job to MAST from Python.

Adapted from ``fbsource//fbcode/pytorch/torchtitan/fb/mast_rl/launcher.py``.

Uses ``monarch.tools.commands.create`` (the canonical Monarch submit API) with
a ``Workspace`` that conda-packs the active ``rlmast`` env and ships this
directory (``mast_rl/`` -- run.sh + main.py + retrieval_server.py) as a
workspace dir. torchtitan itself is installed into the conda env by
``build_conda.sh`` (non-editable ``pip install`` from the OSS checkout) so it
travels with the conda-pack rather than as a workspace dir. ``mast_conda``
replaces the placeholder ``monarch_conda:latest_conveyor_build`` image with the
resulting ephemeral fbpkg.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

# Note: ``mast`` resolves to ``mast_rl/mast.py`` because Python adds this
# script's directory to ``sys.path[0]``.
from mast import build_single_host_appdef, sanitize_job_name
from monarch.tools import commands
from monarch.tools.config import Config, Workspace
from monarch.tools.config.workspace import ACTIVE_CONDA_ENV
from torchx.specs import Workspace as TorchxWorkspace


logger: logging.Logger = logging.getLogger(__name__)


class _AllRolesWorkspace(Workspace):
    """``Workspace`` subclass that attaches itself to every role.

    Default ``Workspace.set_env_vars`` populates env vars on each role but only
    sets ``role.workspace`` on the first one, so multi-role apps don't trigger
    the ``caching_build_workspace_and_update_role`` flow for the trailing roles.
    We override to set ``role.workspace`` on all of them.
    """

    def __init__(self, dirs=None, env=None) -> None:
        super().__init__(dirs=dirs, env=env)
        self._workspace_out: str | None = None

    def merge(self, outdir) -> None:
        self._workspace_out = str(outdir)
        super().merge(outdir)

    def set_env_vars(self, appdef) -> None:
        super().set_env_vars(appdef)
        if self._workspace_out and self.dirs:
            ws = TorchxWorkspace.from_str(self._workspace_out)
            for role in appdef.roles:
                role.workspace = ws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-name",
        default="",
        help=(
            "human-friendly job name prefix; a 6-char uuid is appended "
            "to disambiguate multiple submissions"
        ),
    )
    parser.add_argument("--host-type", default="grandteton_80g_roce")
    parser.add_argument(
        "--oilfs",
        default="",
        help=(
            "if set, mount ws://<value> on the MAST host instead of the "
            "default Manifold bucket"
        ),
    )
    # MAST scheduler args. Defaults reuse the existing fb/mast_rl entitlement.
    parser.add_argument("--hpc-identity", default="genai_llm_research-llama")
    parser.add_argument("--rm-attribution", default="msl_infra_pytorch_dev")
    parser.add_argument("--hpc-oncall", default="meta_conda")
    parser.add_argument(
        "--region",
        default="eag",
        help=(
            "MAST region option (e.g., 'eag', 'pci'); passed as the second "
            "element of localityConstraints alongside the hardcoded 'region' "
            "locality type"
        ),
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="forwarded to run.sh on the MAST host (typically: main.py ...)",
    )
    args = parser.parse_args()
    if args.script_args and args.script_args[0] == "--":
        args.script_args = args.script_args[1:]
    assert args.script_args, "must provide a script and its args after `--`"

    # MAST caps job names at 76 chars. Use ``<prefix>-<6-char-uuid>`` so resubmits
    # of the same ``--job-name`` don't collide; pre-truncate the prefix so the
    # uuid we picked is preserved end-to-end.
    suffix = uuid.uuid4().hex[:6]
    prefix = sanitize_job_name(args.job_name) if args.job_name else "torchtitan-rl"
    prefix = prefix[:69]  # 76 - 7 (``-`` + 6-char suffix)
    job_name = f"{prefix}-{suffix}"
    wandb_run_name = args.job_name or job_name

    appdef = build_single_host_appdef(
        script_args=tuple(args.script_args),
        job_name=job_name,
        host_type=args.host_type,
        wandb_run_name=wandb_run_name,
        oilfs_workspace=args.oilfs,
    )

    # Ship only this directory (``mast_rl/``) -- run.sh, main.py and
    # retrieval_server.py live here. torchtitan rides along in the conda-packed
    # env (non-editable install in ``build_conda.sh``). On MAST the dir lands at
    # ``/packages/monarch_conda/workspace/mast_rl/``.
    mast_rl_root = Path(__file__).resolve().parent
    scheduler_args: dict[str, object] = {
        "hpcIdentity": args.hpc_identity,
        "rmAttribution": args.rm_attribution,
        "hpcJobOncall": args.hpc_oncall,
        "hpcClusterUuid": "MastGenAICluster",
        "localityConstraints": ["region", args.region],
    }
    config = Config(
        scheduler="mast_conda",
        scheduler_args=scheduler_args,
        appdef=appdef,
        workspace=_AllRolesWorkspace(
            dirs={mast_rl_root: "mast_rl"},
            env=ACTIVE_CONDA_ENV,
        ),
    )
    server_handle = commands.create(config, name=job_name)
    print(f"submitted: {server_handle}")
    print(f"https://www.internalfb.com/intern/mast/job/{job_name}")


if __name__ == "__main__":
    main()

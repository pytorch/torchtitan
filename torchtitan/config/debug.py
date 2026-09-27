# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Debug configuration."""

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class DebugConfig:
    seed: int | None = None
    """Choose the base RNG seed used for training"""

    spmd_typechecking: bool = False
    """Enable global SPMD type checking."""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""

    deterministic_warn_only: bool = False
    """Only warns about ops without deterministic implementations rather than erroring out  """

    moe_force_load_balance: bool = False
    """If True, we force each experts to get the same amount of tokens via round-robin. This option is for debugging usage only."""

    detect_anomaly: bool = False
    """Enable torch.autograd anomaly detection to help track down NaN/Inf gradients.
    Note: incurs significant overhead; for debugging only."""

    batch_invariant: bool = False
    """Enable batch-invariant mode to use batch-invariant ops in model
    forward and deterministic NCCL collective reduction order"""

    print_config: bool = False
    """Print the job configs to terminal"""

    save_config_file: str | None = None
    """Path to save job config into"""

    enable_structured_logging: bool = True
    """Whether to enable the structured per-rank trace logger (see
    ``torchtitan.observability.structured_logger``). When False, all
    ``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar`` calls
    are no-ops. Disable to fully eliminate trace overhead."""

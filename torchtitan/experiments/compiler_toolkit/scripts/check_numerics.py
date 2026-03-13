# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import numerics_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.numerics_utils import run_numerics_test


def main():
    parser = argparse.ArgumentParser(
        description="Run two training jobs and compare their tensorboard metrics"
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        required=True,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--dp-shard-degree",
        type=int,
        default=1,
        help="Data parallel shard degree",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=1,
        help="Tensor parallel degree",
    )
    parser.add_argument(
        "--cp-degree",
        type=int,
        default=1,
        help="Context parallel degree",
    )
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=1,
        help="Expert parallel degree",
    )
    parser.add_argument(
        "--ac-mode",
        type=str,
        default="selective",
        choices=["selective", "none", "full"],
        help="Activation checkpoint mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic training",
    )
    parser.add_argument(
        "--eager-tb-folder",
        type=str,
        default="tb/eager_run",
        help="Tensorboard folder for eager run",
    )
    parser.add_argument(
        "--compiled-tb-folder",
        type=str,
        default="tb/compiled_run",
        help="Tensorboard folder for compiled run",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["loss_metrics/global_avg_loss", "grad_norm"],
        help="Metrics to compare",
    )
    parser.add_argument(
        "--passes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of compiler passes to apply "
            "(e.g., 'autobucketing_reordering' or 'autobucketing_reordering,regional_inductor')"
        ),
    )

    args = parser.parse_args()

    success = run_numerics_test(
        ngpu=args.ngpu,
        config_file=args.config_file,
        dp_shard_degree=args.dp_shard_degree,
        tp_degree=args.tp_degree,
        cp_degree=args.cp_degree,
        ep_degree=args.ep_degree,
        ac_mode=args.ac_mode,
        steps=args.steps,
        seed=args.seed,
        eager_tb_folder=args.eager_tb_folder,
        compiled_tb_folder=args.compiled_tb_folder,
        metrics=args.metrics,
        passes=args.passes,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

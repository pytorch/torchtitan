# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchrun worker entry point for ladder runs.

Launched per rank by the launcher as ``torchrun ... -m
torchtitan.experiments.scaling_ladders.train --config-file PATH``. The launcher
has already resolved the schedule and written the full build spec (plan + knobs)
to that JSON file, so the worker just loads it and trains -- no config is rebuilt
from argv. See launcher.run_from_spec.
"""

import argparse

from .launcher import run_from_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ladder rung from a spec.")
    parser.add_argument(
        "--config-file", required=True, help="Path to the launcher's JSON build spec."
    )
    run_from_spec(parser.parse_args().config_file)


if __name__ == "__main__":
    main()

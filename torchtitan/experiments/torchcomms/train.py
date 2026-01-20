# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchComms training entry point.

This module provides backward compatibility for running training with torchcomms.
The recommended approach is to use `--parallelism.collective_api=torchcomms` or
`--parallelism.collective_api=torchcomms_via_process_group` with the base Trainer.
This module exists for users who want to run `torchtitan/experiments/torchcomms/train.py`
directly and defaults to torchcomms_via_process_group for backward compatibility.
"""

from torchtitan.config import JobConfig
from torchtitan.train import main, Trainer


class TorchCommsTrainer(Trainer):
    """Trainer that forces torchcomms_via_process_group collective API.

    This class ensures that torchcomms is used regardless of the config setting,
    providing backward compatibility for users running this module directly.
    Uses torchcomms_via_process_group (TORCHCOMMS_PATCH_FOR_COMPILE=0) by default.
    """

    def __init__(self, job_config: JobConfig):
        # Force torchcomms_via_process_group for backward compatibility
        # (original behavior used init_device_mesh which corresponds to PATCH=0)
        if job_config.parallelism.collective_api == "process_group":
            job_config.parallelism.collective_api = "torchcomms_via_process_group"
        super().__init__(job_config)


if __name__ == "__main__":
    main(TorchCommsTrainer)

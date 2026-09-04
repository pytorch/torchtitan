# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan_recipes.tests.b200 as recipes

from tests.integration_tests import OverrideDefinitions


def build_b200_tests_list() -> list[OverrideDefinitions]:
    """Build integration tests that require B200-class hardware."""
    return [
        OverrideDefinitions(
            configs=[recipes.kimi_k3_debugmodel_mm_fsdp2],
            test_descr="Kimi K3 multimodal FSDP",
            test_name="kimi_k3_mm_fsdp",
            ngpu=2,
        ),
        # TODO: re-enable once the B200 job installs torchao. It currently
        # installs only nightly torch/torchvision, requirements.txt and
        # requirements-vlm.txt, none of which pull torchao in, so MXFP8Linear
        # cannot import and MXFP8LinearConverter raises at construction.
        # A plain `pip install torchao` is not enough either: the 32x32
        # swizzled cast kernels landed in pytorch/ao#4777 and are unreleased as
        # of v0.18.0, so this needs a source install or a later release.
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_mxfp8_fsdp2],
            test_descr="MXFP8 linear with an FSDP-managed weight cache",
            test_name="mxfp8_linear_fsdp",
            ngpu=2,
            disabled=True,
        ),
    ]

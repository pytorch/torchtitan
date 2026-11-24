# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flux integration-test suite.

Configurations live in ``torchtitan_recipes.tests.models.flux``.
"""

import torchtitan_recipes.tests.models.flux as recipes

from tests.integration_tests import IntegrationTestDefinition

FLUX_INFERENCE_ENTRYPOINT = "torchtitan.models.flux.inference.infer"


def build_flux_test_list() -> list[IntegrationTestDefinition]:
    """Build the list of Flux integration tests."""
    return [
        IntegrationTestDefinition(
            # Train and save a checkpoint, then generate images from it.
            configs=[
                recipes.flux_debugmodel_hsdp2x2_cp2_validation,
                recipes.flux_debugmodel_inference_test,
            ],
            entrypoints=["torchtitan.train", FLUX_INFERENCE_ENTRYPOINT],
            test_descr="HSDP+CP+Validation+Inference",
            test_name="hsdp+cp+validation+inference",
            ngpu=8,
            use_real_pg=True,
        ),
    ]

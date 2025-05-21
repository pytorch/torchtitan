# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Import to register quantization modules.
import torchtitan.components.quantization  # noqa: F401

# Import the built-in models here so that the corresponding register_model_spec()
# will be called.
import torchtitan.experiments  # noqa: F401
import torchtitan.models  # noqa: F401

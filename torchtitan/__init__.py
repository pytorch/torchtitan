# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Import to register Float8Converter.
import torchtitan.components.float8  # noqa: F401

# Import the built-in models here so that the corresponding register_model_spec()
# will be called.
import torchtitan.models  # noqa: F401

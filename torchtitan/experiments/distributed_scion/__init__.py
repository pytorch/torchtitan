# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .abstract_disco import AbstractDiSCO  # noqa: F401
from .disco import Disco  # noqa: F401
from .utils import (  # noqa: F401  # noqa: F401  # noqa: F401
    create_disco_optimizer_kwargs_from_optimizer_config,
    create_disco_param_groups,
    remove_orig_mod_and_weight_for_p_name,
)

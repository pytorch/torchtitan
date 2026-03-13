# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan.config import Configurable


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.
    Combines nn.Module with Configurable, so subclasses only inherit from Module.

    Subclasses with learnable parameters should override ``init_weights``.
    The default implementation is a no-op, which is appropriate for modules
    that have no learnable parameters or are loaded from external checkpoints.
    """

    def init_weights(self, **kwargs) -> None:
        """Initialize weights. Override in subclasses with learnable parameters."""
        pass

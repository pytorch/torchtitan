# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import torch.nn as nn

from torchtitan.config import Configurable


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.
    Combines nn.Module with Configurable, so subclasses only inherit from Module.

    All Module subclasses must implement ``init_weights``.
    """

    @abstractmethod
    def init_weights(self, **kwargs) -> None:
        """Initialize weights. Subclasses define specific signature."""
        ...

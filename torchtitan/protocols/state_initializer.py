# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import Configurable


class StateInitializer(Configurable):
    """Base class for weight/state initialization strategies.

    A ``StateInitializer`` is a plain ``Configurable`` (not an ``nn.Module``)
    that owns init-related config fields and implements an ``init_states``
    method to initialize the parameters of a target module.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        pass

    @abstractmethod
    def init_states(self, module: nn.Module, *, buffer_device=None) -> None:
        raise NotImplementedError


class NoOpStateInitializer(StateInitializer):
    """A no-op state initializer that does nothing.

    Used as the default for ``Module.Config.state_initializer`` so that
    Module subclasses that override ``init_states()`` directly (e.g. Flux
    model classes) don't need to provide a custom ``StateInitializer``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        pass

    def init_states(self, module: nn.Module, *, buffer_device=None) -> None:
        pass

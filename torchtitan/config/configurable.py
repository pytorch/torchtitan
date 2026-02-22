# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields
from typing import ClassVar


class Configurable:
    """Base class for all configurable components.

    Every configurable class:
    - Inherits from Configurable (or Module for nn.Module components)
    - Defines a nested Config(Configurable.Config) with @dataclass(kw_only=True, slots=True)
    - Gets build() auto-wired via __init_subclass__ (no manual override needed)
    - Accepts __init__(self, config: Config, **runtime_kwargs)

    Enforcement: Configurable.__init_subclass__ checks that every Config uses
    @dataclass(kw_only=True, slots=True). This check runs on the OUTER class
    (not Config.__init_subclass__) because @dataclass(slots=True) replaces the
    class, so Config.__init_subclass__ sees the pre-decorator version.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        _owner: ClassVar[type | None] = None

        def build(self, **kwargs):
            """Construct the owning class. Auto-wired by __init_subclass__."""
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            return self._owner(config=self, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if issubclass(config_cls, Configurable.Config):
                # Enforce @dataclass(kw_only=True, slots=True)
                if "__slots__" not in config_cls.__dict__:
                    raise TypeError(
                        f"{cls.__name__}.Config must use "
                        "@dataclass(kw_only=True, slots=True)"
                    )
                for f in fields(config_cls):
                    if not f.kw_only:
                        raise TypeError(
                            f"{cls.__name__}.Config field '{f.name}' "
                            "must be keyword-only"
                        )
                # Auto-wire build() to construct this class
                config_cls._owner = cls

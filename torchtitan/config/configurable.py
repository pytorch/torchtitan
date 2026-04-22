# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from typing import ClassVar

logger = logging.getLogger(__name__)


class Configurable:
    """Base class for all configurable components.

    Every configurable class:
    - Inherits from Configurable (or Module for nn.Module components)
    - Defines a nested Config(Configurable.Config) with @dataclass(kw_only=True, slots=True)
    - Gets build() auto-wired via __init_subclass__ (no manual override needed)
    - Accepts ``__init__(self, config: Config)``

    build() has two modes:
    - No kwargs: ``self._owner(config=replace(self))``
    - With kwargs (runtime objects not in config): forwarded to
      ``self._owner(config=..., **kwargs)``.  Used by non-model
      Configurables (tokenizer, dataloader, optimizer, etc.)
      that receive runtime objects at construction time.

    Enforcement: Configurable.__init_subclass__ checks that every Config uses
    @dataclass(kw_only=True, slots=True). This check runs on the OUTER class
    (not Config.__init_subclass__) because @dataclass(slots=True) replaces the
    class, so Config.__init_subclass__ sees the pre-decorator version.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        """Base config class for all configurable components."""

        _owner: ClassVar[type | None] = None

        def to_dict(self) -> dict:
            """Serialize config to a plain dict (recursing into nested configs)."""

            def _convert(val):
                if hasattr(val, "to_dict"):
                    return val.to_dict()
                elif dataclasses.is_dataclass(val):
                    return dataclasses.asdict(val)
                elif isinstance(val, (list, tuple)):
                    return type(val)(_convert(v) for v in val)
                elif isinstance(val, dict):
                    return {k: _convert(v) for k, v in val.items()}
                elif isinstance(val, (str, int, float, bool, type(None))):
                    return val
                elif callable(val):
                    return repr(val)
                else:
                    logger.warning(
                        f"Config field value of type {type(val).__name__} "
                        f"may not be JSON serializable"
                    )
                    return repr(val)

            return {f.name: _convert(getattr(self, f.name)) for f in fields(self)}

        def walk(
            self, config_cls: type, *, _prefix: str = ""
        ) -> Iterator[tuple[str, "Configurable.Config", object, str]]:
            """Yield ``(fqn, config, parent, field_name)`` for every nested config of *config_cls*.

            Recursively walks dataclass fields, including items inside lists.
            The *fqn* mirrors the module FQN that ``build()`` would produce
            (e.g. ``"layers.0.feed_forward.w1"``).

            *parent* and *field_name* allow replacing the config in the tree::

                for fqn, cfg, parent, attr in model_config.walk(Linear.Config):
                    setattr(parent, attr, NewConfig(...))
            """
            for f in fields(self):
                val = getattr(self, f.name)
                fqn = f"{_prefix}.{f.name}" if _prefix else f.name
                if isinstance(val, config_cls):
                    yield fqn, val, self, f.name
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        item_fqn = f"{fqn}.{i}"
                        if isinstance(item, config_cls):
                            yield item_fqn, item, val, i
                        elif hasattr(item, "walk"):
                            yield from item.walk(config_cls, _prefix=item_fqn)
                elif hasattr(val, "walk"):
                    yield from val.walk(config_cls, _prefix=fqn)

        def build(self, **kwargs):
            """Construct the owning class. Auto-wired by __init_subclass__.

            Two modes:
            - No kwargs: ``self._owner(config=replace(self))``
            - With kwargs (runtime objects not in config): forwarded to
              ``self._owner(config=..., **kwargs)``.  Used by non-model
              Configurables (tokenizer, dataloader, optimizer, etc.)
              that receive runtime objects at construction time.
            """
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            if not kwargs:
                return self._owner(config=replace(self))

            config_fields = {f.name for f in fields(self)}
            overlap = config_fields & kwargs.keys()
            if overlap:
                raise ValueError(
                    f"build() kwargs {overlap} overlap with config fields. "
                    "Put these values in the Config, not in build() kwargs."
                )
            return self._owner(config=replace(self), **kwargs)

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
                    if f.init and not f.kw_only:
                        raise TypeError(
                            f"{cls.__name__}.Config field '{f.name}' "
                            "must be keyword-only"
                        )
                # Auto-wire build() to construct this class
                config_cls._owner = cls

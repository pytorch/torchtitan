# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
from dataclasses import dataclass, fields, replace
from typing import ClassVar

logger = logging.getLogger(__name__)

import torch


class Configurable:
    """Base class for all configurable components.

    Every configurable class:
    - Inherits from Configurable (or Module for nn.Module components)
    - Defines a nested Config(Configurable.Config) with @dataclass(kw_only=True, slots=True)
    - Gets build() auto-wired via __init_subclass__ (no manual override needed)
    - Accepts __init__(self, config: Config) or __init__(self, config: Config, **runtime_kwargs)
      We will deprecate the later usage once we migrate all components to make all
      required fields in config.

    build() auto-detects kwargs mode:
    - All kwargs are config fields -> absorbed into a cloned config, __init__() only gets
      config arg.
    - All kwargs are NOT config fields -> forwarded to __init__() as keyword arguments.
      This is the legacy style.
    - Mixed -> raises TypeError.

    Fields that are supplied at ``build()`` time should use ``field(init=False)`` so they
    are excluded from ``Config.__init__()``.
    Pre-set values (via attribute assignment or inheritance) that conflict with a
    ``build()`` kwarg raise ``ValueError``; matching values are accepted.

    Enforcement: Configurable.__init_subclass__ checks that every Config uses
    @dataclass(kw_only=True, slots=True). This check runs on the OUTER class
    (not Config.__init_subclass__) because @dataclass(slots=True) replaces the
    class, so Config.__init_subclass__ sees the pre-decorator version.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        """Base config class for all configurable components.

        .. warning::

            Do **not** use ``dataclasses.asdict()`` on ``Config`` instances.
            Configs may contain ``field(init=False)`` slots that are only populated at
            ``build()`` time; ``asdict()`` will raise ``AttributeError`` for those fields.
            Use :meth:`to_dict` instead.
        """

        _owner: ClassVar[type | None] = None

        def __repr__(self) -> str:
            """Safe repr that handles unset ``field(init=False)`` slots.

            The default dataclass ``__repr__`` raises ``AttributeError`` when
            ``field(init=False)`` slots have not been set yet.  This override
            shows ``<UNSET>`` for those fields instead of crashing, which lets
            external libraries (e.g. tyro) safely print configs.
            """
            cls_name = type(self).__name__
            parts: list[str] = []
            for f in fields(self):
                try:
                    val = getattr(self, f.name)
                except AttributeError:
                    parts.append(f"{f.name}=<UNSET>")
                    continue
                parts.append(f"{f.name}={val!r}")
            return f"{cls_name}({', '.join(parts)})"

        def to_dict(self) -> dict:
            """Serialize to a dict, safely handling unset ``field(init=False)`` slots."""

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
                else:
                    logger.warning(
                        f"Config field value of type {type(val).__name__} "
                        f"may not be JSON serializable"
                    )
                    return val

            result = {}
            for f in fields(self):
                try:
                    val = getattr(self, f.name)
                except AttributeError:
                    # field(init=False) not yet set, ignore this field.
                    continue
                result[f.name] = _convert(val)
            return result

        def _replace(self, **overrides):
            """Copy this config via ``replace()``, apply *overrides* to every
            ``field(init=False)`` slot, and validate that every
            ``field(init=False)`` slot has been set.

            Raises ``TypeError`` if any ``init=False`` field is neither
            pre-set on *self* nor supplied in *overrides*.
            """
            clone = replace(self)
            for f in fields(self):
                if f.init:
                    continue

                if f.name in overrides:
                    setattr(clone, f.name, overrides[f.name])
                elif hasattr(self, f.name):
                    setattr(clone, f.name, getattr(self, f.name))
                else:
                    raise TypeError(
                        f"{type(self).__name__} field '{f.name}' "
                        f"(init=False) was not provided via build()"
                    )
            return clone

        def build(self, **kwargs):
            """Construct the owning class. Auto-wired by __init_subclass__."""
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            if not kwargs:
                return self._owner(config=self._replace())

            config_field_names = {f.name for f in fields(self)}
            kwargs_in_config = set(kwargs) & config_field_names
            kwargs_not_in_config = set(kwargs) - config_field_names

            if kwargs_in_config and kwargs_not_in_config:
                raise TypeError(
                    f"{type(self).__name__}.build() kwargs must either all be "
                    f"config fields or all be constructor arguments. Got config "
                    f"fields {kwargs_in_config} mixed with non-config fields "
                    f"{kwargs_not_in_config}."
                )

            if kwargs_in_config:
                # All kwargs are config fields: validate and absorb into clone.
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        existing = getattr(self, key)
                        if isinstance(existing, torch.Tensor):
                            mismatch = not torch.equal(existing, value)
                        else:
                            mismatch = existing != value
                        if mismatch:
                            raise ValueError(
                                f"{type(self).__name__}.build() conflict for "
                                f"'{key}': config has {existing!r} "
                                f"but got {value!r}"
                            )
                return self._owner(config=self._replace(**kwargs))

            # TODO: Old style, will be deprecated.
            return self._owner(config=self._replace(), **kwargs)

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
                # Override @dataclass-generated __repr__ with our safe
                # version that handles unset field(init=False) slots.
                config_cls.__repr__ = Configurable.Config.__repr__
                # Auto-wire build() to construct this class
                config_cls._owner = cls

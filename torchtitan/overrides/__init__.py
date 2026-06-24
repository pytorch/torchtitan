# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
In-repo override implementations.

Each module here registers one or more overrides via the ``@override``
decorator from ``torchtitan.config``. These are held to a lower code-quality
bar than core (they may be hardware-specific or experimental) and are opt-in:
nothing here runs unless the user lists the module in ``--override.imports``.

External packages (e.g. hardware vendors) follow the same pattern in their own
namespace; this folder is the in-repo example of the convention.

See ``torchtitan/overrides/README.md``.
"""

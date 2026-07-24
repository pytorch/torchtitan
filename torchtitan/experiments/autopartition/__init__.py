# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "get_deepseek_v3_train_spec",
    "get_llama3_train_spec",
]


from .deepseek_v3_tain_spec import get_deepseek_v3_train_spec
from .llama3_tain_spec import get_llama3_train_spec

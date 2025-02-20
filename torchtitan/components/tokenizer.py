# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    # basic tokenizer interface, for typing purpose mainly
    def __init__(self, tokenizer_path: str):
        assert os.path.exists(
            tokenizer_path
        ), f"The tokenizer path does not exist: {tokenizer_path}"
        assert os.path.isfile(tokenizer_path), tokenizer_path
        self._n_words = 8

    @abstractmethod
    def encode(self, *args, **kwargs) -> List[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @property
    def n_words(self) -> int:
        return self._n_words

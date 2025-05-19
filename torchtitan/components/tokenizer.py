# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod


class Tokenizer(ABC):
    # basic tokenizer interface, for typing purpose mainly
    def __init__(self):
        self._n_words = 8
        self.eos_id = 0

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @property
    def n_words(self) -> int:
        return self._n_words

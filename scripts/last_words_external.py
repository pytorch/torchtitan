# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import runpy

_implementation = runpy.run_module("torchtitan.last_words.external")
globals().update(
    {key: value for key, value in _implementation.items() if key != "__name__"}
)

if __name__ == "__main__":
    main()

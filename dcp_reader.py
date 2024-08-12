# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed.checkpoint as dcp

reader = dcp.FileSystemReader("./outputs/checkpoint/step-0")
print(reader.read_data())

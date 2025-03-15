# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def cross_entropy_loss_hf(preds, labels):
    loss = torch.nn.functional.cross_entropy(
        preds[0].flatten(0, 1).float(), labels.flatten(0, 1)
    )
    return loss

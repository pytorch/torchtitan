# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan RL: async rollout generation with vLLM and policy training.

Importing this package has no side effects. Entry points
(``torchtitan.rl.train``, ``torchtitan.rl.generate``) apply the RL process
environment defaults from ``torchtitan.rl._env`` before importing torch.
"""

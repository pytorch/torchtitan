# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.passes import AVAILABLE_MEMORY_POLICIES

from .paged_stash_graph_pass import (
    paged_stash_save_only_tag_policy,
    paged_stash_tag_policy,
)

AVAILABLE_MEMORY_POLICIES["paged_stash"] = paged_stash_tag_policy
AVAILABLE_MEMORY_POLICIES["paged_stash_save_only"] = paged_stash_save_only_tag_policy
AVAILABLE_MEMORY_POLICIES["paged_stash_spillover"] = paged_stash_tag_policy
AVAILABLE_MEMORY_POLICIES["paged_stash_overflow_test"] = paged_stash_tag_policy

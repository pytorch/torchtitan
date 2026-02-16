# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.experiments.ft.manager import FTManager


@dataclass(kw_only=True, slots=True)
class FaultTolerance(FTManager.Config):
    """
    Extends FTManager.Config to also support Streaming DiLoCo
    """

    sync_steps: int = 5
    """
    Number of steps to wait before performing synchronization. This is only used when "semi_sync_method"
    is set.
    """

    should_quantize: bool = False
    """
    Whether to quantize the gradients before allreduce.

    Disabled by default since the quantization does utilize the GPU
    and uses more collectives. Enabling this requires knowing about
    the tradeoffs between GPU utilization and communication.


    This is only used when "semi_sync_method" is set.
    """

    fragment_sync_delay: int = 0
    """
    Controls the number of inner steps to wait before blocking on a
    model fragment's synchronization. This is the "tao" parameter in
    the Streaming DiLoCo paper.

    By default, each model fragment will be synced at the same step
    at which the allreduce is issued. Enabling delay can improve
    communication and computation overlap, but at the cost of compromising
    model quality

    This is only used when "semi_sync_method" is set.
    """

    fragment_update_alpha: float = 0.0
    """
    Determines how to mix the local and global optimized parameters

    By default, we just use the global parameters. This ensures all
    DDP replicas have the same parameters after synchronizing on
    the fragment. Tuning this can also affect the model quality.

    This is only used when "semi_sync_method" is set.
    """

    module_fqns_per_model_fragment: list[list[str]] = field(default_factory=list)
    """
    Specify a list of lists containing the FQNs (Fully Qualified Names) of modules for each model fragment.
    Each inner list represents one model fragment and contains the module names that belong to that fragment.
    e.g. [['tok_embeddings', 'layers.0'], ['layers.1', 'layers.2'], ['layers.3', 'layers.4']]
    will create 3 chunks: the first containing tok_embeddings and layers.0,
    the second containing layers.1 and layers.2, and the third containing layers.3 and layers.4.
    """

    num_fragments: int = 1
    """
    Number of fragments to split the model into. This is only used when "semi_sync_method" is "diloco".
    This is used to automatically split the model into fragments provided that the model
    implements FaultTolerantModelSpec
    """


@dataclass
class JobConfig:
    fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)

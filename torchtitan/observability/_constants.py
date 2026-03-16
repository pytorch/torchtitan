# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Logger names and markers for the observability module."""

# Separate loggers so system events (phase timing) and experiment metrics
# (loss, reward) go to independent JSONL files with independent formatters.
SYSTEM_LOGGER_NAME = "torchtitan.observability.system"
EXPERIMENT_LOGGER_NAME = "torchtitan.observability.experiment"

# Key set on LogRecord.extra to distinguish record_metric entries in
# ExperimentJSONFormatter.
_METRIC_ENTRY = "_metric_entry"

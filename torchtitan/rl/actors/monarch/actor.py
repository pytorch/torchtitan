# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class of every torchtitan RL Monarch actor."""

from __future__ import annotations

import logging

from monarch.actor import (
    Actor as MonarchActor,
    InvalidReferenceReason,
    UndeliverableMessageEnvelope,
)

logger = logging.getLogger(__name__)


class Actor(MonarchActor):
    """A Monarch ``Actor`` that drops messages it sends to a stopped or failed actor."""

    # Sync, not async: Monarch rejects an async hook on an actor with sync endpoints.
    def _handle_invalid_reference(self, message: UndeliverableMessageEnvelope) -> bool:
        """By default, Monarch fails the sender of such a message. That is wrong when nobody waits for
        it: a generator replying to ``generate`` after the rollout worker that called it stopped at
        shutdown would fail the job. A call request to a dead actor still fails that call's future
        through supervision, so dropping does not turn it into a hang, and the dead actor's own
        failure still reaches its owner. Other delivery failures keep Monarch's defaults.
        """
        failure = message.root_delivery_failure()
        if failure is None or failure.invalid_reference_reason not in (
            InvalidReferenceReason.ActorStopped,
            InvalidReferenceReason.ActorFailed,
        ):
            return False
        logger.debug(
            "Dropping a message to %s: %s",
            failure.target,
            failure.invalid_reference_reason,
        )
        return True

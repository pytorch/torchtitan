# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest session hooks for the RL tests.

Arms a faulthandler watchdog (opt-in via ``RL_TEST_FAULTHANDLER_TIMEOUT``, in
seconds) so a deadlocked rank dumps every thread's traceback and exits instead of
hanging forever. The multi-rank torchrun bitwise-parity job otherwise hangs until
the CI job hits its ~60 min limit and is force-cancelled, which discards all
streamed logs; the watchdog converts that silent hang into a normal failure with
a stack trace that pinpoints the stuck rank. It fires from a dedicated thread, so
it catches both in-test collective deadlocks and teardown/exit hangs. No effect
unless the env var is set, so local runs are untouched.
"""

import faulthandler
import os

_WATCHDOG_SECONDS = int(os.environ.get("RL_TEST_FAULTHANDLER_TIMEOUT", "0"))
if _WATCHDOG_SECONDS > 0:
    faulthandler.dump_traceback_later(_WATCHDOG_SECONDS, exit=True)

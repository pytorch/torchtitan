# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import getpass
import logging
import multiprocessing
import os
import pdb
import subprocess
import sys
import time

import torch.distributed as dist

logger = logging.getLogger()

__all__ = ["set_trace"]


_stdin = [None]
_stdin_lock = multiprocessing.Lock()
try:
    _stdin_fd = sys.stdin.fileno()
except Exception:
    _stdin_fd = None


class MultiprocessingPdb(pdb.Pdb):
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage:
    insert this in the code you want to debug:
    ```
        from tools.debugging.pdb import set_trace; set_trace()
    ```
    """

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        stdin_bak = sys.stdin
        with _stdin_lock:
            try:
                if _stdin_fd is not None:
                    if not _stdin[0]:
                        _stdin[0] = os.fdopen(_stdin_fd)  # type: ignore[arg-type]
                    sys.stdin = _stdin[0]
                self.cmdloop()
            finally:
                sys.stdin = stdin_bak


def set_trace():
    pdb = MultiprocessingPdb()
    pdb.set_trace(sys._getframe().f_back)


def set_trace_rank0():
    """
    `set_trace_rank0` will break the program at rank 0 and will sleep the
    program on other processes. The next, `n` step functionality works well
    to some extent on rank 0, however the continue, `c` functionality does
    not work well as there are subsequent syncs across processes which are
    not handled.
    """
    rank = dist.get_rank()
    if rank == 0:
        logger.info(f"Process rank: {rank} set_trace_rank0, press n and debug")
        set_trace()
    else:
        while True:
            logger.info(f"Process rank: {rank} sleeping for 10mins")
            time.sleep(600)  # sleep for 10 mins


_IS_DEBUGGING = False


def vscode_debug(
    port: int = 5678,
    breakpoint: bool = True,
    kill_zombie_debugpy: bool = True,
    max_port_attempts: int = 10,
):
    """Break the program and wait for a debugger to attach to the program.

    This function will break the program and wait for a debugger to attach on the given port.
    To attach:
        1. Go to the debug panel in VS Code (provided via `[Meta-Internal] Python Language Support` extension)
        2. In the top left corner, lect `attach debug` and click green "start" button
    Screenshots with an example: https://github.com/fairinternal/xlformers/pull/3052

    Parameters:
    -----------
    port: Port to listen on for the debugger.
        E.g., you may be able to use different ports to debug multiple processes.
    breakpoint:
        If True, will break the program at the first line of the main function.
        Otherwise, after attaching the debugger, the program will continue to run.
    kill_zombie_debugpy:
        If True, will kill any lingering debugpy processes belonging to the current user.
    max_port_attempts:
        Maximum number of ports to try if the initial port is in use.

    Troubleshooting:
    ----------------
    If you encounter an "Address already in use" error, you can kill zombie debugpy processes:
    >>> pkill -f debugpy
    Or more specifically:
    >>> ps aux | grep debugpy | grep -v grep | awk '{print $2}' | xargs kill -9

    Additional troubleshooting commands:
    >>> ss -tulpn | grep :5678  # Check what's using the port
    >>> ss -tan | grep 5678     # Check for TIME_WAIT states
    >>> netstat -tulpn | grep 5678  # Alternative port check
    """
    global _IS_DEBUGGING
    if _IS_DEBUGGING:
        return
    _IS_DEBUGGING = True

    import debugpy

    if kill_zombie_debugpy:
        username = getpass.getuser()
        # Try multiple approaches to kill zombie debugpy processes
        commands = [
            f"pkill -f debugpy -u {username}",
            f"ps aux | grep {username} | grep debugpy | grep -v grep | awk '{{print $2}}' | xargs -r kill -9",
            "pkill -f 'python.*debugpy'",
        ]

        for command in commands:
            logger.info(f"Attempting to kill debugpy processes via: `{command}`")
            try:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    logger.info("Successfully killed debugpy processes")
                    break
            except Exception as e:
                logger.warning(f"Command failed: {e}")

        # Give processes time to clean up
        time.sleep(2)

    # Try to find an available port
    original_port = port
    for attempt in range(max_port_attempts):
        try:
            logger.info(f"Attempting to listen on port {port}...")

            # Check if port is available before trying to bind
            check_cmd = f"ss -tulpn | grep :{port}"
            result = subprocess.run(
                check_cmd, shell=True, capture_output=True, text=True
            )
            if result.stdout.strip():
                logger.warning(
                    f"Port {port} appears to be in use: {result.stdout.strip()}"
                )
                port += 1
                continue

            debugpy.listen(port)
            logger.info(
                f"Successfully listening on port {port}. Waiting for debugger to attach..."
            )
            break

        except RuntimeError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {port} is in use, trying next port...")
                port += 1
                if attempt == max_port_attempts - 1:
                    logger.error(
                        f"Failed to find available port after {max_port_attempts} attempts"
                    )
                    logger.error("Manual cleanup required. Try running:")
                    logger.error(
                        "  ss -tulpn | grep :5678  # Check what's using the port"
                    )
                    logger.error("  pkill -f debugpy")
                    logger.error("  or")
                    logger.error(
                        "  ps aux | grep debugpy | grep -v grep | awk '{print $2}' | xargs kill -9"
                    )
                    raise RuntimeError(
                        f"Could not find available port starting from {original_port}"
                    )
            else:
                logger.error(
                    f"Unexpected error while trying to listen on port {port}: {e}"
                )
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    debugpy.wait_for_client()
    if breakpoint:
        debugpy.breakpoint()


def vscode_debug_for_rank(
    for_rank: int = 0,
    port: int = 5678,
    breakpoint: bool = True,
    kill_zombie_debugpy: bool = True,
    sleep_seconds: int = 600,
):
    """Break the program at the given rank and will sleep the program on other processes.

    Example usage:
    --------------
    Here we are setting a VSCode debugger breakpoint right after distributed init is done
    This way this function can enable debugging for the given rank and put others to sleep.
    >>> logger.info("Done init of model parallel.")
    >>> from tools.debugging.pdb import vscode_debug_for_rank; vscode_debug_for_rank()
    """
    current_rank = dist.get_rank()
    if current_rank == for_rank:
        logger.info(
            f"Process rank: {current_rank} is going to wait for debugger to attach on port {port}"
        )
        vscode_debug(port, breakpoint, kill_zombie_debugpy)
    else:
        logger.info(f"Process rank: {current_rank} sleeping for {sleep_seconds}s")
        time.sleep(sleep_seconds)


def vscode_debug_for_all_ranks(
    rank_0_port: int = 5678,
    port_step: int = 1000,
    breakpoint: bool = True,
):
    """Break the program on all ranks and wait for a debugger to attach to the program for each."""
    current_rank = dist.get_rank()
    port = rank_0_port + current_rank * port_step
    vscode_debug(port, breakpoint=breakpoint)


def debug(func):
    """Decorator to debug a function, auto-dropping into a pdb session on exceptions.

    The debugger will only trigger if the function exits with an exception, if it handles the exception
    inside of it, the debugger will not be triggered.

    Example
    -------
    >>> @debug
    >>> def foo():
    >>>     print("Foo entry")
    >>>     raise Exception("Foo exception")
    >>>     print("Foo exit")
    >>> foo()  # this will drop into pdb session, with position at the line of the exception
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            # Retrieve the current exception's traceback
            exc_type, exc_value, tb = sys.exc_info()
            # Start the post-mortem debugging session
            pdb.post_mortem(tb)
            # Optionally, re-raise the exception if you want the program to crash after debugging
            raise

    return wrapper

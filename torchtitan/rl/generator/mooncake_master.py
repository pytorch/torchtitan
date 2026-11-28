# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The job's Mooncake Store master, which the controller starts on its host for
the generators' Mooncake Store (SGLang's HiCache storage backend and vLLM's
MooncakeStoreConnector)."""

import ctypes
import ipaddress
import os
import signal
import socket
import subprocess
import time

_MASTER_START_TIMEOUT_S = 60
_PR_SET_PDEATHSIG = 1


def _free_ports(num_ports: int) -> list[int]:
    sockets = [socket.socket() for _ in range(num_ports)]
    try:
        for s in sockets:
            s.bind(("", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


def _host_address() -> str:
    """An address of this host that other hosts can connect to. Prefer an IP:
    with an IP per task, only the task itself resolves its name."""
    fqdn = socket.getfqdn()
    for *_, sockaddr in socket.getaddrinfo(fqdn, None):
        ip = ipaddress.ip_address(sockaddr[0])
        if not (ip.is_loopback or ip.is_link_local):
            return f"[{ip}]" if ip.version == 6 else str(ip)
    return fqdn


def start_master() -> str:
    """Start ``mooncake_master`` on this host, for the lifetime of this process.
    Returns the ``host:port`` address generators connect to."""
    import mooncake

    binary = os.path.join(os.path.dirname(mooncake.__file__), "mooncake_master")
    host = _host_address()
    # The default ports (50051, metrics 9003) collide when two jobs share a host.
    rpc_port, metrics_port = _free_ports(2)
    args = [binary, f"--rpc_port={rpc_port}", f"--metrics_port={metrics_port}"]
    if host.startswith("["):
        # Listen on IPv6 and IPv4 (the default, 0.0.0.0, is IPv4 only). Only for an
        # IPv6 host: binding an IPv6 address fails on hosts without IPv6.
        args.append("--rpc_address=::")
    # Stop the master when this process exits, also without cleanup (e.g. a
    # Monarch supervision abort). The signal fires when the calling thread exits,
    # so call this from the main thread. preexec_fn is not fork-safe with other
    # threads running; the child only calls prctl, resolved before the fork, then
    # execs. setpriv --pdeathsig would avoid it but is not on every host.
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    process = subprocess.Popen(
        args, preexec_fn=lambda: prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    )
    deadline = time.monotonic() + _MASTER_START_TIMEOUT_S
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"mooncake_master exited with {process.returncode}")
        try:
            socket.create_connection(("localhost", rpc_port), timeout=1).close()
            break
        except OSError:
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError(
                    f"mooncake_master does not listen on port {rpc_port} "
                    f"after {_MASTER_START_TIMEOUT_S} s"
                ) from None
            time.sleep(0.5)
    return f"{host}:{rpc_port}"

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""File-relay bridge: Claude Code in a Daytona sandbox <-> the inbound-firewalled
adapter on this trainer box.

Daytona refuses ``ssh -R``, so we relay HTTP over the Daytona ``fs`` control
plane: the agent hits an in-sandbox proxy that writes a req file; the host poller
downloads it and replays to the adapter, then uploads a resp file the proxy reads
back. One-shot delivery is equivalent to a dial-back since token capture happens
host-side in the adapter regardless.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import aiohttp

logger = logging.getLogger(__name__)

SPOOL = "/tmp/cagent_bridge"
REQ_DIR = f"{SPOOL}/req"
RESP_DIR = f"{SPOOL}/resp"
DEFAULT_PORT = 18001

# Hop-by-hop / length headers we must not blindly forward to the adapter.
_DROP_REQ_HEADERS = {
    "host",
    "content-length",
    "accept-encoding",
    "connection",
    "transfer-encoding",
}

# In-sandbox proxy. Pure stdlib, kept 3.6+ compatible for old sandbox pythons.
# Threaded so concurrent agent connections and per-response long-polls don't
# head-of-line block each other.
_PROXY_SRC = r"""
import sys, os, json, time, base64, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

PORT = int(sys.argv[1])
SPOOL = sys.argv[2]
REQ = os.path.join(SPOOL, "req")
RESP = os.path.join(SPOOL, "resp")
WAIT = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0

class TS(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _relay(self, method):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length else b""
        rid = uuid.uuid4().hex
        req = {
            "method": method,
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body_b64": base64.b64encode(body).decode("ascii"),
        }
        jp = os.path.join(REQ, rid + ".json")
        with open(jp + ".tmp", "w") as f:
            json.dump(req, f)
        os.rename(jp + ".tmp", jp)
        open(os.path.join(REQ, rid + ".ready"), "w").close()

        donep = os.path.join(RESP, rid + ".done")
        respp = os.path.join(RESP, rid + ".json")
        deadline = time.time() + WAIT
        while time.time() < deadline:
            if os.path.exists(donep):
                with open(respp) as f:
                    resp = json.load(f)
                rbody = base64.b64decode(resp.get("body_b64", ""))
                self.send_response(int(resp.get("status", 200)))
                for k, v in (resp.get("headers") or {}).items():
                    if k.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(rbody)))
                self.end_headers()
                self.wfile.write(rbody)
                for p in (donep, respp):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                return
            time.sleep(0.04)
        self.send_response(504)
        self.send_header("Content-Length", "13")
        self.end_headers()
        self.wfile.write(b"bridge timeout")

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")

    def log_message(self, *a):
        pass

for d in (REQ, RESP):
    try:
        os.makedirs(d)
    except OSError:
        pass
TS(("127.0.0.1", PORT), H).serve_forever()
"""


class DaytonaBridge:
    def __init__(
        self,
        sb,
        host_adapter_url: str,
        *,
        port: int = DEFAULT_PORT,
        poll_interval: float | None = None,
    ):
        import os

        self._sb = sb  # DaytonaSandbox wrapper (has .daytona -> AsyncSandbox)
        self._fs = sb.daytona.fs
        self.host_adapter_url = host_adapter_url.rstrip("/")
        self.port = port
        # The host busy-polls each sandbox's req dir over the Daytona fs API; at a
        # large rollout fanout (e.g. 8 prompts x 8 samples = 64 live bridges) a tight
        # interval blows past the account's fs request-rate cap (Tier 3 ~833 req/s):
        # 64 / 0.05s = 1280 list_files/s. 0.2s -> 320/s leaves headroom for the
        # per-turn download/upload calls. Adds at most poll_interval to per-turn
        # latency, negligible against multi-second generation. Tune via env.
        if poll_interval is None:
            poll_interval = float(os.environ.get("SWE_BRIDGE_POLL_INTERVAL", "0.2"))
        self.poll_interval = poll_interval
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self._inflight: set[str] = set()
        self._processed: set[str] = set()

    @property
    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self, *, ready_timeout: float = 60.0) -> "DaytonaBridge":
        await self._sb.exec(f"mkdir -p {REQ_DIR} {RESP_DIR}", user="root", check=True)
        await self._sb.write_file(
            "/tmp/daytona_bridge_proxy.py", _PROXY_SRC, user="root"
        )
        # Detached: short RPC returns immediately, proxy keeps serving.
        await self._sb.exec(
            "setsid bash -c 'nohup python3 /tmp/daytona_bridge_proxy.py "
            f"{self.port} {SPOOL} 900 > /tmp/daytona_bridge_proxy.log 2>&1 &' "
            "< /dev/null > /dev/null 2>&1",
            user="root",
            check=True,
            timeout=30,
        )
        # trust_env=False so the loopback adapter call ignores the box's fwdproxy.
        self._session = aiohttp.ClientSession(trust_env=False)
        self._task = asyncio.create_task(
            self._poll_loop(), name=f"daytona-bridge-{self._sb.sandbox_id[:8]}"
        )
        await self._wait_ready(ready_timeout)
        return self

    async def _wait_ready(self, timeout: float) -> None:
        # Round-trip a real /healthz through proxy -> relay -> adapter so the
        # agent never starts before the full path is live.
        deadline = asyncio.get_event_loop().time() + timeout
        # urllib, not curl: python3 is guaranteed in-sandbox; curl is not.
        probe = (
            'python3 -c "import urllib.request as u,sys;'
            f"sys.stdout.write(str(u.urlopen('{self.local_url}/healthz',timeout=8).status))\" "
            "2>/dev/null"
        )
        last = ""
        while asyncio.get_event_loop().time() < deadline:
            rc, out, _ = await self._sb.exec(probe, user="root", timeout=20)
            last = (out or "").strip()
            if last == "200":
                logger.info("[daytona_bridge] ready: %s/healthz -> 200", self.local_url)
                return
            await asyncio.sleep(1.0)
        raise RuntimeError(
            f"daytona bridge not ready after {timeout}s (last healthz={last!r})"
        )

    async def _poll_loop(self) -> None:
        try:
            while True:
                try:
                    files = await self._fs.list_files(REQ_DIR)
                    names = {getattr(f, "name", "") for f in files}
                    for ready in [n for n in names if n.endswith(".ready")]:
                        rid = ready[: -len(".ready")]
                        if rid in self._inflight or rid in self._processed:
                            continue
                        if f"{rid}.json" not in names:
                            continue
                        self._inflight.add(rid)
                        asyncio.create_task(self._handle(rid))
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("[daytona_bridge] poll error: %s", e)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _handle(self, rid: str) -> None:
        try:
            raw = await self._fs.download_file(f"{REQ_DIR}/{rid}.json")
            req = json.loads(raw)
            status, headers, body = await self._forward(req)
            payload = json.dumps(
                {
                    "status": status,
                    "headers": headers,
                    "body_b64": base64.b64encode(body).decode("ascii"),
                }
            ).encode("utf-8")
            await self._fs.upload_file(payload, f"{RESP_DIR}/{rid}.json")
            await self._fs.upload_file(b"", f"{RESP_DIR}/{rid}.done")
        except Exception as e:
            logger.warning("[daytona_bridge] handle %s failed: %s", rid[:8], e)
            try:
                err = json.dumps(
                    {
                        "status": 502,
                        "headers": {"Content-Type": "text/plain"},
                        "body_b64": base64.b64encode(
                            f"bridge error: {e}".encode()
                        ).decode("ascii"),
                    }
                ).encode("utf-8")
                await self._fs.upload_file(err, f"{RESP_DIR}/{rid}.json")
                await self._fs.upload_file(b"", f"{RESP_DIR}/{rid}.done")
            except Exception:
                pass
        finally:
            self._processed.add(rid)
            self._inflight.discard(rid)
            for suffix in (".ready", ".json"):
                try:
                    await self._fs.delete_file(f"{REQ_DIR}/{rid}{suffix}")
                except Exception:
                    pass

    async def _forward(self, req: dict) -> tuple[int, dict, bytes]:
        method = req.get("method", "POST")
        path = req.get("path", "/")
        headers = {
            k: v
            for k, v in (req.get("headers") or {}).items()
            if k.lower() not in _DROP_REQ_HEADERS
        }
        body = base64.b64decode(req.get("body_b64", ""))
        url = self.host_adapter_url + path
        assert self._session is not None
        async with self._session.request(
            method,
            url,
            data=body if body else None,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=900),
        ) as resp:
            rbody = await resp.read()
            ctype = resp.headers.get("Content-Type", "application/json")
            return resp.status, {"Content-Type": ctype}, rbody

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        try:
            await self._sb.exec(
                "pkill -f daytona_bridge_proxy.py || true", user="root", timeout=15
            )
        except Exception:
            pass


async def start_bridge(
    sb, host_adapter_url: str, *, port: int = DEFAULT_PORT
) -> DaytonaBridge:
    """Start the file-relay bridge for ``sb`` and return a started handle whose
    ``local_url`` the in-sandbox agent should use as ``ANTHROPIC_BASE_URL``."""
    bridge = DaytonaBridge(sb, host_adapter_url, port=port)
    return await bridge.start()

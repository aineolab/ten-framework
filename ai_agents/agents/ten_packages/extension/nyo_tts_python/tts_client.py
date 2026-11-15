"""NRSP v1 TTS client: one websocket, many sequential requests, no reconnect on cancel.

client -> server  {"type":"text","request_id","text","end"} | {"type":"flush","request_id"}
                  | {"type":"close"}
server -> client  {"type":"audio_start","request_id","request_seq","sample_rate"}
                  binary  [u32 BE request_seq][u32 BE audio_offset_ms][PCM16]
                  {"type":"audio_end","request_id","reason","audio_ms"}
                  {"type":"alignment",...} {"type":"usage",...} {"type":"error",...}

Cancel contract: ``flush(request_id)`` marks the request's ``request_seq`` stale at
once and tells the server; every later frame carrying a stale seq is dropped here.
The socket stays up, so cancelling costs one tiny text frame instead of a 100-300 ms
reconnect.
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode, urlparse

import websockets
from websockets.protocol import State

from . import transport
from .const import (
    CONNECT_TIMEOUT_S,
    DEFAULT_BASE_URL,
    NRSP_SUBPROTOCOL,
    STALE_SET_MAX,
)

TTS_HDR = struct.Struct(">II")  # [u32 request_seq][u32 audio_offset_ms], big-endian


class NYOTTSConnectError(Exception):
    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _status_of(e: BaseException) -> int | None:
    v = getattr(e, "status_code", None)
    if isinstance(v, int):
        return v
    v = getattr(getattr(e, "response", None), "status_code", None)
    return v if isinstance(v, int) else None


class _BoundedSet:
    """Insertion-ordered set that forgets its oldest entries past ``maxlen``."""

    def __init__(self, maxlen: int):
        self._d: OrderedDict = OrderedDict()
        self._maxlen = maxlen

    def add(self, v) -> None:
        self._d.pop(v, None)
        self._d[v] = None
        while len(self._d) > self._maxlen:
            self._d.popitem(last=False)

    def discard(self, v) -> None:
        self._d.pop(v, None)

    def __contains__(self, v) -> bool:
        return v in self._d


class NYOTTSClient:
    """Callbacks are plain coroutine functions set by the extension."""

    def __init__(self, config, ten_env):
        self.config = config
        self.ten_env = ten_env
        self.websocket = None
        self._reader: asyncio.Task | None = None
        self._closing = False
        # request bookkeeping
        self._seq_by_rid: dict[str, int] = {}
        self._rid_by_seq: dict[int, str] = {}
        self._live: set[str] = set()
        self._cancelled_rids = _BoundedSet(STALE_SET_MAX)
        self._stale_seqs = _BoundedSet(STALE_SET_MAX)
        self.last_sample_rate: int | None = None
        # counters (exposed in vendor metrics and bench)
        self.stale_frames_dropped = 0
        self.unmapped_frames_dropped = 0
        self.frames_received = 0
        self.recv_ns_total = 0
        self.connect_delay_ms = 0
        # callbacks
        self.on_audio_start: Callable[[str, int, dict], Awaitable[None]] = _noop
        self.on_audio: Callable[[str, int, bytes], Awaitable[None]] = _noop
        self.on_audio_end: Callable[[str, str, int, dict], Awaitable[None]] = _noop
        self.on_alignment: Callable[[dict], Awaitable[None]] = _noop
        self.on_usage: Callable[[dict], Awaitable[None]] = _noop
        self.on_error: Callable[[dict], Awaitable[None]] = _noop
        self.on_closed: Callable[[bool, str], Awaitable[None]] = _noop

    # ------------------------------------------------------------- lifecycle
    def build_url(self) -> str:
        q = {"model": self.config.model}
        if self.config.voice:
            q["voice"] = self.config.voice
        if self.config.language:
            q["language"] = self.config.language
        base = str(self.config.base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("base_url is required")
        return f"{base}/v1/tts?{urlencode(q)}"

    async def start(self, timeout: float = CONNECT_TIMEOUT_S) -> None:
        url = self.build_url()
        self.ten_env.log_info(f"Connecting to NYO TTS: {url}")
        t0 = time.perf_counter()
        try:
            self.websocket = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {self.config.api_key}"},
                subprotocols=[NRSP_SUBPROTOCOL],
                open_timeout=timeout,
                max_size=4 << 20,
                ping_interval=20,
                ping_timeout=20,
                ssl=transport.ssl_context(url, self.config.tls_verify) if url.startswith("wss://") else None,
            )
        except Exception as e:
            if transport.is_tls_verify_error(e):
                host = urlparse(url).netloc
                raise NYOTTSConnectError(f"TLS certificate verification failed for {host}: {e}; fix the certificate or set params.tls_verify=false", None, code="tls_error") from e
            raise NYOTTSConnectError(str(e), _status_of(e)) from e
        self.connect_delay_ms = int((time.perf_counter() - t0) * 1000)
        self._closing = False
        self._reader = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        self._closing = True
        ws, self.websocket = self.websocket, None
        if ws is not None:
            try:
                if ws.state == State.OPEN:
                    await ws.send(json.dumps({"type": "close"}))
                    await ws.close()
            except Exception as e:
                self.ten_env.log_info(f"Error closing NYO TTS websocket: {e}")
        task, self._reader = self._reader, None
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def is_connected(self) -> bool:
        return self.websocket is not None and self.websocket.state == State.OPEN

    # ------------------------------------------------------------------ send
    async def send_text(self, request_id: str, text: str, end: bool) -> None:
        if request_id in self._cancelled_rids:
            return  # cancelled while this call was in flight; never resurrect it
        if not self.is_connected():
            raise ConnectionError("NYO TTS websocket is not connected")
        self._live.add(request_id)
        await self.websocket.send(
            json.dumps({"type": "text", "request_id": request_id, "text": text, "end": end})
        )

    async def flush(self, request_id: str) -> None:
        """Cancel one request in place: stale-mark its seq, tell the server, keep the socket."""
        self._cancel_local(request_id)
        if self.is_connected():
            try:
                await self.websocket.send(json.dumps({"type": "flush", "request_id": request_id}))
            except Exception as e:  # socket died; nothing left to cancel server-side
                self.ten_env.log_warn(f"NRSP flush send failed: {e}")

    def _cancel_local(self, request_id: str) -> None:
        self._cancelled_rids.add(request_id)
        self._live.discard(request_id)
        seq = self._seq_by_rid.pop(request_id, None)
        if seq is not None:
            self._rid_by_seq.pop(seq, None)
            self._stale_seqs.add(seq)

    def forget(self, request_id: str) -> None:
        """Request finished normally: release its bookkeeping."""
        self._live.discard(request_id)
        seq = self._seq_by_rid.pop(request_id, None)
        if seq is not None:
            self._rid_by_seq.pop(seq, None)
            self._stale_seqs.add(seq)  # anything arriving later with this seq is stray

    # --------------------------------------------------------------- receive
    async def _read_loop(self) -> None:
        reason = "closed"
        try:
            async for raw in self.websocket:
                if isinstance(raw, bytes):
                    await self._on_binary(raw)
                else:
                    await self._on_text(raw)
        except websockets.exceptions.ConnectionClosed as e:
            reason = f"closed: {e}"
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        except Exception as e:
            reason = f"reader error: {e}"
            self.ten_env.log_error(f"NYO TTS reader error: {e}")
        finally:
            await self.on_closed(self._closing, reason)

    async def _on_binary(self, frame: bytes) -> None:
        t0 = time.perf_counter_ns()
        if len(frame) < TTS_HDR.size:
            self.ten_env.log_warn(f"NRSP: short binary frame ({len(frame)} bytes)")
            return
        seq, offset_ms = TTS_HDR.unpack_from(frame)
        if seq in self._stale_seqs:
            self.stale_frames_dropped += 1
            return
        rid = self._rid_by_seq.get(seq)
        if rid is None:
            # audio_start without request_seq: bind the unique unbound live request.
            unbound = [r for r in self._live if r not in self._seq_by_rid and r not in self._cancelled_rids]
            if len(unbound) == 1:
                rid = unbound[0]
                self._seq_by_rid[rid] = seq
                self._rid_by_seq[seq] = rid
            else:
                self.unmapped_frames_dropped += 1
                return
        pcm = frame[TTS_HDR.size:]
        self.frames_received += 1
        self.recv_ns_total += time.perf_counter_ns() - t0
        await self.on_audio(rid, offset_ms, pcm)

    async def _on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.ten_env.log_warn(f"NRSP: non-JSON text frame: {raw[:200]!r}")
            return
        mtype = msg.get("type")
        rid = msg.get("request_id")
        if mtype == "audio_start":
            seq = msg.get("request_seq")
            if rid in self._cancelled_rids:
                if isinstance(seq, int):
                    self._stale_seqs.add(seq)
                return
            if isinstance(seq, int) and rid is not None:
                self._seq_by_rid[rid] = seq
                self._rid_by_seq[seq] = rid
            sr = msg.get("sample_rate")
            if isinstance(sr, int):
                self.last_sample_rate = sr
            await self.on_audio_start(rid, int(sr) if isinstance(sr, int) else 0, msg)
        elif mtype == "audio_end":
            if rid in self._cancelled_rids:
                self._cancelled_rids.discard(rid)  # the ack of our own flush
                return
            await self.on_audio_end(rid, str(msg.get("reason", "end")), int(msg.get("audio_ms", 0) or 0), msg)
        elif mtype == "alignment":
            if rid not in self._cancelled_rids:
                await self.on_alignment(msg)
        elif mtype == "usage":
            await self.on_usage(msg)
        elif mtype == "error":
            if rid in self._cancelled_rids:
                return
            await self.on_error(msg)
        else:
            self.ten_env.log_debug(f"NRSP: ignoring message type {mtype!r}")


async def _noop(*_a: Any, **_k: Any) -> None:
    return None

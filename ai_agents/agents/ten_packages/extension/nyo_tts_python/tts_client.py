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
    TEXT_FRAME_MAX_BYTES,
)

TTS_HDR = struct.Struct(">II")  # [u32 request_seq][u32 audio_offset_ms], big-endian


class NYOTTSConnectError(Exception):
    def __init__(self, message: str, status_code: int | None = None, code: str | None = None,
                 retry_after: float | None = None, body_code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body_code = body_code  # the NRSP error code of the upgrade rejection body, as sent
        self.retry_after = retry_after  # seconds, from Retry-After on a 429/503 upgrade rejection


def _status_of(e: BaseException) -> int | None:
    v = getattr(e, "status_code", None)
    if isinstance(v, int):
        return v
    v = getattr(getattr(e, "response", None), "status_code", None)
    return v if isinstance(v, int) else None


def _body_code_of(e: BaseException) -> tuple[str | None, str | None]:
    """(code, message) of an NRSP error body on a rejected upgrade ({"type":"error","code",...}), else (None, None)."""
    body = getattr(getattr(e, "response", None), "body", None)
    if not body:
        return None, None
    try:
        d = json.loads(bytes(body).decode("utf-8", "replace"))
    except (ValueError, TypeError):
        return None, None
    if not isinstance(d, dict) or not isinstance(d.get("code"), str):
        return None, None
    msg = d.get("message")
    return d["code"], msg if isinstance(msg, str) else None


# NRSP rejection codes -> the extension's codes (unknown_voice: conclusive; model/language: maybe an engine down)
_BODY_CODES = {
    "unknown_voice": "unknown_voice",
    "unsupported_voice": "unknown_voice",
    "unknown_model": "model_not_found",
    "unsupported_language": "language_not_supported",
    "unsupported_sample_rate": "unsupported_sample_rate",
}


def split_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into pieces of at most ``max_chars`` characters, at whitespace where one is near the cut;
    the pieces concatenate back to ``text`` exactly."""
    if len(text) <= max_chars:
        return [text]
    out, i = [], 0
    while len(text) - i > max_chars:
        cut = i + max_chars
        ws = max(text.rfind(" ", i + max_chars // 2, cut), text.rfind("\n", i + max_chars // 2, cut))
        if ws > i:
            cut = ws + 1
        out.append(text[i:cut])
        i = cut
    out.append(text[i:])
    return out


def _retry_after_of(e: BaseException) -> float | None:
    """Retry-After (delta-seconds) of a rejected upgrade, or None."""
    headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        v = headers.get("Retry-After") if headers is not None else None
        return max(0.0, float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None


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
        self._unusable = False  # the server announced it closes this socket (idle_timeout, engine swap, ...)
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
            status = _status_of(e)
            body_code, body_msg = _body_code_of(e)
            msg = f"{e} ({body_code}: {body_msg})" if body_code else str(e)
            raise NYOTTSConnectError(msg, status, code=_BODY_CODES.get(body_code or ""), body_code=body_code,
                                     retry_after=_retry_after_of(e) if status in (429, 503) else None) from e
        self.connect_delay_ms = int((time.perf_counter() - t0) * 1000)
        self._closing = False
        self._reader = asyncio.create_task(self._read_loop())

    async def stop(self, cancel_reader: bool = True) -> None:
        """Close the socket. ``cancel_reader=False`` lets a reader that is still delivering a callback (e.g. the
        one reporting the error that caused this stop) finish it; the reader then ends on the closed socket."""
        self._closing = True
        ws, self.websocket = self.websocket, None
        if ws is not None:
            try:
                if ws.state == State.OPEN:
                    await ws.send(json.dumps({"type": "close"}))
                    await ws.close()
            except Exception as e:
                self.ten_env.log_info(f"Error closing NYO TTS websocket: {e}")
        if not cancel_reader:
            return  # the reader ends by itself on the closed socket; a later stop() still cancels it
        task, self._reader = self._reader, None
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def wait_closed(self, timeout: float) -> None:
        """Wait (bounded) until the reader has delivered everything the server sent before its close."""
        task = self._reader
        if task is not None and not task.done() and task is not asyncio.current_task():
            await asyncio.wait({task}, timeout=timeout)

    def is_connected(self) -> bool:
        """Open AND usable for new requests (false once the server announced its close)."""
        return not self._unusable and self.websocket is not None and self.websocket.state == State.OPEN

    def mark_unusable(self) -> None:
        """The server closes this socket next: take no new request; frames still in flight are delivered."""
        self._unusable = True

    # ------------------------------------------------------------------ send
    async def send_text(self, request_id: str, text: str, end: bool) -> None:
        if request_id in self._cancelled_rids:
            return  # cancelled while this call was in flight; never resurrect it
        if not self.is_connected():
            raise ConnectionError("NYO TTS websocket is not connected")
        self._live.add(request_id)
        msg = json.dumps({"type": "text", "request_id": request_id, "text": text, "end": end})
        if len(msg) <= TEXT_FRAME_MAX_BYTES:  # json.dumps is ASCII-only: len() is the encoded size
            await self.websocket.send(msg)
            return
        # never send a frame the gateway refuses (frame_too_large is fatal for the socket): split the increment
        # into several text messages of the same request; worst case 12 encoded bytes per character
        pieces = split_text(text, max(1, (TEXT_FRAME_MAX_BYTES - 256 - len(request_id) * 12) // 12))
        for i, piece in enumerate(pieces):
            if request_id in self._cancelled_rids:
                return  # flushed between two pieces: the rest is never sent
            last = i == len(pieces) - 1
            await self.websocket.send(
                json.dumps({"type": "text", "request_id": request_id, "text": piece, "end": end and last}))

    async def flush(self, request_id: str) -> None:
        """Cancel one request in place: stale-mark its seq, tell the server, keep the socket."""
        self._cancel_local(request_id)
        if self.websocket is not None and self.websocket.state == State.OPEN:  # also on a socket marked unusable
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

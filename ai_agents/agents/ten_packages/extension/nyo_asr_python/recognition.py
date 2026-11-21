"""NRSP v1 ASR client: one websocket per session, no TEN runtime imports.

Client -> server: binary PCM16 mono frames; text ``config`` / ``finalize`` / ``close``.
Server -> client: ``result``, ``finalize_end``, ``error``, ``usage`` (optional ``ready``).
"""

import asyncio
import contextlib
import inspect
import json
import time
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlencode, urlparse

import websockets
from websockets.protocol import State

from . import transport
from .const import CONNECT_TIMEOUT_S, NRSP_SUBPROTOCOL

LOG_VENDOR = "vendor"
LOG_TRANSCRIPTS = "transcripts"


class NRSPConnectError(Exception):
    """Handshake failed. ``status_code`` is the HTTP status when known."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        code: str | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after  # seconds, from a 429/503 handshake's Retry-After


class _FinalizeEvent(asyncio.Event):
    """Wakes a finalize waiter; ``ended`` tells a received finalize_end from a close."""

    def __init__(self) -> None:
        super().__init__()
        self.ended = False


class NYOASRRecognitionCallback(ABC):
    @abstractmethod
    async def on_open(self) -> None: ...

    @abstractmethod
    async def on_result(self, message: dict[str, Any]) -> None: ...

    @abstractmethod
    async def on_usage(self, message: dict[str, Any]) -> None: ...

    @abstractmethod
    async def on_error(
        self,
        message: str,
        code: str,
        fatal: bool,
        vendor_name: str,
        vendor_code: str,
    ) -> None: ...

    @abstractmethod
    async def on_close(self, expected: bool, reason: str) -> None: ...

    async def on_finalize_end(self, finalize_id: str) -> None:
        """A finalize_end arrived (before its waiter resumes). Optional."""


def extract_status_code(error: BaseException) -> int | None:
    """HTTP status from a websockets handshake error (InvalidStatus & co)."""
    for attr in ("status_code",):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def extract_retry_after(error: BaseException) -> float | None:
    """Retry-After (delta-seconds form) of a rejected handshake, if any."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    try:
        value = headers.get("Retry-After") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class NYOASRRecognition:
    def __init__(
        self,
        api_key: str,
        audio_timeline,
        ten_env,
        config: dict[str, Any],
        callback: NYOASRRecognitionCallback,
    ):
        self.api_key = api_key
        self.audio_timeline = audio_timeline
        self.ten_env = ten_env
        self.config = config or {}
        self.callback = callback
        self.websocket = None
        self.is_started = False
        self.ready_event = asyncio.Event()
        self.server_info: dict[str, Any] = {}
        self._last_model: str | None = None  # last model reported via on_usage
        self._finalize_events: dict[str, asyncio.Event] = {}
        self._message_task: asyncio.Task | None = None
        self._closing = False
        # Client-side overhead accounting (bench/results/ten.json)
        self.frames_sent = 0
        self.send_ns_total = 0

    # ------------------------------------------------------------------ url
    def build_url(self) -> str:
        base = str(self.config.get("base_url") or "").strip().rstrip("/")
        if not base:
            raise ValueError("base_url is required")
        query: dict[str, Any] = {}
        for key in ("model", "language", "sample_rate"):
            value = self.config.get(key)
            if value not in (None, ""):
                query[key] = value
        return f"{base}/v1/asr?{urlencode(query)}"

    def config_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "type": "config",
            "sample_rate": int(self.config.get("sample_rate", 16000)),
            "interim": bool(self.config.get("interim_results", True)),
        }
        if self.config.get("language"):  # unknown language: omitted, the gateway uses the model's own
            msg["language"] = self.config["language"]
        if self.config.get("endpointing_ms"):
            msg["endpointing_ms"] = int(self.config["endpointing_ms"])
        if self.config.get("hotwords"):
            msg["hotwords"] = self.config["hotwords"]
        return msg

    # -------------------------------------------------------------- lifecycle
    async def start(self, timeout: float = CONNECT_TIMEOUT_S) -> None:
        url = self.build_url()
        self.ten_env.log_info(f"Connecting to NYO ASR: {url}")
        # websockets rejects ssl= on ws:// URLs; only pass it for wss://.
        kwargs: dict[str, Any] = {}
        if url.startswith("wss://"):
            kwargs["ssl"] = transport.ssl_context(
                url, bool(self.config.get("tls_verify", True))
            )
        try:
            ws = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                subprotocols=[NRSP_SUBPROTOCOL],
                open_timeout=timeout,
                max_size=1 << 20,
                ping_interval=20,
                ping_timeout=20,
                **kwargs,
            )
        except Exception as e:
            if transport.is_tls_verify_error(e):
                raise NRSPConnectError(
                    f"TLS certificate verification failed for "
                    f"{urlparse(url).hostname}: {e}; fix the certificate or "
                    "set params.tls_verify=false",
                    None,
                    code="tls_error",
                ) from e
            raise NRSPConnectError(str(e), extract_status_code(e), retry_after=extract_retry_after(e)) from e
        if self._closing:
            # close() ran while the handshake was in flight: honour it, never undo it
            with contextlib.suppress(Exception):
                await ws.close()
            return
        self.websocket = ws
        try:
            self.is_started = True
            self.ready_event.clear()
            self._message_task = asyncio.create_task(self._message_handler())
            try:
                await self.websocket.send(json.dumps(self.config_message()))
            except websockets.exceptions.ConnectionClosed as e:
                # The server may reject the session (unknown model, bad version)
                # right after the upgrade; its `error` frame is already queued for
                # the reader, which will report it and then on_close().
                self.ten_env.log_warn(f"NRSP config not sent, server closed first: {e}")
                return
            if self._closing:  # closed while the config went out
                return
            self.ready_event.set()
            await self.callback.on_open()
        except BaseException:  # also CancelledError (stop during connect): never leak the socket/reader
            self.is_started = False
            await self._abort_socket()
            raise

    async def _abort_socket(self) -> None:
        self._closing = True  # our own abort: the reader's on_close is expected, not a reconnect trigger
        ws, self.websocket = self.websocket, None
        task = self._message_task
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def close(self) -> None:
        self._closing = True
        ws = self.websocket
        if ws is not None:
            try:
                if ws.state == State.OPEN:
                    await ws.send(json.dumps({"type": "close"}))
                    await ws.close()
            except Exception as e:
                self.ten_env.log_info(f"Error closing NYO ASR websocket: {e}")
        task = self._message_task
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for ev in self._finalize_events.values():
            ev.set()  # unblock any waiter; caller still emits finalize_end
        self.is_started = False

    def is_connected(self) -> bool:
        return (
            self.is_started
            and self.websocket is not None
            and self.websocket.state == State.OPEN
        )

    # ------------------------------------------------------------------ send
    @property
    def bytes_per_ms(self) -> float:
        return int(self.config.get("sample_rate", 16000)) / 1000 * 2

    async def send_audio_frame(self, audio_data: bytes) -> None:
        if not self.websocket or not self.is_connected():
            raise RuntimeError("WebSocket not connected")
        t0 = time.perf_counter_ns()
        duration_ms = int(len(audio_data) / self.bytes_per_ms)
        self.audio_timeline.add_user_audio(duration_ms)
        await self.websocket.send(audio_data)
        self.frames_sent += 1
        self.send_ns_total += time.perf_counter_ns() - t0

    async def send_finalize(self, finalize_id: str) -> None:
        if not self.websocket or not self.is_connected():
            raise RuntimeError("WebSocket not connected")
        ev = self._finalize_events.get(finalize_id)
        if ev is None or ev.is_set():
            # a duplicate id still in flight shares its event: the first finalize_end ends both waiters
            self._finalize_events[finalize_id] = _FinalizeEvent()
        await self.websocket.send(
            json.dumps({"type": "finalize", "id": finalize_id})
        )

    async def wait_finalize_end(self, finalize_id: str, timeout_ms: int) -> bool:
        ev = self._finalize_events.get(finalize_id)
        if ev is None:
            return False
        try:
            await asyncio.wait_for(ev.wait(), timeout_ms / 1000)
            # True when the finalize_end arrived (even if the socket closed right after it); False when only a
            # close / abort woke the waiter
            return bool(getattr(ev, "ended", False))
        except asyncio.TimeoutError:
            return False
        finally:
            self._finalize_events.pop(finalize_id, None)

    # --------------------------------------------------------------- receive
    async def _message_handler(self) -> None:
        reason = "closed"
        try:
            async for raw in self.websocket:
                if isinstance(raw, bytes):
                    continue  # ASR direction carries no binary downstream
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    self.ten_env.log_warn(f"NRSP: non-JSON text frame: {raw[:200]!r}")
                    continue
                await self._dispatch(msg)
        except websockets.exceptions.ConnectionClosed as e:
            reason = f"closed: {e}"
            self.ten_env.log_info(f"NYO ASR websocket {reason}")
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        except Exception as e:
            reason = f"handler error: {e}"
            await self.callback.on_error(
                f"WebSocket message handler error: {e}",
                "handler_error",
                False,
                "nyo",
                "handler_error",
            )
        finally:
            self.is_started = False
            for ev in self._finalize_events.values():
                ev.set()
            await self.callback.on_close(self._closing, reason)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        # Never log transcript text (no PII in logs): type, id, final flag and length only.
        self.ten_env.log_debug(
            f"vendor_result: {mtype} id={msg.get('id')} final={bool(msg.get('final', False))} "
            f"chars={len(str(msg.get('text') or ''))}",
            category=LOG_TRANSCRIPTS,
        )
        if mtype == "result":
            if "ttfw_ms" in msg:  # gateway-measured time to first word
                await self.callback.on_usage({"server_ttfw_ms": msg["ttfw_ms"]})
            model = (msg.get("metadata") or {}).get("model")
            if isinstance(model, str) and model != self._last_model:
                self._last_model = model  # report model swaps once, as metrics
                await self.callback.on_usage({"model": model})
            await self.callback.on_result(msg)
        elif mtype == "finalize_end":
            if "ttlw_ms" in msg:  # gateway-measured time to last word (finalize -> final)
                await self.callback.on_usage({"server_ttlw_ms": msg["ttlw_ms"]})
            # D21: ids the client never sent (e.g. "auto-3" from server-side
            # endpointing) match no event here and are ignored.
            fid = str(msg.get("id"))
            ev = self._finalize_events.get(fid)
            hook = getattr(self.callback, "on_finalize_end", None)
            if hook is not None:
                res = hook(fid)
                if inspect.isawaitable(res):
                    await res
            if ev is not None:
                ev.ended = True
                ev.set()
        elif mtype == "error":
            vendor = msg.get("vendor") or {}
            code = str(msg.get("code", "unknown"))
            await self.callback.on_error(
                str(vendor.get("message") or msg.get("message") or code),
                code,
                bool(msg.get("fatal", False)),
                str(vendor.get("name") or "nyo"),
                str(vendor.get("code") or code),
            )
        elif mtype == "usage":
            await self.callback.on_usage(msg)
        elif mtype == "ready":
            self.server_info = msg
        else:
            self.ten_env.log_debug(
                f"NRSP: ignoring unknown message type {mtype!r}",
                category=LOG_VENDOR,
            )

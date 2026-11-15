#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
from __future__ import annotations

import asyncio
import gc
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from ten_ai_base import utils
from ten_ai_base.const import (
    LOG_CATEGORY_KEY_POINT,
    LOG_CATEGORY_TRANSCRIPTS,
    LOG_CATEGORY_VENDOR,
)
from ten_ai_base.helper import PCMWriter
from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    ModuleErrorVendorInfo,
    ModuleMetrics,
    ModuleType,
    TTSAudioEndReason,
)
from ten_ai_base.struct import TTSTextInput, TTSTextResult, TTSWord
from ten_ai_base.tts2 import AsyncTTS2BaseExtension
from ten_runtime import AsyncTenEnv

from . import transport
from .config import NYOTTSConfig
from .const import MODELS_TIMEOUT_S, RECONNECT_MAX_ATTEMPTS, VALID_SAMPLE_RATES, VENDOR_NAME
from .tts_client import NYOTTSClient, NYOTTSConnectError

FATAL_HTTP = {401, 403}
# connect-time failures that retrying cannot fix (preflight and TLS)
_FATAL_CODES = {"model_not_found", "language_not_supported", "tls_error", "401", "403"}
# 400/404 also appear while a gateway restarts; retry, FATAL after this many in a row
SOFT_4XX = {400, 404}
SOFT_4XX_LIMIT = 3
# NRSP audio_end.reason -> TEN reason
_END_REASONS = {
    "end": TTSAudioEndReason.REQUEST_END,
    "flushed": TTSAudioEndReason.INTERRUPTED,
    "interrupted": TTSAudioEndReason.INTERRUPTED,
    "error": TTSAudioEndReason.ERROR,
}


def _dump_profile(path: str, name: str, samples_ns: list) -> None:
    """NYO_TEN_PROFILE=<file>: per-call cost inside the real TEN runtime (benchmarking aid)."""
    import json as _json
    xs = sorted(samples_ns)
    if not xs:
        return
    pick = lambda p: xs[min(len(xs) - 1, int(len(xs) * p))] / 1000  # noqa: E731
    with open(path, "a") as f:
        f.write(_json.dumps({"name": name, "n": len(xs), "p50_us": pick(0.5), "p95_us": pick(0.95),
                             "p99_us": pick(0.99), "max_us": xs[-1] / 1000}) + "\n")


@dataclass
class _Request:
    request_id: str
    start_ts: float = field(default_factory=time.perf_counter)  # first text sent
    first_audio_ts: float | None = None
    last_activity: float = field(default_factory=time.monotonic)
    audio_bytes: int = 0
    audio_start_sent: bool = False
    end_sent: bool = False
    sent_text: bool = False  # any non-blank text went to the server
    open: bool = True
    gaps: int = 0
    watchdog: asyncio.Task | None = None


class NYOTTSExtension(AsyncTTS2BaseExtension):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.config: NYOTTSConfig | None = None
        self.client: NYOTTSClient | None = None
        self._req: _Request | None = None
        self._sample_rate: int | None = None
        self._connect_lock = asyncio.Lock()
        self._reconnect_task: asyncio.Task | None = None
        self._bg_tasks: set[asyncio.Task] = set()
        self._stopped = False
        self._fatal = False
        self._preflight_ok = False
        self._fatal_sent = False  # exactly one FATAL tts_error per extension instance (review F1)
        self._prewarm_task: asyncio.Task | None = None
        self._recycle_pending = False
        self._soft_4xx = 0
        self.recorder_map: dict[str, PCMWriter] = {}
        self._prof_path = os.environ.get("NYO_TEN_PROFILE", "")
        self._prof: list[int] = []

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ------------------------------------------------------------ identity
    def vendor(self) -> str:
        return VENDOR_NAME

    def vendor_metadata(self) -> dict:
        c = self.config
        return {
            "url": c.base_url if c else "",
            "key": utils.encrypt(c.api_key) if c and c.api_key else "",  # masked at source (base class redaction is a second layer)
            "model": c.model if c else "",
            "region": "",
            "mode": "nrsp.v1",
        }

    def synthesize_audio_sample_rate(self) -> int:
        if self._sample_rate:
            return self._sample_rate
        return self.config.sample_rate if self.config else 24000

    # ----------------------------------------------------------- lifecycle
    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        try:
            await super().on_init(ten_env)
            config_json, _ = await self.ten_env.get_property_to_json("")
            if not config_json or config_json.strip() == "{}":
                raise ValueError("Configuration is empty. Required parameter 'api_key' is missing.")
            self.config = NYOTTSConfig.model_validate_json(config_json)
            self.config.update_params()
            ten_env.log_info(
                f"config: {self.config.to_str(sensitive_handling=True)}",
                category=LOG_CATEGORY_KEY_POINT,
            )
            self.config.validate_config()
        except Exception as e:
            ten_env.log_error(f"on_init failed: {traceback.format_exc()}")
            self._fatal = True
            await self._send_error(
                "",
                ModuleError(
                    message=f"Initialization failed: {e}",
                    module=ModuleType.TTS,
                    code=ModuleErrorCode.FATAL_ERROR,
                    vendor_info=ModuleErrorVendorInfo(
                        vendor=VENDOR_NAME, code=getattr(e, "code", None) or "invalid_config", message=str(e)
                    ),
                ),
            )
            return
        # A gen-2 GC pass over the (large) TEN+pydantic heap stalls the loop 4-8 ms,
        # i.e. up to 10% of an 85 ms frame. Freezing the init-time heap removes it
        # (measured: max per-frame 4067 us -> 125 us, see bench/results/ten.json).
        if self.config.gc_freeze:
            gc.collect()
            gc.freeze()
            ten_env.log_info("gc.freeze() applied (params.gc_freeze=true): process-wide, freezes the init-time heap of every extension in this host")
        # Prewarm so the first sentence does not pay the handshake. In the background: the
        # /v1/models check plus the upgrade can take a second or more over a WAN, and on_init
        # must not hold back the first requests (an empty request ends without the socket).
        self._prewarm_task = asyncio.create_task(self._prewarm())

    async def _prewarm(self) -> None:
        try:
            await self._ensure_connected()
        except Exception as e:
            self.ten_env.log_warn(f"initial NYO TTS connect failed: {e}")
            if not self._fatal:
                self._schedule_reconnect()

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        self._stopped = True
        self._cancel_reconnect_task()
        if self._prewarm_task and not self._prewarm_task.done():
            self._prewarm_task.cancel()
        if self._req and self._req.watchdog:
            self._req.watchdog.cancel()
        if self.client:
            await self.client.stop()
            self.client = None
        for rid, rec in list(self.recorder_map.items()):
            try:
                await rec.flush()
            except Exception as e:
                ten_env.log_error(f"Error flushing PCMWriter for request_id {rid}: {e}")
        if self._prof_path:
            _dump_profile(self._prof_path, "tts.send_tts_audio_data", self._prof)
        await super().on_stop(ten_env)

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        await super().on_deinit(ten_env)

    # ------------------------------------------------------------ connection
    async def _ensure_connected(self) -> None:
        async with self._connect_lock:
            if self.client and self.client.is_connected():
                return
            if self._fatal:
                raise NYOTTSConnectError("fatal error earlier; not reconnecting", 401)
            await self._connect()

    async def _preflight(self) -> None:
        """GET /v1/models before the first connect: fail fast on a bad model, key or certificate.

        Raises NYOTTSConnectError with a FATAL code (``model_not_found``, ``tls_error``, ``401``/``403``);
        _connect reports it like a failed handshake. A gateway that is down or restarting is not fatal:
        the check is skipped and retried on the next connect.
        """
        cfg = self.config
        assert cfg is not None
        try:
            ids = await transport.fetch_models(cfg.base_url, cfg.api_key, cfg.tls_verify, MODELS_TIMEOUT_S)
        except transport.TransportError as e:
            if e.code == "unauthorized":
                raise NYOTTSConnectError(str(e), e.status, code=str(e.status)) from e
            if e.code == "tls_error":
                raise NYOTTSConnectError(str(e), None, code="tls_error") from e
            self.ten_env.log_warn(f"NYO TTS /v1/models preflight skipped: {e}")
            return
        bad = transport.check_model(ids, cfg.model, "tts", None, cfg.base_url)
        if bad is not None:
            raise NYOTTSConnectError(bad[1], None, code=bad[0])
        self._preflight_ok = True

    async def _connect(self) -> None:
        assert self.config is not None
        if self.client:
            old, self.client = self.client, None
            await old.stop()
        await self.on_connecting()
        client = NYOTTSClient(self.config, self.ten_env)
        client.on_audio_start = self._on_audio_start
        client.on_audio = self._on_audio
        client.on_audio_end = self._on_audio_end
        client.on_usage = self._on_usage
        client.on_alignment = self._on_alignment
        client.on_error = self._on_error
        client.on_closed = self._on_closed
        try:
            if not self._preflight_ok:
                await self._preflight()
            await client.start()
        except NYOTTSConnectError as e:
            if e.code in _FATAL_CODES or self._is_fatal_status(e.status_code):
                self._fatal = True
            await self.on_disconnected(
                code=e.status_code or 1006,
                message=str(e),
                vendor_info=ModuleErrorVendorInfo(
                    vendor=VENDOR_NAME,
                    code=e.code or str(e.status_code or "connect_failed"),
                    message=str(e),
                ),
            )
            if self._fatal:
                await self._send_error(
                    "",
                    ModuleError(
                        message=str(e),
                        module=ModuleType.TTS,
                        code=ModuleErrorCode.FATAL_ERROR,
                        vendor_info=ModuleErrorVendorInfo(
                            vendor=VENDOR_NAME, code=e.code or str(e.status_code), message=str(e)
                        ),
                    ),
                )
            raise
        self.client = client
        self._soft_4xx = 0
        await self.on_connected()
        await self.metrics_connect_delay(client.connect_delay_ms)

    def _is_fatal_status(self, status: int | None) -> bool:
        if status in SOFT_4XX:
            self._soft_4xx += 1
            return self._soft_4xx >= SOFT_4XX_LIMIT
        self._soft_4xx = 0
        return status in FATAL_HTTP

    def _schedule_reconnect(self) -> None:
        if self._stopped or self._fatal:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    def _cancel_reconnect_task(self) -> None:
        t, self._reconnect_task = self._reconnect_task, None
        if t and not t.done() and t is not asyncio.current_task():
            t.cancel()

    def _error_once(self, err: ModuleError | None) -> ModuleError | None:
        """Gate every outgoing error: after the first FATAL the extension has stopped and stays silent."""
        if err is None or self._fatal_sent:
            return None
        if int(err.code) == int(ModuleErrorCode.FATAL_ERROR.value):
            self._fatal_sent = True
            self._fatal = True
            self._cancel_reconnect_task()
            client, self.client = self.client, None
            if client is not None:              # stop the socket; never from inside its reader
                self._spawn(client.stop())
        return err

    async def _send_error(self, request_id: str, err: ModuleError) -> None:
        err = self._error_once(err)
        if err is not None:
            await self.send_tts_error(request_id=request_id, error=err)

    def _reconnect_delay(self, attempt: int) -> float:
        return random.uniform(0.5, 1.0) * min(0.5 * 2 ** (attempt - 1), 4.0)  # jittered

    async def _reconnect_loop(self) -> None:
        last: Exception | None = None
        try:
            for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
                if self._stopped or self._fatal:
                    return
                await asyncio.sleep(self._reconnect_delay(attempt))
                try:
                    await self._ensure_connected()
                    return
                except Exception as e:
                    last = e
                    self.ten_env.log_warn(
                        f"NYO TTS reconnect #{attempt} failed: {e}",
                        category=LOG_CATEGORY_VENDOR,
                    )
            if not self._stopped and not self._fatal:
                await self._send_error(
                    "",
                    ModuleError(
                        message=f"NYO TTS: {RECONNECT_MAX_ATTEMPTS} reconnect attempts failed: {last}",
                        module=ModuleType.TTS,
                        code=ModuleErrorCode.FATAL_ERROR,
                        vendor_info=ModuleErrorVendorInfo(
                            vendor=VENDOR_NAME, code="connect_failed", message=str(last)
                        ),
                    ),
                )
        finally:
            if self._reconnect_task is asyncio.current_task():
                self._reconnect_task = None

    # ------------------------------------------------------------ requests
    async def request_tts(self, t: TTSTextInput) -> None:
        if self._stopped:
            self.ten_env.log_debug(f"TTS stopped, skipping request_id: {t.request_id}")
            return
        assert self.config is not None
        # Never log the text itself (no prompt/PII in logs): id, length and end flag only.
        self.ten_env.log_info(
            f"Requesting TTS: request ID: {t.request_id}, chars: {len(t.text)}, "
            f"text_input_end: {t.text_input_end}",
            category=LOG_CATEGORY_TRANSCRIPTS,
        )
        req = self._req
        if req is None or req.request_id != t.request_id:
            if req is not None and req.open:  # base serialises requests; defensive only
                await self._cancel_active(TTSAudioEndReason.INTERRUPTED, finish=True)
            if self._recycle_pending:
                await self._recycle_socket()
            req = await self._begin_request(t)
        elif not req.open or req.end_sent:
            self.ten_env.log_error(
                f"Received a message for a finished request_id '{t.request_id}' "
                f"after text_input_end; ignoring."
            )
            return
        if self._fatal:
            # stopped after a FATAL error (already reported once): end the request, contact nothing
            if t.text_input_end or not t.text:
                req.end_sent = True
                await self._finalize_request(TTSAudioEndReason.ERROR)
            return
        if t.text == "" and not t.text_input_end:
            return
        if t.text_input_end and not req.sent_text and not t.text.strip():
            # Nothing to synthesise: end the request here instead of a server round trip
            # (the tts_guarder wants tts_audio_end within 500 ms for empty requests).
            req.end_sent = True
            await self._finalize_request(TTSAudioEndReason.REQUEST_END)
            return
        try:
            await self._ensure_connected()
            assert self.client is not None
            self.metrics_add_input_characters(len(t.text))
            await self.client.send_text(t.request_id, t.text, t.text_input_end)
            req.sent_text = req.sent_text or bool(t.text.strip())
            req.last_activity = time.monotonic()
            if t.text_input_end:
                req.end_sent = True
                self._start_watchdog(req)
        except NYOTTSConnectError as e:
            await self._fail_request(req, e, status=e.status_code, fatal=bool(self._fatal))
            self._schedule_reconnect()
        except Exception as e:
            self.ten_env.log_error(
                f"Error in request_tts for request ID {t.request_id} ({len(t.text)} chars): "
                f"{traceback.format_exc()}",
                category=LOG_CATEGORY_TRANSCRIPTS,
            )
            await self._fail_request(req, e, status=None, fatal=False)
            self._schedule_reconnect()

    async def _begin_request(self, t: TTSTextInput) -> _Request:
        req = _Request(request_id=t.request_id)
        self._req = req
        if t.metadata:
            self.session_id = t.metadata.get("session_id", self.session_id)
        await self._setup_recorder(t.request_id)
        return req

    async def cancel_tts(self) -> None:
        """tts_flush: NRSP flush + stale-seq drop. The socket is NOT recycled."""
        req = self._req
        if req is None or not req.open:
            self.ten_env.log_debug("cancel_tts: no open request")
            return
        await self._cancel_active(TTSAudioEndReason.INTERRUPTED, finish=False)

    async def _cancel_active(self, reason: TTSAudioEndReason, finish: bool) -> None:
        req = self._req
        if req is None or not req.open:
            return
        if self.client:
            await self.client.flush(req.request_id)
        await self._finalize_request(reason, finish=finish)

    async def _fail_request(
        self, req: _Request, e: Exception, status: int | None, fatal: bool
    ) -> None:
        if fatal:
            self._fatal = True
        err = ModuleError(
            message=str(e),
            module=ModuleType.TTS,
            code=ModuleErrorCode.FATAL_ERROR if fatal else ModuleErrorCode.NON_FATAL_ERROR,
            vendor_info=ModuleErrorVendorInfo(
                vendor=VENDOR_NAME,
                code=str(status) if status is not None else "request_failed",
                message=str(e),
            ),
        )
        if self.client:
            self.client.forget(req.request_id)
        await self._finalize_request(TTSAudioEndReason.ERROR, error=err)

    async def _finalize_request(
        self,
        reason: TTSAudioEndReason,
        error: ModuleError | None = None,
        finish: bool = True,
    ) -> None:
        req = self._req
        if req is None or not req.open:
            return
        req.open = False  # first: callbacks and cancel race on this flag
        rid = req.request_id
        error = self._error_once(error)  # the request still ends (ERROR); the error only if not after a FATAL
        if req.watchdog and req.watchdog is not asyncio.current_task():
            req.watchdog.cancel()
        if not req.audio_start_sent:
            req.audio_start_sent = True
            await self.send_tts_audio_start(request_id=rid)
        # since the request began (first text sent), so it includes TTFB, as the other TTS2 vendors do
        interval_ms = int((time.perf_counter() - req.start_ts) * 1000)
        duration_ms = self._audio_ms(req.audio_bytes)
        if self.config and self.config.enable_words and reason != TTSAudioEndReason.ERROR:
            await self.send_tts_text_result(
                TTSTextResult(
                    request_id=rid,
                    text="",
                    start_ms=duration_ms,
                    duration_ms=0,
                    words=[],
                    text_result_end=True,
                    metadata=self.update_metadata(rid, None),
                )
            )
        await self.send_tts_audio_end(
            request_id=rid,
            request_event_interval_ms=interval_ms,
            request_total_audio_duration_ms=duration_ms,
            reason=reason,
        )
        if reason == TTSAudioEndReason.REQUEST_END:
            await self.send_usage_metrics(rid)
        rec = self.recorder_map.get(rid)
        if rec:
            await rec.flush()
        if self.client:
            self.client.forget(rid)
        if finish:
            await self.finish_request(request_id=rid, reason=reason, error=error)
        self.ten_env.log_debug(
            f"Finalized request {rid}, reason: {reason}, interval: {interval_ms}ms, "
            f"duration: {duration_ms}ms, gaps: {req.gaps}"
        )

    def _audio_ms(self, nbytes: int) -> int:
        return int(nbytes * 1000 / (self.synthesize_audio_sample_rate() * 2))

    # ----------------------------------------------------------- watchdog
    def _start_watchdog(self, req: _Request) -> None:
        if req.watchdog and not req.watchdog.done():
            return
        req.watchdog = asyncio.create_task(self._watch(req))

    async def _watch(self, req: _Request) -> None:
        assert self.config is not None
        timeout = self.config.request_timeout_ms / 1000
        while req.open:
            remaining = req.last_activity + timeout - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(remaining)
        if not req.open or self._req is not req:
            return
        self.ten_env.log_error(
            f"NYO TTS request {req.request_id} silent for {self.config.request_timeout_ms} ms",
            category=LOG_CATEGORY_VENDOR,
        )
        if self.client:
            await self.client.flush(req.request_id)
        await self._fail_request(
            req, TimeoutError(f"no audio_end within {self.config.request_timeout_ms} ms"),
            status=None, fatal=False,
        )

    # --------------------------------------------------- NRSP -> TEN callbacks
    def _is_current(self, rid: str | None) -> _Request | None:
        if self._stopped:  # never touch ten_env once on_stop has begun
            return None
        req = self._req
        return req if req is not None and req.open and req.request_id == rid else None

    async def _on_audio_start(self, rid: str, sample_rate: int, msg: dict) -> None:
        req = self._is_current(rid)
        if req is None:
            return
        req.last_activity = time.monotonic()
        if sample_rate in VALID_SAMPLE_RATES:
            if sample_rate != self._sample_rate:
                self.ten_env.log_info(
                    f"NYO TTS server sample_rate={sample_rate} (backend={msg.get('backend', 'unknown')})",
                    category=LOG_CATEGORY_KEY_POINT,
                )
            self._sample_rate = sample_rate
        elif sample_rate:
            self.ten_env.log_warn(f"NRSP audio_start has unsupported sample_rate {sample_rate}")

    async def _on_audio(self, rid: str, offset_ms: int, pcm: bytes) -> None:
        req = self._is_current(rid)
        if req is None or not pcm:
            return
        if len(pcm) % 2:  # PCM16: never let an odd byte leak into the next request
            pcm = pcm[:-1]
            if not pcm:
                return
        now = time.perf_counter()
        req.last_activity = time.monotonic()
        if req.first_audio_ts is None:
            req.first_audio_ts = now
            ttfb_ms = int((now - req.start_ts) * 1000)
            req.audio_start_sent = True
            await self.send_tts_audio_start(request_id=rid)
            await self.send_tts_ttfb_metrics(
                request_id=rid,
                ttfb_ms=ttfb_ms,
                extra_metadata={"model": self.config.model if self.config else ""},
            )
        expected = self._audio_ms(req.audio_bytes)
        if abs(offset_ms - expected) > 20:  # server-side gap or reorder
            req.gaps += 1
            self.ten_env.log_warn(
                f"NRSP audio gap on {rid}: offset {offset_ms} ms, expected {expected} ms",
                category=LOG_CATEGORY_VENDOR,
            )
        req.audio_bytes += len(pcm)
        self.metrics_add_recv_audio_chunks(pcm)
        await self._write_dump(rid, pcm)
        if self._prof_path:
            t0 = time.perf_counter_ns()
            await self.send_tts_audio_data(pcm, offset_ms)
            self._prof.append(time.perf_counter_ns() - t0)
        else:
            await self.send_tts_audio_data(pcm, offset_ms)

    async def _on_audio_end(self, rid: str, reason: str, audio_ms: int, msg: dict) -> None:
        req = self._is_current(rid)
        if req is None:
            return
        # gateway-measured latencies, exported next to the client-side ones; every
        # finished request also reports the model id it ran on.
        server = {f"server_{k}": msg[k] for k in ("ttfb_ms", "ttlw_ms") if k in msg}
        server["model"] = msg.get("model") or self.config.model
        await self.send_metrics(
            ModuleMetrics(
                id=self.get_uuid(), module=ModuleType.TTS, vendor=self.vendor(),
                metrics=server,
                metadata=self.update_metadata(rid, None),
            ),
            rid,
        )
        await self._finalize_request(_END_REASONS.get(reason, TTSAudioEndReason.REQUEST_END))

    async def _on_usage(self, msg: dict[str, Any]) -> None:
        """Server `usage` (per request after audio_end, or periodic per-session totals without request_id)."""
        metrics = {k: msg[k] for k in ("audio_ms", "chars", "credits") if k in msg}
        if not metrics:
            return
        rid = str(msg.get("request_id") or "")
        await self.send_metrics(
            ModuleMetrics(id=self.get_uuid(), module=ModuleType.TTS, vendor=self.vendor(),
                          metrics={f"server_{k}" if k != "credits" else k: v for k, v in metrics.items()},
                          metadata=self.update_metadata(rid or None, None)),
            rid,
        )

    async def _on_alignment(self, msg: dict) -> None:
        rid = msg.get("request_id")
        if not self.config or not self.config.enable_words or self._is_current(rid) is None:
            return
        words = [
            TTSWord(
                word=str(w.get("word", w.get("text", ""))),
                start_ms=int(w.get("start_ms", 0)),
                duration_ms=int(w.get("duration_ms", 0)),
            )
            for w in msg.get("words") or []
        ]
        await self.send_tts_text_result(
            TTSTextResult(
                request_id=rid,
                text=str(msg.get("text", "")),
                start_ms=int(msg.get("start_ms", words[0].start_ms if words else 0)),
                duration_ms=int(msg.get("duration_ms", 0)),
                words=words,
                text_result_end=False,
                metadata=self.update_metadata(rid, None),
            )
        )

    async def _on_error(self, msg: dict[str, Any]) -> None:
        rid = msg.get("request_id")
        vendor = msg.get("vendor") or {}
        code = str(msg.get("code", "unknown"))
        fatal = bool(msg.get("fatal", False))
        message = str(vendor.get("message") or msg.get("message") or code)
        if vendor.get("name") and vendor.get("name") != VENDOR_NAME:
            message = f"[{vendor.get('name')}] {message}"
        self.ten_env.log_error(
            f"vendor_error: vendor={vendor.get('name')} code={code} fatal={fatal}: {message}",
            category=LOG_CATEGORY_VENDOR,
        )
        err = ModuleError(
            message=message,
            module=ModuleType.TTS,
            code=ModuleErrorCode.FATAL_ERROR if fatal else ModuleErrorCode.NON_FATAL_ERROR,
            vendor_info=ModuleErrorVendorInfo(
                vendor=VENDOR_NAME,  # always ours; the engine that raised it is named in the message (F5)
                code=str(vendor.get("code") or code),
                message=message,
            ),
        )
        if code == "stale_request" and not fatal:
            return  # late text for a request we already finished/flushed; harmless race
        if fatal:
            self._fatal = True
        req = self._is_current(rid) if rid else (self._req if self._req and self._req.open else None)
        if req is not None:
            await self._finalize_request(TTSAudioEndReason.ERROR, error=err)
        else:
            await self._send_error(rid or "", err)

    async def _on_closed(self, expected: bool, reason: str) -> None:
        if expected or self._stopped:
            return
        self.ten_env.log_warn(f"NYO TTS websocket lost: {reason}", category=LOG_CATEGORY_VENDOR)
        vi = ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="connection_lost", message=reason)
        await self.on_disconnected(code=1006, message=reason, vendor_info=vi)
        req = self._req
        if req is not None and req.open:
            await self._finalize_request(
                TTSAudioEndReason.ERROR,
                error=ModuleError(
                    message=f"connection lost: {reason}",
                    module=ModuleType.TTS,
                    code=ModuleErrorCode.NON_FATAL_ERROR,
                    vendor_info=vi,
                ),
            )
        self._schedule_reconnect()

    # ------------------------------------------------------ update_configs
    async def update_configs(self, configs: dict) -> None:
        """cmd `update_configs`: switch voice / language / model between requests.

        The values ride in the websocket URL, so a change takes effect on a fresh
        socket. It is applied right away when idle and on the next request otherwise.
        """
        assert self.config is not None
        params = configs.get("params", configs) if isinstance(configs, dict) else {}
        changed = False
        for name in ("voice", "language", "model"):
            if name in params and params[name] != getattr(self.config, name):
                setattr(self.config, name, params[name])
                changed = True
        if not changed:
            return
        self.config.validate_config()
        self.ten_env.log_info(
            f"update_configs: voice={self.config.voice!r} language={self.config.language!r} "
            f"model={self.config.model!r}",
            category=LOG_CATEGORY_KEY_POINT,
        )
        self._sample_rate = None  # the new route announces its own rate
        if self._req is None or not self._req.open:
            await self._recycle_socket()
        else:
            self._recycle_pending = True

    async def _recycle_socket(self) -> None:
        self._recycle_pending = False
        if self.client:
            old, self.client = self.client, None
            await old.stop()
            await self.on_disconnected(code=0, message="config changed")
        try:
            await self._ensure_connected()
        except Exception as e:
            self.ten_env.log_warn(f"reconnect after update_configs failed: {e}")
            self._schedule_reconnect()

    # --------------------------------------------------------------- dump
    async def _setup_recorder(self, request_id: str) -> None:
        if not (self.config and self.config.dump):
            return
        for old in [r for r in self.recorder_map if r != request_id]:
            try:
                await self.recorder_map.pop(old).flush()
            except Exception as e:
                self.ten_env.log_error(f"Error cleaning up PCMWriter for {old}: {e}")
        if request_id not in self.recorder_map:
            path = os.path.join(self.config.dump_path, f"nyo_dump_{request_id}.pcm")
            self.recorder_map[request_id] = PCMWriter(path)

    async def _write_dump(self, rid: str, data: bytes) -> None:
        rec = self.recorder_map.get(rid)
        if rec is None:
            return
        try:
            await rec.write(data)
        except Exception as e:
            self.ten_env.log_error(f"Dump write failed: {e}")

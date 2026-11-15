import asyncio
import copy
import gc
import json
import os
import re
import time
from datetime import datetime
from typing import Any

from typing_extensions import override

from ten_ai_base.asr import (
    ASRBufferConfig,
    ASRBufferConfigModeKeep,
    ASRResult,
    AsyncASRBaseExtension,
)
from ten_ai_base.const import (
    LOG_CATEGORY_KEY_POINT,
    LOG_CATEGORY_TRANSCRIPTS,
    LOG_CATEGORY_VENDOR,
)
from ten_ai_base.dumper import Dumper
from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    ModuleErrorVendorInfo,
)
from ten_ai_base.struct import ASRWord
from ten_ai_base.utils import encrypt
from ten_runtime import AsyncTenEnv, AudioFrame

from .config import NYOASRConfig, normalize_language
from . import transport
from .const import (
    DEFAULT_BUFFER_MS,
    DUMP_FILE_NAME,
    MODELS_TIMEOUT_S,
    MODULE_NAME_ASR,
    RECONNECT_MAX_ATTEMPTS,
    VENDOR_NAME,
)
from .reconnect_manager import ReconnectManager
from .recognition import (
    NRSPConnectError,
    NYOASRRecognition,
    NYOASRRecognitionCallback,
)

FATAL_HTTP = {401, 403}
# 400 (bad version) / 404 (unknown model) are also what a gateway answers while it restarts and its
# engines are not registered yet, so they are retried and only become FATAL after this many in a row.
SOFT_4XX = {400, 404}
SOFT_4XX_LIMIT = 3

_REOPEN_ID = re.compile(r"^r(\d+)-")


def _reopen_gen(result_id) -> int:
    """Gateway engine re-open generation of a result id (`r<N>-utt-K` -> N; plain ids -> 0)."""
    m = _REOPEN_ID.match(str(result_id or ""))
    return int(m[1]) if m else 0


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


class NYOASRExtension(AsyncASRBaseExtension, NYOASRRecognitionCallback):
    def __init__(self, name: str):
        super().__init__(name)
        self.recognition: NYOASRRecognition | None = None
        self.config: NYOASRConfig | None = None
        self.audio_dumper: Dumper | None = None
        self.reconnect_manager: ReconnectManager | None = None
        self.sent_user_audio_duration_ms_before_last_reset = 0
        self.last_finalize_timestamp = 0
        self.connection_start_timestamp = 0
        self.last_backend: str | None = None
        self._last_frame_ms = 10
        self._dropped_accounted = 0
        self._last_interim: dict[str, Any] | None = None
        self._suppress_late_final = False
        self._stop_requested = False
        self._fatal = False
        self._fatal_sent = False  # exactly one FATAL asr_error per extension instance (review F1)
        self._promoted_id: str | None = None  # server utterance id promoted on a finalize timeout
        self._promoted_fid: str | None = None  # finalize id whose wait promoted it
        self._promoted_span: tuple[int, int] | None = None  # raw server start_ms .. start_ms + duration_ms
        self._promoted_gen = 0
        self._turn_t0: float | None = None  # perf_counter of the first audio frame of the current turn
        self._turn_ttfw_done = False
        self._fin_t0: float | None = None
        self._fin_id: str | None = None
        self._fin_ttlw_ms: float | None = None
        self._fin_server_ttlw = False
        self._preflight_ok = False
        self._soft_4xx = 0
        self._init_failed = False
        self._can_announce = True
        self._reconnect_task: asyncio.Task | None = None
        self._bg_tasks: set[asyncio.Task] = set()
        # Serialises "replay buffered frames after reconnect" with live frames
        # so audio reaches the server in order.
        self._replay_lock = asyncio.Lock()
        # False until on_open has replayed the base class' Keep buffer. While False,
        # is_connected() is False, so the base keeps queueing live frames behind the
        # replay instead of draining the same queue concurrently (that reordered audio
        # when the connect took long enough for frames to pile up).
        self._live = False
        self._prof_path = os.environ.get("NYO_TEN_PROFILE", "")
        self._prof: list[int] = []

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ------------------------------------------------------------ identity
    @override
    def vendor(self) -> str:
        return VENDOR_NAME

    @override
    def vendor_metadata(self) -> dict[str, Any]:
        p = self.config.params if self.config else {}
        return {
            "url": p.get("base_url", ""),
            "key": encrypt(str(p.get("api_key", "") or "")),  # masked at source (base class redaction is a second layer)
            "model": p.get("model", ""),
            "region": "",
            "mode": "nrsp.v1",
        }

    # ----------------------------------------------------------- lifecycle
    @override
    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        await super().on_init(ten_env)
        self.reconnect_manager = ReconnectManager(
            logger=ten_env, max_attempts=RECONNECT_MAX_ATTEMPTS
        )
        config_json, _ = await ten_env.get_property_to_json("")
        try:
            self.config = NYOASRConfig.model_validate_json(config_json)
            self.config.apply_defaults()
            self.config.validate_config()
            self._can_announce = callable(getattr(self, "_emit_connection_transition", None)) and callable(
                getattr(getattr(self, "_connection_machine", None), "try_connecting", None))
            if not self._can_announce:
                ten_env.log_warn("ten_ai_base has no _emit_connection_transition/_connection_machine.try_connecting: "
                                 "reconnect attempts will not be announced as CONNECTING")
            ten_env.log_info(
                f"config: {self.config.to_json(sensitive_handling=True)}",
                category=LOG_CATEGORY_KEY_POINT,
            )
            if self.config.params.get("hotwords"):
                ten_env.log_warn("params.hotwords is set but the NYO gateway / asr_fw do not apply hotwords yet; it is sent and ignored")
            if self.config.dump:
                self.audio_dumper = Dumper(
                    os.path.join(self.config.dump_path, DUMP_FILE_NAME)
                )
                await self.audio_dumper.start()
            if self.config.params.get("gc_freeze", True):
                gc.collect()
                gc.freeze()  # keep gen-2 GC passes over the init-time heap off the audio path
                ten_env.log_info("gc.freeze() applied (params.gc_freeze=true): process-wide, freezes the init-time heap of every extension in this host")
        except Exception as e:
            ten_env.log_error(f"Invalid NYO ASR config: {e}")
            self._init_failed = True
            self.config = NYOASRConfig.model_validate_json("{}")
            self.config.apply_defaults()
            await self._send_error(
                ModuleError(
                    module=MODULE_NAME_ASR,
                    code=ModuleErrorCode.FATAL_ERROR.value,
                    message=str(e),
                ),
                ModuleErrorVendorInfo(
                    vendor=VENDOR_NAME,
                    code=getattr(e, "code", "invalid_config"),
                    message=str(e),
                ),
            )

    @override
    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        self._cancel_reconnect_task()
        await super().on_deinit(ten_env)
        if self.audio_dumper:
            await self.audio_dumper.stop()
            self.audio_dumper = None

    @override
    async def start_connection(self) -> None:
        if self._init_failed or self._fatal:
            return
        assert self.config is not None
        self._stop_requested = False
        if not self._preflight_ok and not await self._preflight():
            return
        try:
            await self._connect_recognition()
        except Exception as e:
            await self._report_connect_failure(e)
            if not self._fatal:
                self._schedule_reconnect()

    @override
    async def stop_connection(self) -> None:
        self._stop_requested = True
        self._cancel_reconnect_task()
        rec, self.recognition = self.recognition, None
        if rec:
            await rec.close()
        if self._prof_path:
            _dump_profile(self._prof_path, "asr.send_audio", self._prof)

    @override
    def is_connected(self) -> bool:
        return (not self._fatal and self._live and self.recognition is not None
                and self.recognition.is_connected())

    async def _connect_recognition(self) -> None:
        assert self.config is not None
        if self.recognition is not None:
            rec, self.recognition = self.recognition, None
            await rec.close()
        self.connection_start_timestamp = int(datetime.now().timestamp() * 1000)
        self._live = False
        rec = NYOASRRecognition(
            api_key=str(self.config.params.get("api_key", "")),
            audio_timeline=self.audio_timeline,
            ten_env=self.ten_env,
            config=self.config.params,
            callback=self,
        )
        self.recognition = rec
        try:
            await rec.start()
        except BaseException:
            if self.recognition is rec:
                self.recognition = None
            raise

    # --------------------------------------------------------------- errors
    async def _send_error(self, error: ModuleError, vendor_info: ModuleErrorVendorInfo | None = None) -> None:
        """All asr_error output. After the first FATAL nothing more is sent: the extension has stopped."""
        if self._fatal_sent:
            self.ten_env.log_info(f"suppressed after FATAL: {error.code} {error.message}")
            return
        if int(error.code) == int(ModuleErrorCode.FATAL_ERROR.value):
            self._fatal_sent = True
            self._fatal = True
            self._cancel_reconnect_task()
            self._live = False                     # send_audio stops; the base class only buffers now
            rec, self.recognition = self.recognition, None
            if rec is not None:                    # never inside the reader task: close in the background
                self._spawn(rec.close())
        await self.send_asr_error(error, vendor_info)

    # ------------------------------------------------------------ preflight
    async def _preflight(self) -> bool:
        """Verify the configured model against GET /v1/models (runs once)."""
        assert self.config is not None
        params = self.config.params
        base_url = str(params.get("base_url") or "")
        try:
            ids = await transport.fetch_models(  # entries: id, kind, languages
                base_url,
                str(params.get("api_key") or ""),
                bool(params.get("tls_verify", True)),
                MODELS_TIMEOUT_S,
            )
        except transport.TransportError as e:
            if e.code in ("unauthorized", "tls_error"):
                # 401/403 keep the HTTP status as vendor code, as on the websocket path
                await self._preflight_fatal(str(e), str(e.status) if e.status else e.code)
                return False
            # Gateway down or restarting: the websocket path and reconnect loop
            # handle transients, so proceed. _preflight_ok stays False so the
            # next start/reconnect retries the check.
            self.ten_env.log_warn(f"models preflight inconclusive: {e}")
            return True
        bad = transport.check_model(ids, str(params.get("model") or ""), "asr",
                                    str(params.get("language") or "") or None, base_url)
        if bad is not None:
            await self._preflight_fatal(bad[1], bad[0])
            return False
        self._preflight_ok = True
        return True

    async def _preflight_fatal(self, message: str, code: str) -> None:
        self._fatal = True
        self.ten_env.log_error(f"models preflight failed: {message}")
        vendor_info = ModuleErrorVendorInfo(vendor=VENDOR_NAME, code=code, message=message)
        await self._send_error(
            ModuleError(
                module=MODULE_NAME_ASR,
                code=ModuleErrorCode.FATAL_ERROR.value,
                message=message,
            ),
            vendor_info,
        )
        # close the CONNECTING the base class announced, like a failed handshake does
        await self.on_disconnected(
            code=int(code) if code.isdigit() else 1006, message=message, vendor_info=vendor_info
        )

    # ------------------------------------------------------ connect errors
    def _is_fatal_status(self, status: int | None) -> bool:
        """Classify one failed connect. Counts consecutive 400/404 (call once per failure)."""
        if status in SOFT_4XX:
            self._soft_4xx += 1
            return self._soft_4xx >= SOFT_4XX_LIMIT
        self._soft_4xx = 0
        return status in FATAL_HTTP

    async def _report_connect_failure(self, e: Exception, fatal: bool | None = None) -> None:
        status = getattr(e, "status_code", None)
        code = getattr(e, "code", None)  # NRSPConnectError, e.g. "tls_error"
        if fatal is None:
            fatal = code == "tls_error" or self._is_fatal_status(status)
        self.ten_env.log_error(f"Failed to start NYO ASR connection: {e}")
        vendor_info = ModuleErrorVendorInfo(
            vendor=VENDOR_NAME,
            code=code or (str(status) if status is not None else "connect_failed"),
            message=str(e),
        )
        if fatal:
            self._fatal = True
        await self._send_error(
            ModuleError(
                module=MODULE_NAME_ASR,
                code=(
                    ModuleErrorCode.FATAL_ERROR.value
                    if fatal
                    else ModuleErrorCode.NON_FATAL_ERROR.value
                ),
                message=str(e),
            ),
            vendor_info,
        )
        await self.on_disconnected(
            code=status or 1006, message=str(e), vendor_info=vendor_info
        )

    # ------------------------------------------------------------ reconnect
    def _schedule_reconnect(self) -> None:
        if self._stop_requested or self._fatal:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    def _cancel_reconnect_task(self) -> None:
        task, self._reconnect_task = self._reconnect_task, None
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _announce_connecting(self) -> None:
        # The base class only emits CONNECTING around start_connection();
        # reconnect attempts run outside it, so announce them here.
        if not self._can_announce:
            return
        try:
            await self._emit_connection_transition(
                self._connection_machine.try_connecting(),
                already_done="reconnect: already connecting, skip.",
            )
        except Exception as e:  # base-class internals moved; never fatal
            self.ten_env.log_debug(f"announce connecting skipped: {e}")

    async def _reconnect_once(self) -> None:
        if not self._preflight_ok and not await self._preflight():
            return
        await self._announce_connecting()
        try:
            await self._connect_recognition()
        except Exception as e:
            status = getattr(e, "status_code", None)
            if getattr(e, "code", None) == "tls_error" or self._is_fatal_status(
                status
            ):
                await self._report_connect_failure(e, fatal=True)  # sets _fatal
                return
            await self.on_disconnected(
                code=status or 1006,
                message=str(e),
                vendor_info=ModuleErrorVendorInfo(
                    vendor=VENDOR_NAME, code="connect_failed", message=str(e)
                ),
            )
            raise

    async def _reconnect_loop(self) -> None:
        mgr = self.reconnect_manager
        if mgr is None:
            return
        try:
            while not self._stop_requested and not self._fatal and mgr.can_retry():
                ok = await mgr.handle_reconnect(
                    connection_func=self._reconnect_once,
                    error_handler=self._send_error,
                    vendor_name=VENDOR_NAME,
                    vendor_code="connect_failed",
                )
                if ok:
                    return
                self.ten_env.log_debug(
                    f"Reconnection attempt failed. Status: {mgr.get_attempts_info()}"
                )
        finally:
            if self._reconnect_task is asyncio.current_task():
                self._reconnect_task = None

    # ----------------------------------------------------------- audio path
    @override
    def buffer_strategy(self) -> ASRBufferConfig:
        buffer_ms = DEFAULT_BUFFER_MS
        if self.config is not None:
            buffer_ms = int(self.config.params.get("buffer_ms", DEFAULT_BUFFER_MS))
        return ASRBufferConfigModeKeep(
            byte_limit=int(buffer_ms * self.input_audio_sample_rate() / 1000 * 2)
        )

    @override
    def input_audio_sample_rate(self) -> int:
        if self.config is None:
            return 16000
        return int(self.config.params.get("sample_rate", 16000))

    @override
    async def send_audio(self, frame: AudioFrame, _session_id: str | None) -> bool:
        if not self.is_connected():
            return False
        if self._prof_path:
            t0 = time.perf_counter_ns()
            async with self._replay_lock:
                ok = await self._send_frame(frame)
            self._prof.append(time.perf_counter_ns() - t0)
            return ok
        async with self._replay_lock:
            return await self._send_frame(frame)

    async def _send_frame(self, frame: AudioFrame) -> bool:
        rec = self.recognition
        if rec is None or not rec.is_connected():
            return False
        buf = None
        try:
            buf = frame.lock_buf()
            audio_data = bytes(buf)
            frame.unlock_buf(buf)
            buf = None
            if self.audio_dumper:
                await self.audio_dumper.push_bytes(audio_data)
            await rec.send_audio_frame(audio_data)
            if self._turn_t0 is None:
                self._turn_t0 = time.perf_counter()
                self._turn_ttfw_done = False
            self._last_frame_ms = max(1, int(len(audio_data) / rec.bytes_per_ms))
            return True
        except Exception as e:
            self.ten_env.log_error(f"Error sending audio to NYO ASR: {e}")
            if buf is not None:
                try:
                    frame.unlock_buf(buf)
                except Exception:
                    pass
            return False

    async def _flush_buffered_audio_frames(self) -> None:
        while True:
            try:
                buffered = self.buffered_frames.get_nowait()
            except asyncio.QueueEmpty:
                self.buffered_frames_size = 0
                return
            metadata, _ = buffered.get_property_to_json("metadata")
            if metadata:
                try:
                    md = copy.deepcopy(json.loads(metadata))
                    self.metadata = md
                    self.session_id = md.get("session_id", self.session_id)
                except Exception:
                    pass
            await self._send_frame(buffered)

    # ----------------------------------------------------------- finalize
    @override
    async def finalize(self, _session_id: str | None) -> None:
        assert self.config is not None
        self.last_finalize_timestamp = int(datetime.now().timestamp() * 1000)
        rec = self.recognition
        if rec is None or not rec.is_connected():
            self.ten_env.log_warn(
                "asr_finalize: service not connected.",
                category=LOG_CATEGORY_KEY_POINT,
            )
            await self._finalize_end()
            return
        fid = str(self.finalize_id or self.get_finalize_fallback_id())
        try:
            self._fin_t0, self._fin_id, self._fin_ttlw_ms, self._fin_server_ttlw = time.perf_counter(), fid, None, False
            await rec.send_finalize(fid)
            ok = await rec.wait_finalize_end(fid, self.config.finalize_timeout_ms)
            if not ok:
                await self._on_finalize_timeout(fid, rec)
            elif self._fin_ttlw_ms is not None and not self._fin_server_ttlw:
                await self.send_vendor_metrics({"client_ttlw_ms": self._fin_ttlw_ms})
        except asyncio.CancelledError:
            self.ten_env.log_warn(
                "asr_finalize: wait for finalize_end was cancelled.",
                category=LOG_CATEGORY_KEY_POINT,
            )
        except Exception as e:
            self.ten_env.log_error(f"asr_finalize failed: {e}")
            await self._on_finalize_timeout(fid, rec)
        finally:
            self._fin_t0 = None
            await self._finalize_end()

    def get_finalize_fallback_id(self) -> str:
        return self._get_uuid()

    def _disarm_late_final(self) -> None:
        self._suppress_late_final = False
        self._promoted_id = None
        self._promoted_fid = None
        self._promoted_span = None
        self._promoted_gen = 0

    async def _on_finalize_timeout(self, fid: str, rec=None) -> None:
        """No finalize_end in time: do not lose the last hypothesis."""
        self.ten_env.log_warn(
            f"asr_finalize: no finalize_end for {fid} within "
            f"{self.config.finalize_timeout_ms} ms",  # type: ignore[union-attr]
            category=LOG_CATEGORY_KEY_POINT,
        )
        pending, self._last_interim = self._last_interim, None
        if pending and pending.get("text"):
            if rec is not None and rec is self.recognition and rec.is_connected():
                # the server's real final may still arrive; it must not become a 2nd final
                self._suppress_late_final = True
                self._promoted_id = pending.get("id")
                self._promoted_gen = _reopen_gen(pending.get("id"))
                self._promoted_fid = fid
                self._promoted_span = None
                if "start_ms" in pending:
                    start_ms = int(pending["start_ms"] or 0)
                    self._promoted_span = (start_ms, start_ms + int(pending.get("duration_ms") or 0))
            else:
                self.ten_env.log_info(
                    "socket closed during finalize: promoted last interim, nothing to suppress"
                )
            await self._emit_asr_result(pending, final=True, promoted=True)

    async def _finalize_end(self) -> None:
        if self.last_finalize_timestamp != 0:
            self.last_finalize_timestamp = 0
            await self.send_asr_finalize_end()

    # ------------------------------------------------------------- results
    async def _emit_asr_result(
        self, message: dict[str, Any], final: bool, promoted: bool = False
    ) -> None:
        assert self.config is not None
        text = message.get("text", "")
        if not text:
            return
        start_ms = int(message.get("start_ms", 0) or 0)
        duration_ms = int(message.get("duration_ms", 0) or 0)
        offset = self.sent_user_audio_duration_ms_before_last_reset
        actual_start_ms = int(
            self.audio_timeline.get_audio_duration_before_time(start_ms) + offset
        )
        base_metadata = copy.deepcopy(self.metadata) if self.metadata else {}
        session_id = base_metadata.pop("session_id", None)
        backend = message.get("backend") or self.last_backend or "unknown"
        self.last_backend = backend
        metadata: dict[str, Any] = {}
        if session_id is not None:
            metadata["session_id"] = session_id
        metadata["asr_info"] = {
            **base_metadata,
            "vendor": self.vendor(),
            "backend": backend,
            "model": (message.get("metadata") or {}).get("model")
            or self.config.params.get("model", ""),
            "locked": bool(final),
            **({"promoted_on_timeout": True} if promoted else {}),
        }
        words = []
        for w in message.get("words") or []:
            w_start = int(w.get("start_ms", 0) or 0)
            words.append(
                ASRWord(
                    word=w.get("word", w.get("text", "")),
                    start_ms=int(
                        self.audio_timeline.get_audio_duration_before_time(w_start)
                        + offset
                    ),
                    duration_ms=max(0, int(w.get("duration_ms", 0) or 0)),
                    stable=bool(w.get("stable", final)),
                )
            )
        # the server's language when it reports one (a swapped model or auto-detect may differ),
        # else the configured one (ten review 7)
        language = normalize_language(message.get("language")) or self.config.normalized_language
        await self.send_asr_result(
            ASRResult(
                id=self._get_uuid(),
                text=text,
                final=final,
                start_ms=actual_start_ms,
                duration_ms=duration_ms,
                language=language,
                words=words,
                metadata=metadata,
            )
        )

    # ------------------------------------------- recognition callbacks (NRSP)
    @override
    async def on_open(self) -> None:
        delay = int(datetime.now().timestamp() * 1000) - self.connection_start_timestamp
        self.ten_env.log_info(
            "vendor_status_changed: on_open", category=LOG_CATEGORY_VENDOR
        )
        await self.send_connect_delay_metrics(delay)
        if self.reconnect_manager:
            self.reconnect_manager.mark_connection_successful()
        # The server timeline restarts at 0 on every socket.
        self.sent_user_audio_duration_ms_before_last_reset += (
            self.audio_timeline.get_total_user_audio_duration()
        )
        self.audio_timeline.reset()
        # Frames the Keep buffer had to drop during a long outage never reach the
        # server; tell the timeline so later start_ms values do not drift.
        newly_dropped = self.dropped_audio_frames - self._dropped_accounted
        self._dropped_accounted = self.dropped_audio_frames
        if newly_dropped > 0:
            self.audio_timeline.add_dropped_audio(newly_dropped * self._last_frame_ms)
        self._last_interim = None
        self._disarm_late_final()  # new socket: new id space and timeline; the promoted turn is not on it
        self._turn_t0 = None
        self._fin_t0 = None
        self._soft_4xx = 0
        await self.on_connected()
        async with self._replay_lock:
            await self._flush_buffered_audio_frames()
            self._live = True  # no await since the queue was seen empty: nothing can slip in between

    @override
    async def on_result(self, message: dict[str, Any]) -> None:
        final = bool(message.get("final", False))
        if self._suppress_late_final:
            # Drop the real final of the turn we promoted on a finalize timeout. A result belongs to
            # that turn when it carries the promoted utterance id, the finalize_id we armed with, or
            # a start_ms inside the promoted span while coming from a newer re-opened engine session
            # (`r<N>-` id with N above the promoted generation: a replayed decode of audio already
            # reported final). Interims of that turn keep the flag; any result of another turn ends it.
            same = (
                (self._promoted_id is not None and message.get("id") == self._promoted_id)
                or (self._promoted_fid is not None
                    and (message.get("metadata") or {}).get("finalize_id") == self._promoted_fid)
                or (self._promoted_span is not None and "start_ms" in message
                    and int(message["start_ms"] or 0) < self._promoted_span[1]
                    and _reopen_gen(message.get("id")) > self._promoted_gen)
            )
            no_ids = (
                self._promoted_id is None
                and self._promoted_span is None
                and message.get("id") is None
                and not (message.get("metadata") or {}).get("finalize_id")
            )
            if final and (same or no_ids):
                self.ten_env.log_info("dropping late final already promoted on timeout")
                self._disarm_late_final()
                return
            if not same:
                self._disarm_late_final()
            elif not final:
                return  # an interim of the promoted turn would reopen a closed turn
        if final:
            self._last_interim = None
        elif message.get("text"):
            self._last_interim = message
        now = time.perf_counter()
        if message.get("text") and self._turn_t0 is not None and not self._turn_ttfw_done:
            self._turn_ttfw_done = True
            if "ttfw_ms" not in message:  # the gateway measured it for this result: its value wins
                await self.send_vendor_metrics({"client_ttfw_ms": round((now - self._turn_t0) * 1000, 1)})
        if final:
            fid = (message.get("metadata") or {}).get("finalize_id")
            if self._fin_t0 is not None and (fid is None or fid == self._fin_id):
                self._fin_ttlw_ms = round((now - self._fin_t0) * 1000, 1)
                self._fin_t0 = None
            self._turn_t0 = None
        await self._emit_asr_result(message, final=final)

    @override
    async def on_usage(self, message: dict[str, Any]) -> None:
        if "server_ttlw_ms" in message:
            self._fin_server_ttlw = True
        metrics = {
            k: message[k]
            for k in (
                "audio_ms",
                "credits",
                "server_ttfw_ms",
                "server_ttlw_ms",
                "model",
            )
            if k in message
        }
        if metrics:
            await self.send_vendor_metrics(metrics)

    @override
    async def on_error(
        self,
        message: str,
        code: str,
        fatal: bool,
        vendor_name: str,
        vendor_code: str,
    ) -> None:
        self.ten_env.log_error(
            f"vendor_error: vendor: {vendor_name}, code: {vendor_code}, "
            f"fatal: {fatal}, reason: {message}",
            category=LOG_CATEGORY_VENDOR,
        )
        # vendor_info.vendor is always this extension's vendor (as upstream vendors do); the
        # upstream engine that raised it (e.g. "halohalo") goes into the message (review F5)
        if vendor_name and vendor_name != VENDOR_NAME:
            message = f"[{vendor_name}] {message}"
        await self._send_error(
            ModuleError(
                module=MODULE_NAME_ASR,
                code=(
                    ModuleErrorCode.FATAL_ERROR.value
                    if fatal
                    else ModuleErrorCode.NON_FATAL_ERROR.value
                ),
                message=message,
            ),
            ModuleErrorVendorInfo(
                vendor=VENDOR_NAME,
                code=vendor_code or code,
                message=message,
            ),
        )

    @override
    async def on_close(self, expected: bool, reason: str) -> None:
        self.ten_env.log_info(f"vendor_status_changed: on_close ({reason})")
        self._live = False
        if not expected:
            self.recognition = None
        await self.on_disconnected(
            code=0 if expected else 1006, message=reason
        )
        if expected or self._stop_requested or self._fatal:
            return
        self._schedule_reconnect()

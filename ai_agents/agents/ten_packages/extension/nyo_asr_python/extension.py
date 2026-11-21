import asyncio
import copy
import gc
import json
import os
import re
import time
from collections import deque
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
    NEMOTRON_FINALIZE_TIMEOUT_MS,
    DEFAULT_BUFFER_MS,
    DEFAULT_RECONNECT_WINDOW_MS,
    DUMP_FILE_NAME,
    IDLE_TIMEOUT_CODE,
    UNKNOWN_MODEL_CODE,
    MODELS_TIMEOUT_S,
    MODULE_NAME_ASR,
    RECONNECTABLE_CODES,
    VENDOR_NAME,
)
from .reconnect_manager import ReconnectManager
from .recognition import (
    NRSPConnectError,
    NYOASRRecognition,
    NYOASRRecognitionCallback,
)

FATAL_HTTP = {401, 403}
# 400 / 404 are also what a gateway answers while it swaps, boots or has lost an engine (the model is not
# registered then). They are FATAL only when GET /v1/models conclusively shows the model (or its language)
# is not served; otherwise they are retried inside reconnect_window_ms like any other connect failure.
SOFT_4XX = {400, 404}
# A replayed result of the promoted turn may end this much (plus endpointing_ms, the trailing silence of an
# endpointed cut) after the promoted interim did; a result that runs further is a later turn merged into it.
PROMOTED_SPAN_SLACK_MS = 500

_REOPEN_ID = re.compile(r"^r(\d+)-")
_NO_INTERIM = object()  # finalize snapshot: no interim was pending when it was sent


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


class _CurrentSocketCallback(NYOASRRecognitionCallback):
    """A recognition's callbacks, forwarded only while it is the extension's current one.

    A superseded or detached socket (replaced by a reconnect, closed by stop / FATAL) must not touch the live
    socket's state: its late on_close would otherwise drop the live recognition, its idle_timeout would mark
    the live session idle. A socket that opens after it was superseded closes itself.
    """

    def __init__(self, ext: "NYOASRExtension", rec: NYOASRRecognition):
        self._ext = ext
        self._rec = rec

    def _current(self) -> bool:
        return self._ext.recognition is self._rec

    async def on_open(self) -> None:
        if self._current():
            await self._ext.on_open()
        else:
            self._ext.ten_env.log_info("superseded socket opened: closing it")
            await self._rec.close()

    async def on_result(self, message: dict[str, Any]) -> None:
        if self._current():
            await self._ext.on_result(message)

    async def on_usage(self, message: dict[str, Any]) -> None:
        if self._current():
            await self._ext.on_usage(message)

    async def on_error(self, message: str, code: str, fatal: bool, vendor_name: str, vendor_code: str) -> None:
        if self._current():
            await self._ext.on_error(message, code, fatal, vendor_name, vendor_code)
        else:
            self._ext.ten_env.log_info(f"error from a superseded socket ignored: {code}")

    async def on_close(self, expected: bool, reason: str) -> None:
        if self._current():
            await self._ext.on_close(expected, reason)
        else:
            self._ext.ten_env.log_debug(f"close of a superseded socket ignored ({reason})")

    async def on_finalize_end(self, finalize_id: str) -> None:
        if self._current():
            await self._ext.on_finalize_end(finalize_id)


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
        # Latest interim per server result id, least recently updated first (an utterance's interims share its
        # id). A final removes only its own utterance's interim (and older ones it covers), never a later turn's.
        self._interims: dict[Any, dict[str, Any]] = {}
        # finalize id -> the result id of the latest interim when that finalize was sent: the only interim its
        # finalize may promote (an interim that arrived later belongs to the next turn)
        self._fin_snap: dict[str, Any] = {}
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
        self._init_failed = False
        # GET /v1/models did not list our model (last check): an engine that is down, or a typo. Retried in
        # reconnect_window_ms; the FATAL at its end carries vendor code model_not_found with this message.
        self._model_missing: str | None = None
        self._can_announce = True
        self._reconnect_task: asyncio.Task | None = None
        self._bg_tasks: set[asyncio.Task] = set()
        # The server closed the socket for inactivity (idle_timeout): not an error; the next
        # audio frame reconnects (on_audio_frame).
        self._idle_closed = False
        # asr_finalize ids still owed an asr_finalize_end (overlapping finalizes each get their own).
        self._open_finalizes: list[str] = []
        # finalize ids this addon sent (recent ones): a result naming one of them belongs to that finalize
        self._client_fids: deque = deque(maxlen=64)
        # finals attributed to each pending finalize (oldest pending first; emitted or dropped both count),
        # and the finalizes whose finalize_end has arrived (they take no more finals)
        self._fin_finals: dict[str, int] = {}
        self._fin_ended: set[str] = set()
        # Serialises "replay buffered frames after reconnect" with live frames
        # so audio reaches the server in order.
        self._replay_lock = asyncio.Lock()
        # Single flight: one connect attempt at a time (start_connection / trigger_connect, the reconnect loop).
        self._connect_lock = asyncio.Lock()
        # True while on_open runs (connect metrics, CONNECTED, Keep-buffer replay): a finalize waits for it.
        self._replaying = False
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

    # ------------------------------------------------------------ interims
    @property
    def _last_interim(self) -> dict[str, Any] | None:
        """The most recently updated interim (None when there is none)."""
        return next(reversed(self._interims.values()), None) if self._interims else None

    @_last_interim.setter
    def _last_interim(self, message: dict[str, Any] | None) -> None:
        if message is None:
            self._interims.clear()
        else:
            self._note_interim(message)

    def _note_interim(self, message: dict[str, Any]) -> None:
        key = message.get("id")
        self._interims.pop(key, None)  # re-insert: the dict stays ordered by last update
        self._interims[key] = message

    def _note_final(self, message: dict[str, Any]) -> None:
        """A final ends its own utterance's interim and any older one it covers (start_ms before the final's end:
        e.g. the original session's interim of a turn a re-opened engine just re-decoded). A later turn's interim
        (it starts at or after the cut) stays."""
        key = message.get("id")
        if key is None:
            self._interims.clear()
            return
        self._interims.pop(key, None)
        if "start_ms" in message:
            end = int(message.get("start_ms") or 0) + int(message.get("duration_ms") or 0)
            for k in [k for k, m in self._interims.items()
                      if "start_ms" in m and int(m.get("start_ms") or 0) < end]:
                del self._interims[k]

    def _latest_interim_key(self):
        return next(reversed(self._interims), None) if self._interims else _NO_INTERIM

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
        self.reconnect_manager = ReconnectManager(logger=ten_env, window_ms=DEFAULT_RECONNECT_WINDOW_MS)
        config_json, _ = await ten_env.get_property_to_json("")
        try:
            self.config = NYOASRConfig.model_validate_json(config_json)
            self.config.apply_defaults()
            self.config.validate_config()
            self.reconnect_manager.window_ms = int(self.config.params["reconnect_window_ms"])
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
            if self.config.params.get("gc_freeze", False):
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
        self._stop_requested = True
        await self._await_reconnect_task_cancelled()
        rec, self.recognition = self.recognition, None
        if rec is not None:  # e.g. a connect that raced on_stop: no socket outlives the extension
            try:
                await rec.close()
            except Exception as e:
                ten_env.log_warn(f"closing the NYO ASR socket on deinit: {e}")
        await self._drain_bg_tasks()
        await super().on_deinit(ten_env)
        if self.audio_dumper:
            await self.audio_dumper.stop()
            self.audio_dumper = None

    async def _drain_bg_tasks(self, timeout: float = 5.0) -> None:
        """Await background closes (e.g. the socket close started on a FATAL) so no socket outlives us."""
        tasks = [t for t in self._bg_tasks if not t.done()]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @override
    async def on_audio_frame(self, ten_env: AsyncTenEnv, audio_frame: AudioFrame) -> None:
        if self._idle_closed and not self._stop_requested and not self._fatal and not self.stopped:
            # the gateway closed the idle socket: audio is back, reconnect (frames buffer meanwhile)
            self._idle_closed = False
            ten_env.log_info("audio after idle_timeout: reconnecting", category=LOG_CATEGORY_KEY_POINT)
            self._schedule_reconnect()
        await super().on_audio_frame(ten_env, audio_frame)

    @override
    async def start_connection(self) -> None:
        # The base class has announced CONNECTING: every early return settles it as DISCONNECTED.
        if self.stopped:
            self.ten_env.log_info("start_connection after on_stop: ignored", category=LOG_CATEGORY_KEY_POINT)
            await self.on_disconnected(code=0, message="stopped")
            return
        if self._init_failed or self._fatal:
            await self.on_disconnected(code=1006, message="not connecting: a FATAL error was reported")
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            # trigger_connect during an outage: the running reconnect loop is the connect attempt
            self.ten_env.log_info("start_connection: a reconnect is in progress, joining it",
                                  category=LOG_CATEGORY_KEY_POINT)
            return
        assert self.config is not None
        self._stop_requested = False
        self._idle_closed = False
        if not self._preflight_ok and not await self._preflight():
            return
        if self.stopped or self._stop_requested:  # on_stop arrived during the preflight
            self.ten_env.log_info("start_connection: stopped during the preflight, not connecting",
                                  category=LOG_CATEGORY_KEY_POINT)
            await self.on_disconnected(code=0, message="stopped")
            return
        try:
            await self._connect_recognition()
        except Exception as e:
            if self.reconnect_manager:
                self.reconnect_manager.start_outage()
                self.reconnect_manager.set_retry_after(getattr(e, "retry_after", None))
            await self._report_connect_failure(e)
            if not self._fatal:
                self._schedule_reconnect()

    @override
    async def stop_connection(self) -> None:
        self._stop_requested = True
        await self._await_reconnect_task_cancelled()
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
        async with self._connect_lock:  # single flight: a second caller waits, then reuses the connection
            if self.stopped or self._stop_requested:
                return
            if self.recognition is not None and self.recognition.is_connected():
                return
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
            rec.callback = _CurrentSocketCallback(self, rec)
            self.recognition = rec
            try:
                await rec.start()
            except BaseException:
                if self.recognition is rec:
                    self.recognition = None
                raise
            if self.stopped or self._stop_requested:  # on_stop during the handshake / on_open: never keep it
                if self.recognition is rec:
                    self.recognition = None
                    self._live = False
                await rec.close()

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
            if rec is not None:  # the detached socket's on_close is ignored: settle the status here
                await self.on_disconnected(code=1006, message=str(error.message), vendor_info=vendor_info)
            return
        await self.send_asr_error(error, vendor_info)

    # ------------------------------------------------------------ preflight
    async def _models_verdict(self) -> tuple[str, str, str]:
        """GET /v1/models against the configured model and language.

        -> ("ok" | "bad" | "missing" | "auth" | "inconclusive", vendor code, message).
        "bad": the model IS listed but not as an ASR model, or without our language (conclusive, FATAL).
        "missing": the gateway answered 200, lists ASR models of the same id namespace (halohalo/, fw/,
        nvidia/ ...) but not ours. The gateway lists only engines that are ready now, so this is either a typo
        or that engine being down: NON_FATAL, retried inside reconnect_window_ms (rule A').
        "inconclusive": no ASR model of that namespace (engine swap, background boot, lost engine: Nemotron
        can stay listed while the ASR engine swaps), a non-200 or no answer. "auth": 401/403 or TLS.
        """
        self._model_missing = None
        assert self.config is not None
        params = self.config.params
        base_url = str(params.get("base_url") or "")
        try:
            entries = await transport.fetch_models(  # entries: id, kind, languages / language
                base_url,
                str(params.get("api_key") or ""),
                bool(params.get("tls_verify", True)),
                MODELS_TIMEOUT_S,
            )
        except transport.TransportError as e:
            if e.code in ("unauthorized", "tls_error"):
                # 401/403 keep the HTTP status as vendor code, as on the websocket path
                return "auth", (str(e.status) if e.status else e.code), str(e)
            return "inconclusive", e.code, str(e)
        model = str(params.get("model") or "")
        entry = next((e for e in entries if e["id"] == model), None)
        if (entry is None and self.config._model_from_language
                and transport.lists_kind(entries, "asr", model)):
            # Only params.language was set. The gateway builds its halohalo/asr-<lang> ids from its configured
            # languages whenever the ASR engine is up (serve/gateway.py models()), so with any halohalo/ ASR id
            # listed the absence of ours is conclusive: use a listed ASR model that declares the language, else
            # FATAL model_not_found at once (no retrying for a model that will not appear).
            chosen = self._model_for_language(entries, str(params.get("language") or ""))
            if chosen is None:
                served = sorted(e["id"] for e in entries if e.get("kind") in (None, "asr"))
                return ("bad", "model_not_found",
                        f"no ASR model on {transport.http_base(base_url)}/v1/models serves language "
                        f"{params.get('language')!r} ({model!r} is not listed); ASR models: {', '.join(served)}")
            self.ten_env.log_info(f"{model!r} is not listed: using {chosen['id']!r}, which declares language "
                                  f"{params.get('language')!r}", category=LOG_CATEGORY_KEY_POINT)
            params["model"] = model = chosen["id"]
            self.config._model_from_language = False
            if model.startswith("nvidia/nemotron") and "finalize_timeout_ms" not in self.config.model_fields_set:
                # a Nemotron final is a batch decode (gateway finalize timer 30 s): wait past it, as for a
                # configured Nemotron model (config.apply_defaults)
                self.config.finalize_timeout_ms = NEMOTRON_FINALIZE_TIMEOUT_MS
            entry = chosen
        if entry is not None and not params.get("language"):
            langs = transport.model_languages(entry)
            if len(langs) == 1:  # only a model was configured: use the one language it declares
                params["language"] = langs[0]
                self.ten_env.log_info(f"language {langs[0]!r} taken from GET /v1/models for {model!r}",
                                      category=LOG_CATEGORY_KEY_POINT)
        if entry is None and not transport.lists_kind(entries, "asr", model):
            return ("inconclusive", "no_asr_models",
                    "GET /v1/models lists no ASR model of this engine (engine swap, boot or outage)")
        bad = transport.check_model(entries, model, "asr",
                                    str(params.get("language") or "") or None, base_url)
        if bad is not None:
            if bad[0] == "model_not_found" and not any(e["id"] == model for e in entries):
                self._model_missing = bad[1]
                return "missing", bad[0], bad[1]
            return "bad", bad[0], bad[1]
        return "ok", "", ""

    @staticmethod
    def _model_for_language(entries: list[dict], language: str) -> dict | None:
        """A listed ASR model that declares language: the gateway's default one first, then listing order."""
        if not language:
            return None
        cands = [e for e in entries if e.get("kind") in (None, "asr")
                 and language in transport.model_languages(e)]
        return next((e for e in cands if e.get("default")), cands[0] if cands else None)

    async def _report_model_missing(self, message: str) -> None:
        """One NON_FATAL model_not_found per outage window (rule D); the outage clock starts here."""
        mgr = self.reconnect_manager
        if mgr is not None:
            mgr.start_outage()
        if mgr is None or mgr.should_report():
            await self._send_error(
                ModuleError(module=MODULE_NAME_ASR, code=ModuleErrorCode.NON_FATAL_ERROR.value, message=message),
                ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="model_not_found", message=message),
            )

    async def _send_reconnect_error(self, error: ModuleError, vendor_info: ModuleErrorVendorInfo | None = None) -> None:
        """Reconnect-loop errors: while the model is missing from /v1/models they carry model_not_found."""
        if self._model_missing is not None:
            vendor_info = ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="model_not_found",
                                                message=self._model_missing)
            error = ModuleError(module=error.module, code=error.code,
                                message=f"{error.message}; {self._model_missing}")
        await self._send_error(error, vendor_info)

    async def _preflight(self) -> bool:
        """Verify the configured model against GET /v1/models until one check is conclusive.

        False only after a FATAL (bad key, TLS, model listed without our language or kind). An inconclusive
        check, or a model missing from the listing (reported NON_FATAL), proceeds: the handshake and the
        reconnect window handle a gateway that is swapping, booting or has that engine down, and
        _preflight_ok stays False so the next start/reconnect checks again.
        """
        verdict, code, message = await self._models_verdict()
        if verdict in ("auth", "bad"):
            await self._preflight_fatal(message, code)
            return False
        if verdict == "missing":
            self.ten_env.log_warn(f"models preflight: {message}; retrying within reconnect_window_ms")
            await self._report_model_missing(message)
            return True
        if verdict == "inconclusive":
            self.ten_env.log_warn(f"models preflight inconclusive: {message}")
            return True
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
        """HTTP statuses that are FATAL at once (bad key)."""
        return status in FATAL_HTTP

    async def _classify_connect_failure(self, e: Exception) -> tuple[bool, str | None, str]:
        """-> (fatal, vendor code override, message). 400/404 re-check GET /v1/models (FATAL only if conclusive)."""
        status = getattr(e, "status_code", None)
        if getattr(e, "code", None) == "tls_error" or self._is_fatal_status(status):
            return True, None, str(e)
        if status in SOFT_4XX:
            verdict, vcode, message = await self._models_verdict()
            if verdict in ("bad", "auth"):
                return True, vcode, f"{e}: {message}"
            if verdict == "missing":
                return False, vcode, f"{e}: {message}"
            self.ten_env.log_warn(f"handshake {status} with the model not conclusively missing ({message}): retrying")
            if status == 404:  # the gateway's 404 is unknown_model (NRSP v1 section 1)
                return False, "model_not_found", f"{e}: model {self.config.params.get('model')!r} not served now"
        return False, None, str(e)

    async def _report_connect_failure(self, e: Exception, fatal: bool | None = None,
                                      vendor_code: str | None = None, message: str | None = None) -> None:
        status = getattr(e, "status_code", None)
        code = vendor_code or getattr(e, "code", None)  # NRSPConnectError, e.g. "tls_error"
        if fatal is None:
            fatal, override, message = await self._classify_connect_failure(e)
            code = override or code
        message = message or str(e)
        self.ten_env.log_error(f"Failed to start NYO ASR connection: {message}")
        vendor_info = ModuleErrorVendorInfo(
            vendor=VENDOR_NAME,
            code=code or (str(status) if status is not None else "connect_failed"),
            message=message,
        )
        if fatal:
            self._fatal = True
        # NON_FATAL connect errors are rate-limited per outage (the first one is always reported)
        if fatal or self.reconnect_manager is None or self.reconnect_manager.should_report():
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
                vendor_info,
            )
        await self.on_disconnected(
            code=status or 1006, message=message, vendor_info=vendor_info
        )

    # ------------------------------------------------------------ reconnect
    def _schedule_reconnect(self) -> None:
        if self._stop_requested or self._fatal or self.stopped:
            return
        if self.reconnect_manager:
            self.reconnect_manager.start_outage()
        if self._reconnect_task and not self._reconnect_task.done():
            # A loop is running (possibly inside on_open of a socket that just died): it re-checks
            # is_connected() after its attempt and keeps going, so nothing is lost here.
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    def _cancel_reconnect_task(self) -> asyncio.Task | None:
        task, self._reconnect_task = self._reconnect_task, None
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            return task
        return None

    async def _await_reconnect_task_cancelled(self, timeout: float = 5.0) -> None:
        task = self._cancel_reconnect_task()
        if task is not None:
            await asyncio.wait({task}, timeout=timeout)

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
            fatal, vcode, message = await self._classify_connect_failure(e)
            if fatal:
                await self._report_connect_failure(e, fatal=True, vendor_code=vcode, message=message)  # sets _fatal
                return
            if self.reconnect_manager:
                self.reconnect_manager.set_retry_after(getattr(e, "retry_after", None))
            # vendor code as on the start_connection path: the handshake's HTTP status ("503", "429"), else
            # the error's own code, else connect_failed
            code = vcode or getattr(e, "code", None) or (str(status) if status is not None else "connect_failed")
            try:
                e.vendor_code = code  # read by ReconnectManager.handle_reconnect for the asr_error it reports
            except Exception:
                pass
            await self.on_disconnected(
                code=status or 1006,
                message=message,
                vendor_info=ModuleErrorVendorInfo(vendor=VENDOR_NAME, code=code, message=message),
            )
            raise

    async def _reconnect_loop(self) -> None:
        mgr = self.reconnect_manager
        if mgr is None:
            return
        try:
            while not self._stop_requested and not self._fatal and not self._idle_closed:
                if mgr.attempts and mgr.window_exceeded():
                    # sockets kept opening and dying before they proved stable: the outage is still on
                    await self._send_reconnect_error(
                        ModuleError(module=MODULE_NAME_ASR, code=ModuleErrorCode.FATAL_ERROR.value,
                                    message=f"no stable NYO ASR connection within {mgr.window_ms} ms"),
                        ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="connect_failed",
                                              message="connection keeps closing"),
                    )
                    return
                ok = await mgr.handle_reconnect(
                    connection_func=self._reconnect_once,
                    error_handler=self._send_reconnect_error,
                    vendor_name=VENDOR_NAME,
                    vendor_code="connect_failed",
                )
                # Done only when actually connected: the socket can die while on_open replays
                # (its on_close then finds this loop still running and does not start another).
                if self.is_connected() or self._stop_requested or self._fatal or self._idle_closed:
                    return
                if ok and mgr.window_exceeded():  # sockets that keep closing right after the upgrade
                    await self._send_reconnect_error(
                        ModuleError(module=MODULE_NAME_ASR, code=ModuleErrorCode.FATAL_ERROR.value,
                                    message=f"no stable NYO ASR connection within {mgr.window_ms} ms"),
                        ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="connect_failed",
                                              message="connection keeps closing"),
                    )
                    return
                self.ten_env.log_debug(
                    f"Reconnection attempt did not connect. Status: {mgr.get_attempts_info()}"
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
            try:  # keep the base class' Keep-buffer size true while live frames queue behind the replay
                self.buffered_frames_size = max(0, self.buffered_frames_size - len(buffered.get_buf()))
            except Exception:
                pass
            metadata, _ = buffered.get_property_to_json("metadata")
            if metadata:
                try:
                    md = copy.deepcopy(json.loads(metadata))
                    self.metadata = md
                    self.session_id = md.get("session_id", self.session_id)
                except Exception:
                    pass
            if not await self._send_frame(buffered):
                # the socket died mid-replay: keep this frame and the rest, in order, for the next connection
                self._requeue_front(buffered)
                return

    def _requeue_front(self, frame: AudioFrame) -> None:
        """Put frame back at the head of the Keep buffer (no await: nothing can be queued in between)."""
        rest = [frame]
        while True:
            try:
                rest.append(self.buffered_frames.get_nowait())
            except asyncio.QueueEmpty:
                break
        size = 0
        for f in rest:
            self.buffered_frames.put_nowait(f)
            try:
                size += len(f.get_buf())
            except Exception:
                pass
        self.buffered_frames_size = size

    # ----------------------------------------------------------- finalize
    @override
    async def finalize(self, _session_id: str | None) -> None:
        """One NRSP finalize per asr_finalize; each gets exactly one asr_finalize_end with its own id.

        TEN dispatches on_data concurrently, so a second asr_finalize can arrive while this one waits:
        the id is captured here and the end is keyed by it (never by the shared self.finalize_id).
        """
        assert self.config is not None
        fid = str(self.finalize_id or self.get_finalize_fallback_id())
        self.last_finalize_timestamp = int(datetime.now().timestamp() * 1000)
        self._open_finalizes.append(fid)
        rec = await self._finalize_socket()
        if rec is None:
            self.ten_env.log_warn(
                "asr_finalize: service not connected.",
                category=LOG_CATEGORY_KEY_POINT,
            )
            # the dead socket's last interim is all this turn has (on_open would drop it): promote it once,
            # as when the socket closes during the wait
            await self._promote_last_interim(fid, None, "finalize while disconnected")
            await self._finalize_end(fid)
            return
        self._fin_finals.setdefault(fid, 0)
        self._client_fids.append(fid)
        try:
            self._fin_t0, self._fin_id, self._fin_ttlw_ms, self._fin_server_ttlw = time.perf_counter(), fid, None, False
            async with self._replay_lock:  # never between replayed frames
                # the turn this finalize closes: the latest interim now, or one that starts before the audio sent
                # so far (the server cuts there; a later turn's interim starts at or after the cut)
                self._fin_snap[fid] = (self._latest_interim_key(),
                                       self.audio_timeline.get_total_user_audio_duration())
                await rec.send_finalize(fid)
            ok = await rec.wait_finalize_end(fid, self.config.finalize_timeout_ms)
            if not ok and (rec is not self.recognition or not rec.is_connected()):
                # the socket closed during the wait: the turn died with it, as in on_close
                await self._promote_last_interim(fid, rec, "socket closed during finalize", on_close=True)
            elif not ok:
                await self._on_finalize_timeout(fid, rec)
            elif self._fin_finals.get(fid, 0) == 0:
                # finalize_end without a final (gateway engine lost/reopened, or its own finalize
                # timeout): the last interim is all this turn has; promote it once.
                await self._promote_last_interim(fid, rec, "finalize_end without a final")
            elif self._fin_ttlw_ms is not None and not self._fin_server_ttlw:
                await self.send_vendor_metrics({"client_ttlw_ms": self._fin_ttlw_ms})
        except asyncio.CancelledError:
            self.ten_env.log_warn(
                "asr_finalize: wait for finalize_end was cancelled.",
                category=LOG_CATEGORY_KEY_POINT,
            )
            raise  # finally still sends this finalize's asr_finalize_end
        except Exception as e:
            self.ten_env.log_error(f"asr_finalize failed: {e}")
            await self._on_finalize_timeout(fid, rec)
        finally:
            self._fin_finals.pop(fid, None)
            self._fin_ended.discard(fid)
            self._fin_snap.pop(fid, None)
            if self._fin_id == fid:
                self._fin_t0 = None
            await self._finalize_end(fid)

    async def _finalize_socket(self):
        """The socket to finalize on, once on_open has replayed the Keep buffer (a finalize must not overtake
        audio captured before it); None while disconnected. Waits at most finalize_timeout_ms."""
        assert self.config is not None
        deadline = time.monotonic() + self.config.finalize_timeout_ms / 1000
        while self._replaying and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        rec = self.recognition
        return rec if rec is not None and rec.is_connected() else None

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
        await self._promote_last_interim(fid, rec, "finalize timeout")

    def _take_interim(self, fid: str | None):
        """Remove and return the interim fid may promote, never a later turn's: the most recently updated of the
        one that was latest when fid was sent and those starting before the audio sent by then (server ms on
        this socket). No snapshot (finalize while disconnected, socket close): the latest one."""
        snap = self._fin_snap.get(fid) if fid is not None else None
        if snap is None:
            key = self._latest_interim_key()
        else:
            snap_key, point_ms = snap
            keys = [k for k, m in self._interims.items()
                    if k == snap_key or ("start_ms" in m and int(m.get("start_ms") or 0) < point_ms)]
            key = keys[-1] if keys else _NO_INTERIM
        if key is _NO_INTERIM or key not in self._interims:
            return None  # nothing was pending, or its final has arrived since
        pending = self._interims.pop(key)
        # interims inside the promoted span are this turn too (e.g. the original session's interim of audio a
        # re-opened engine re-decoded); one that starts earlier is an earlier turn whose final is still due
        if "start_ms" in pending:
            start = int(pending.get("start_ms") or 0)
            end = start + int(pending.get("duration_ms") or 0)
            for k in [k for k, m in self._interims.items()
                      if "start_ms" in m and start <= int(m.get("start_ms") or 0) < max(end, start + 1)]:
                del self._interims[k]
        return pending

    async def _promote_last_interim(self, fid: str | None, rec, why: str, on_close: bool = False) -> None:
        """Emit this turn's last interim as its final (once: it is removed) and arm late-final suppression.

        fid None: the socket closed unexpectedly (on_close): every turn still pending dies with it and is promoted,
        in start order; the finals are marked promoted_on_close."""
        if fid is None:
            taken = []
            while (p := self._take_interim(None)) is not None:
                taken.append(p)
            taken.sort(key=lambda m: int(m.get("start_ms") or 0))
            for p in taken:
                if p.get("text"):
                    self.ten_env.log_info(f"promoting last interim ({why})")
                    await self._emit_asr_result(p, final=True, promoted="close")
            return
        pending = self._take_interim(fid)
        if pending and pending.get("text"):
            self.ten_env.log_info(f"asr_finalize {fid}: promoting last interim ({why})" if fid is not None
                                  else f"promoting last interim ({why})")
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
                    "socket closed: promoted last interim, nothing to suppress (the next socket is a new "
                    "gateway session: it never re-decodes audio sent on this one)"
                )
            await self._emit_asr_result(pending, final=True, promoted="close" if on_close else "timeout")

    async def _finalize_end(self, fid: str) -> None:
        """asr_finalize_end for fid, exactly once (send_asr_finalize_end reads self.finalize_id)."""
        try:
            self._open_finalizes.remove(fid)
        except ValueError:
            return
        if not self._open_finalizes:
            self.last_finalize_timestamp = 0
        self.finalize_id = fid
        await self.send_asr_finalize_end()

    # ------------------------------------------------------------- results
    async def _emit_asr_result(
        self, message: dict[str, Any], final: bool, promoted: str | None = None
    ) -> None:
        assert self.config is not None
        text = message.get("text", "")
        if not text and not (final and self._answers_open_finalize(message)):
            # Empty interims, and empty finals the server's endpointing cut on silence, carry nothing.
            # An empty final that answers an explicit asr_finalize is forwarded (as Deepgram does): it
            # closes the turn, so the caller sees the final for its finalize even when nothing was said.
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
            **({f"promoted_on_{promoted}": True} if promoted else {}),
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
        self._replaying = True
        try:
            await self._on_open()
        finally:
            self._replaying = False

    async def _on_open(self) -> None:
        delay = int(datetime.now().timestamp() * 1000) - self.connection_start_timestamp
        self.ten_env.log_info(
            "vendor_status_changed: on_open", category=LOG_CATEGORY_VENDOR
        )
        await self.send_connect_delay_metrics(delay)
        if self.reconnect_manager:
            # ends the outage only once this socket proves stable (a result / finalize_end, or 10 s up)
            self.reconnect_manager.note_open()
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
        self._idle_closed = False
        self._model_missing = None  # the gateway accepted this model
        await self.on_connected()
        async with self._replay_lock:
            await self._flush_buffered_audio_frames()
            self._live = True  # no await since the queue was seen empty: nothing can slip in between

    @override
    async def on_result(self, message: dict[str, Any]) -> None:
        final = bool(message.get("final", False))
        if self.reconnect_manager:
            self.reconnect_manager.mark_stable()
        if final:
            self._attribute_final(message)  # also for a final dropped below: the finalize did get its final
        if self._suppress_late_final:
            # Drop the real final of the turn we promoted on a finalize timeout. A result belongs to
            # that turn when it carries the promoted utterance id, the finalize_id we armed with, or
            # a start_ms inside the promoted span while coming from a newer re-opened engine session
            # (`r<N>-` id with N above the promoted generation: a replayed decode of audio already
            # reported final). Interims keep the flag; a final of another, later turn ends it (one that ends before the
            # promoted span is an earlier turn's re-decode: emitted, the flag stays).
            # Never the promoted turn: a result that names another finalize this addon sent, or a replayed one
            # that runs past the promoted span (a later turn merged into the replayed audio, e.g. with
            # endpointing_ms=0 the re-opened engine does not cut the replay off the next turn).
            msg_fid = (message.get("metadata") or {}).get("finalize_id")
            other_turn = msg_fid is not None and msg_fid != self._promoted_fid and (
                msg_fid in self._open_finalizes or msg_fid in self._client_fids)
            in_span = False
            before_span = False  # an earlier turn (it ends before the promoted one starts), e.g. its re-decode
            if self._promoted_span is not None and "start_ms" in message:
                start = int(message["start_ms"] or 0)
                end = start + int(message.get("duration_ms") or 0)
                slack = int(self.config.params.get("endpointing_ms") or 0) + PROMOTED_SPAN_SLACK_MS  # type: ignore[union-attr]
                before_span = start < self._promoted_span[0] and end <= self._promoted_span[0] and not other_turn
                in_span = (not before_span and start < self._promoted_span[1]
                           and end <= self._promoted_span[1] + slack)
            same = not other_turn and (
                (self._promoted_id is not None and message.get("id") == self._promoted_id)
                or (self._promoted_fid is not None and msg_fid == self._promoted_fid)
                or (in_span and _reopen_gen(message.get("id")) > self._promoted_gen)
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
            if same and not final:
                return  # an interim of the promoted turn would reopen a closed turn
            identified = self._promoted_id is not None or self._promoted_span is not None
            if (final and not before_span) or not identified:
                # only a final of another (later) turn ends the suppression: the next turn's interim can arrive
                # before the promoted turn's replayed final (finals decode in their own worker). Without an id or
                # span the promoted turn's final cannot be told apart: any other result ends it, as before.
                self._disarm_late_final()
        if final:
            self._note_final(message)
        elif message.get("text"):
            self._note_interim(message)
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

    def _answers_open_finalize(self, message: dict[str, Any]) -> bool:
        """True when this final answers an asr_finalize that has not ended yet."""
        if not self._open_finalizes:
            return False
        fid = (message.get("metadata") or {}).get("finalize_id")
        return fid is None or str(fid) in self._open_finalizes

    def _attribute_final(self, message: dict[str, Any]) -> None:
        """Count a final for the finalize it names (metadata.finalize_id); a final naming no finalize counts for
        the oldest pending one. A final naming an id this addon never sent (server endpointing `auto-N`,
        `r1-auto-N`) answers no asr_finalize and is not counted."""
        fid = (message.get("metadata") or {}).get("finalize_id")
        if fid is not None:
            if str(fid) in self._fin_finals:
                self._fin_finals[str(fid)] += 1
            return
        for pending in self._fin_finals:  # insertion order: oldest first
            if pending not in self._fin_ended:
                self._fin_finals[pending] += 1
                return

    @override
    async def on_finalize_end(self, finalize_id: str) -> None:
        if self.reconnect_manager:
            self.reconnect_manager.mark_stable()
        if finalize_id in self._fin_finals:
            self._fin_ended.add(finalize_id)  # later finals belong to a younger finalize

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
        if fatal and code == IDLE_TIMEOUT_CODE:
            # the gateway closed an idle socket (no audio for its idle window): not an error for the
            # graph; on_close skips the reconnect and the next audio frame reconnects (on_audio_frame)
            self.ten_env.log_info(f"server closed the idle session ({message}); reconnecting on the next audio frame",
                                  category=LOG_CATEGORY_KEY_POINT)
            self._idle_closed = True
            return
        if fatal and code == UNKNOWN_MODEL_CODE:
            # rule A': the gateway does not serve the model right now (its engine may be down, or a typo).
            # NON_FATAL model_not_found; on_close reconnects and the preflight re-checks /v1/models; the
            # reconnect window ends in one FATAL model_not_found if it never comes back.
            self._model_missing = message
            if self.reconnect_manager is not None:
                self.reconnect_manager.note_closed()
                self.reconnect_manager.start_outage()
                if not self.reconnect_manager.should_report():
                    self.ten_env.log_info(f"unknown_model not reported (rate limit): {message}")
                    return
            await self._send_error(
                ModuleError(module=MODULE_NAME_ASR, code=ModuleErrorCode.NON_FATAL_ERROR.value, message=message),
                ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="model_not_found", message=message),
            )
            return
        if fatal and code in RECONNECTABLE_CODES:
            # engine swap / engine lost / slow consumer: the socket ends, the service does not.
            # NON_FATAL; on_close reconnects with backoff and the Keep buffer replays. Rate-limited per
            # outage like connect failures: the first is reported, then at most one per 10 s.
            fatal = False
            if self.reconnect_manager is not None:
                self.reconnect_manager.note_closed()  # this socket is ending: settle "stable" before the report
            if self.reconnect_manager is not None and not self.reconnect_manager.should_report():
                self.ten_env.log_info(f"reconnectable server error not reported (rate limit): {code}")
                return
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
        if self.reconnect_manager and not expected:
            self.reconnect_manager.note_closed()  # a socket that stayed up >= 10 s ended the outage
        if not expected:
            self.recognition = None
        await self.on_disconnected(
            code=0 if expected else 1006, message=reason
        )
        if expected or self._stop_requested or self._fatal:
            return
        # The turn in progress dies with this socket's gateway session (the next socket starts a new one and only
        # gets the audio captured while disconnected): its last interim is all it has, so promote it once.
        # No late-final suppression is armed: nothing on a new socket re-decodes this audio, and on_open resets
        # the id space and timeline.
        await self._promote_last_interim(None, None, "socket closed unexpectedly", on_close=True)
        if self._idle_closed:
            self.ten_env.log_info("idle close: waiting for audio before reconnecting")
            return
        self._schedule_reconnect()

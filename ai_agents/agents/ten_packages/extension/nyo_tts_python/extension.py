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
import re
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import websockets
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
from ten_ai_base.tts2 import DATA_FLUSH, AsyncTTS2BaseExtension
from ten_runtime import AsyncTenEnv, Data

from . import transport
from .config import NYOTTSConfig, language_from_model
from .const import (
    DEFAULT_VOICE_ALIASES,
    IDLE_TIMEOUT_CODE,
    MODELS_TIMEOUT_S,
    PLAIN_DEFAULT_VOICES,
    RECONNECT_BACKOFF_CAP_S,
    RECONNECT_ERROR_INTERVAL_S,
    RECONNECTABLE_CODES,
    RETRY_AFTER_CAP_S,
    STABLE_CONNECTION_S,
    VALID_SAMPLE_RATES,
    STALE_SET_MAX,
    VENDOR_NAME,
)
from .tts_client import NYOTTSClient, NYOTTSConnectError

FATAL_HTTP = {401, 403}
# on_stop waits this long for the cancelled background tasks (timers, resends, background connects) to finish
STOP_TASKS_TIMEOUT_S = 1.0
# connect-time failures that retrying cannot fix (preflight and TLS)
_FATAL_CODES = {"unknown_voice", "unsupported_sample_rate", "tls_error", "401", "403"}
# request-scoped errors that say the route itself is wrong (not a passing failure of one segment): they end the TEN
# request; any other non-fatal error naming one of its segments costs that segment only (engine codes included)
_REQUEST_ENDING_CODES = {"stale_request", "unknown_voice", "unsupported_voice", "unsupported_sample_rate",
                         "unknown_model", "unsupported_language", "unauthorized", "unsupported_version"}
# rule A': "not listed now" may be an engine that is down: NON_FATAL, retried inside reconnect_window_ms
_MAYBE_DOWN_CODES = {"model_not_found", "language_not_supported"}
# 400/404 also appear while a gateway restarts: FATAL only when GET /v1/models conclusively says "not served"
SOFT_4XX = {400, 404}
# NRSP audio_end.reason -> TEN reason
_END_REASONS = {
    "end": TTSAudioEndReason.REQUEST_END,
    "flushed": TTSAudioEndReason.INTERRUPTED,
    "interrupted": TTSAudioEndReason.INTERRUPTED,
    "error": TTSAudioEndReason.ERROR,
}
# sentence-final punctuation, optionally followed by closing quotes / brackets, at the very end of an increment
_SENTENCE_END = re.compile(r"[.!?\u2026\u3002\uff01\uff1f][\"'\u201d\u2019\u00bb\u300d\u300f)\]}]*\Z")


def ends_sentence(text: str) -> bool:
    """The increment ends with sentence-final punctuation (optionally followed by closing quotes / brackets) that does
    not follow a digit.

    The NYO engines split sentences at punctuation FOLLOWED BY whitespace (or at ``end``). main_python
    parse_sentences cuts after every '.', ',', '?', '!' and leaves the whitespace at the start of the NEXT chunk, so
    only a reply's last sentence lacks it; but it also cuts inside "3.50" or "nyolab.ai", whose next chunk starts
    without whitespace. The text therefore always goes out verbatim; a sentence end may get a delayed " " increment
    (boundary_hint_ms) or end its segment (segment_max_ms). After a digit ("3.") it is never treated as one: the
    next chunk or segment_idle_ms covers it."""
    m = _SENTENCE_END.search(text or "")
    return m is not None and not (m.start() > 0 and text[m.start() - 1].isdigit())


# what the engines' sentence splitter cuts at when whitespace follows (modules/tts/frontend.py split_sentences:
# (?<=[.!?\u2026])\s+): the punctuation must be the last character before the whitespace
_HINT_SPLITS = re.compile(r"[.!?\u2026]\Z")


def hint_splits(text: str) -> bool:
    """A " " right after this increment makes the engine speak its sentence now: it ends with ``. ! ? …`` that does
    not follow a digit. After a closing quote or bracket (``po."``) or a CJK stop the engine does not split at the
    space, so no hint is sent (the text is never altered); the idle end (segment_idle_ms) covers that sentence."""
    m = _HINT_SPLITS.search(text or "")
    return m is not None and not (m.start() > 0 and text[m.start() - 1].isdigit())


# segment_max_ms hard cut (a segment older than segment_max_ms + HARD_CUT_EXTRA_MS): a chunk that ends with a clause
# mark (, ; : and their CJK forms), a list number at a line start ("\n2."), or a newline, optionally followed by closing
# quotes / brackets / whitespace. parse_sentences cuts only after punctuation, so this is never mid-word; a digit-period
# after anything but a line start ("₱3." of "₱3.50") is not a cut, so decimals and prices stay whole
# A clause mark not right after a digit ("₱1," + "250" and "10:" + "30" stay whole), a list number at a line
# start ("\n2."; a chunk that only begins with digits, like the "234." of "₱1,234.50", is not one), or a newline.
_HARD_CUT = re.compile(r"(?:(?<!\d)[,;:，；：、]|\n[ \t]*\d+\.|\n)[\"'”’»」』)\]}\s]*\Z")
HARD_CUT_EXTRA_MS = 2000
# an idle-open TEN request (no text_input_end, every segment spoken) checks this often whether the base class buffered
# the next request; if so it ends (REQUEST_END) so finish_request releases that request
IDLE_OPEN_POLL_S = 0.05


def hard_cut(text: str) -> bool:
    """The increment ends where a long segment may be cut although no sentence ended (segment_max_ms hard cut)."""
    return _HARD_CUT.search(text or "") is not None


def _is_conn_error(e: BaseException) -> bool:
    return isinstance(e, (ConnectionError, websockets.exceptions.ConnectionClosed))


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


def served_verdict(entries: list[dict] | None, model: str, voice: str, language: str,
                   where: str = "/v1/models") -> tuple[str, str, bool] | None | bool:
    """Does the gateway serve this TTS route? From a GET /v1/models listing (shared rules A, A').

    Returns ``(code, message, conclusive)`` when the listing says no, ``True`` when it lists the model, and
    ``None`` when it cannot tell. A missing id counts only when the listing names a TTS model in the same id
    namespace (``transport.lists_kind``): during an engine swap the gateway still lists unrelated models. Even
    then it is not conclusive (rule A'): the gateway lists only engines that are ready now, so "missing" may be
    "that engine is down"; ``conclusive`` is False and the caller keeps retrying inside reconnect_window_ms.
    A language no listed TTS model declares is not conclusive either (that language's engine may be down).
    Conclusive: the id is listed as another kind, or a voice the effective language's route refuses.
    Language: the gateway routes any language one of its TTS models declares (``languages`` list or
    ``language`` string), so ``language`` must be in that union. Voice, as the gateway checks it
    (serve/speech.py check_voice): against the voice catalogue (``voice_details``) of the effective language
    (case-insensitive; the default aliases pass); without a catalogue, against the route's ``voices`` allow-list
    (SAPIN_TTS_VOICES: exact names; "" and "default" always pass; an empty list or "*" lets every name through).
    """
    if not entries:
        return None
    entries = [e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]
    tts = [e for e in entries if e.get("kind") in (None, "tts")]
    names = ", ".join(sorted(e["id"] for e in tts))
    entry = next((e for e in entries if e["id"] == model), None)
    if entry is not None and entry.get("kind") not in (None, "tts"):
        return ("model_not_found",
                f"tts model {model!r} is a {entry.get('kind')} model, not tts ({where}); tts models: {names}", True)
    if entry is None:
        if not transport.lists_kind(entries, "tts", model):
            return None
        return ("model_not_found", f"tts model {model!r} is not served by {where} now (engine down, or a typo); "
                                   f"tts models: {names}", False)
    served_langs = sorted({lang for e in tts for lang in transport.model_languages(e)})
    if language and served_langs and language not in served_langs:
        return ("language_not_supported",
                f"tts language {language!r} is not served by {where} now (engine down, or a typo; it lists "
                f"{', '.join(served_langs)})", False)
    lang = language or language_from_model(model) or next(iter(transport.model_languages(entry)), "")
    raw = voice or ""
    v = raw.strip().lower()
    if v and lang:
        cat = next((e for e in tts if e.get("voice_details") and lang in transport.model_languages(e)), None)
        if cat is not None:
            ids = [str(x) for x in cat.get("voices") or []]
            if v not in DEFAULT_VOICE_ALIASES and ids and v not in {i.lower() for i in ids}:
                return ("unknown_voice", f"voice {voice!r} is not served for language {lang!r} by {where}; "
                                         f"voices: {', '.join(ids)}", True)
        else:
            route = next((e for e in tts if lang in transport.model_languages(e)), entry)
            allow = [str(x) for x in route.get("voices") or [] if isinstance(x, (str, int))]
            if allow and "*" not in allow and raw not in PLAIN_DEFAULT_VOICES and raw not in allow:
                return ("unknown_voice", f"voice {voice!r} is not served for language {lang!r} by {where} (a route "
                                         f"without a voice catalogue takes only: {', '.join(allow)})", True)
    return True


@dataclass
class _Request:
    request_id: str
    sent_ts: float | None = None  # first text frame written to the socket (TTFB origin)
    audio_start_ts: float | None = None  # tts_audio_start emitted (request_event_interval_ms origin)
    first_audio_ts: float | None = None
    closed: asyncio.Event = field(default_factory=asyncio.Event)  # set when the request is finalized
    last_activity: float = field(default_factory=time.monotonic)
    audio_bytes: int = 0
    audio_start_sent: bool = False
    end_sent: bool = False
    sent_text: bool = False  # any non-blank text went to the server (blank text before it is never sent)
    open: bool = True
    gaps: int = 0
    watchdog: asyncio.Task | None = None
    sent_on: Any = None  # the NYOTTSClient its text went to: only that socket's close/error may fail it
    chunks: list = field(default_factory=list)  # (text, end) as sent, for the one resend after an idle close
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # keeps text order across a resend
    replied: set = field(default_factory=set)  # segments the gateway answered (audio_start / audio / audio_end)
    resent_wires: set = field(default_factory=set)  # segments already resent once on a fresh socket (idle race)
    resending: bool = False  # a resend is in progress: the old socket's close must not fail the request
    resend_task: asyncio.Task | None = None  # later chunks wait for it, so the fresh socket sees them in order
    end_reason: TTSAudioEndReason | None = None  # set when finalized here
    connecting: bool = False  # waiting for a socket for its next chunk
    # segments: one TEN request = one or more NRSP requests (segment_idle_ms); audio of all of them is this request's
    seg_no: int = 1
    wire_ids: list = field(default_factory=list)  # NRSP request ids of the segments that carry text, in order
    seg_text: bool = False  # the current segment carries non-blank text
    seg_ended: bool = False  # end:true went out for the current segment
    seg_done: set = field(default_factory=set)  # segments whose audio_end(end) arrived
    seg_base_ms: int = 0  # timeline position where the segment now playing starts (gateway: FIFO per socket)
    seg_failed: set = field(default_factory=set)  # segments lost to a transient failure (the request goes on)
    seg_started: float | None = None  # monotonic time the current segment's first text went out (segment_max_ms)
    seg_timer: asyncio.Task | None = None  # idle end (segment_idle_ms)
    hint_timer: asyncio.Task | None = None  # delayed sentence-boundary " " (boundary_hint_ms)
    firing: set = field(default_factory=set)  # timer tasks past their sleep: never cancelled (they hold send_lock)
    chunk_gen: int = 0  # bumped by every chunk about to be sent: a timer armed before it does nothing
    idle_watch: asyncio.Task | None = None  # idle-open: ends the request once the base buffered the next one

    @property
    def wire_id(self) -> str:
        """The NRSP request id of the current segment: the TEN id, then ``<id>~2``, ``<id>~3``, ..."""
        return self.request_id if self.seg_no == 1 else f"{self.request_id}~{self.seg_no}"

    def unfinished(self) -> list:
        return [w for w in self.wire_ids if w not in self.seg_done]


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
        self._retired: set[NYOTTSClient] = set()  # sockets closed after a FATAL; their reader may still be reporting
        self._pending_config: NYOTTSConfig | None = None  # update_configs received mid-request
        self._pending_checked = False  # /v1/models confirmed _pending_config
        self._outage_start: float | None = None  # monotonic time the gateway became unreachable (rule C)
        self._conn_up_at: float | None = None  # when the current/last socket opened
        self._conn_proven = False  # that socket completed a request
        self._outage_err_at: float | None = None  # last NON_FATAL connection error reported in this outage (rule D)
        self._retry_after: float | None = None  # from the last 429/503 rejection
        self._last_connect_error: NYOTTSConnectError | None = None
        self._connected_ev = asyncio.Event()  # set while a usable socket exists
        self._idle_close = False  # the gateway announced idle_timeout; its close is not a failure
        # bumped by every applied update_configs: a socket opened (or a preflight run) under an older generation
        # carries stale voice/language/model and is never used for a request
        self._config_gen = 0
        self.recorder_map: dict[str, PCMWriter] = {}
        # request ids this extension already ended -> their reason; later chunks of them are ignored (bounded; a
        # tts_flush clears it, so an id reused after a flush is a new request)
        self._finished: OrderedDict[str, TTSAudioEndReason] = OrderedDict()
        self._tts_busy = 0  # request_tts calls in progress: a chunk is arriving, no request is idle-open meanwhile
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
        retired = list(self._retired)  # their close tasks are cancelled below; the sockets are closed here
        pending = [t for t in (self._reconnect_task, self._prewarm_task) if t is not None and not t.done()]
        self._cancel_reconnect_task()
        if self._prewarm_task and not self._prewarm_task.done():
            self._prewarm_task.cancel()
        if self._req and self._req.watchdog:
            self._req.watchdog.cancel()
            pending.append(self._req.watchdog)
        # segment timers, idle-close resends, background connects, retire closes: none outlives the extension
        bg = [t for t in self._bg_tasks if not t.done() and t is not asyncio.current_task()]
        for t in bg:
            t.cancel()
        pending = [t for t in pending + bg if t is not asyncio.current_task()]
        if pending:
            _, late = await asyncio.wait(pending, timeout=STOP_TASKS_TIMEOUT_S)
            if late:
                ten_env.log_warn(f"NYO TTS: {len(late)} background task(s) still running after "
                                 f"{STOP_TASKS_TIMEOUT_S:g} s at stop")
        if self.client:
            await self.client.stop()
            self.client = None
        for old in {*retired, *self._retired}:
            await old.stop()
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

    async def on_data(self, ten_env: AsyncTenEnv, data: Data) -> None:
        if data.get_name() == DATA_FLUSH:
            self._finished.clear()  # the graph starts over: an id it sends after the flush is a new request
        await super().on_data(ten_env, data)

    # ------------------------------------------------------------ connection
    def _usable(self) -> bool:
        """A socket that is open, takes new requests and was opened under the current config."""
        c = self.client
        return c is not None and c.is_connected() and getattr(c, "config_gen", None) == self._config_gen

    async def _ensure_connected(self) -> None:
        async with self._connect_lock:
            if self._usable():
                return
            # a connect that finishes after an update_configs was applied is discarded by _connect (False): go again
            while True:
                if self._fatal:
                    raise NYOTTSConnectError("fatal error earlier; not reconnecting", 401)
                if await self._connect() is not False:
                    return

    async def _served(self, cfg: NYOTTSConfig) -> tuple[str, str, bool] | None | bool:
        """served_verdict() for ``cfg`` from a fresh GET /v1/models. 401/403 and TLS failures raise
        NYOTTSConnectError (FATAL codes); an unreachable or non-JSON listing is inconclusive (None)."""
        try:
            entries = await transport.fetch_models(cfg.base_url, cfg.api_key, cfg.tls_verify, MODELS_TIMEOUT_S)
        except transport.TransportError as e:
            if e.code == "unauthorized":
                raise NYOTTSConnectError(str(e), e.status, code=str(e.status)) from e
            if e.code == "tls_error":
                raise NYOTTSConnectError(str(e), None, code="tls_error") from e
            self.ten_env.log_warn(f"NYO TTS /v1/models check inconclusive: {e}")
            return None
        except Exception as e:  # noqa: BLE001  (e.g. http.client.BadStatusLine from another service)
            self.ten_env.log_warn(f"NYO TTS /v1/models check inconclusive: {e}")
            return None
        return served_verdict(entries, cfg.model, cfg.voice, cfg.language, f"{transport.http_base(cfg.base_url)}/v1/models")

    async def _preflight(self) -> None:
        """GET /v1/models before the first connect: fail fast on a bad model, voice, language, key or certificate.

        A model or language missing from the listing raises a NON_FATAL ``model_not_found`` /
        ``language_not_supported`` (rule A': the engine may be down; the reconnect loop retries and the FATAL after
        reconnect_window_ms carries that code). Otherwise raises NYOTTSConnectError with a FATAL code
        (``model_not_found`` for an id listed as another kind, ``unknown_voice``, ``tls_error``, ``401``/``403``); _connect reports it like a failed handshake. An
        inconclusive answer (gateway down or restarting, no TTS model listed) is not fatal: the check runs again
        on the next connect.
        """
        assert self.config is not None
        gen = self._config_gen
        verdict = await self._served(self.config)
        if gen != self._config_gen:
            return  # the config changed meanwhile: this verdict is about the old route; _connect starts over
        if isinstance(verdict, tuple):
            if verdict[2]:
                self._fatal = True
            raise NYOTTSConnectError(verdict[1], None, code=verdict[0])
        self._preflight_ok = verdict is True

    async def _connect(self) -> bool:
        """Open a socket for the current config. False when update_configs was applied meanwhile (nothing installed;
        the caller connects again); raises NYOTTSConnectError on failure."""
        assert self.config is not None
        self._connected_ev.clear()
        if self.client:
            old, self.client = self.client, None
            await old.stop()
        await self.on_connecting()
        gen = self._config_gen
        client = NYOTTSClient(self.config, self.ten_env)
        client.config_gen = gen
        client.on_audio_start = self._on_audio_start
        client.on_audio = self._on_audio
        client.on_audio_end = self._on_audio_end
        client.on_usage = self._on_usage
        client.on_alignment = self._on_alignment
        client.on_error = lambda msg, _c=client: self._on_error(msg, _c)
        client.on_closed = lambda expected, reason, _c=client: self._on_closed(expected, reason, _c)
        try:
            if not self._preflight_ok:
                await self._preflight()
                if gen != self._config_gen:
                    return False  # update_configs applied during the check: _ensure_connected starts over
            try:
                await client.start()
            except NYOTTSConnectError as e:
                if e.status_code in SOFT_4XX and not self._stopped:
                    # rule B: a gateway that restarts answers 400/404 too; FATAL only if /v1/models says "not served"
                    verdict = await self._served(self.config)
                    if isinstance(verdict, tuple):
                        if verdict[2]:
                            self._fatal = True
                        raise NYOTTSConnectError(f"{verdict[1]} (handshake HTTP {e.status_code}"
                                                 f"{', ' + e.body_code if e.body_code else ''})",
                                                 e.status_code, code=verdict[0]) from e
                    if e.code == "unknown_voice":
                        # the gateway refused the voice and the listing does not say the route is down: conclusive
                        self._fatal = True
                # the rejection body's code (tts_client._BODY_CODES) decides from here: unknown_voice and
                # unsupported_sample_rate FATAL, model_not_found / language_not_supported retried (rule A')
                raise
        except NYOTTSConnectError as e:
            self._last_connect_error = e
            self._connected_ev.clear()
            self._note_outage()
            self._retry_after = e.retry_after
            if e.code in _FATAL_CODES or e.status_code in FATAL_HTTP:
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
            elif e.code in _MAYBE_DOWN_CODES:
                # rule A': maybe the engine is down; say so (rate-limited) and keep retrying inside the window
                err = self._rate_limit(ModuleError(
                    message=str(e),
                    module=ModuleType.TTS,
                    code=ModuleErrorCode.NON_FATAL_ERROR,
                    vendor_info=ModuleErrorVendorInfo(vendor=VENDOR_NAME, code=e.code, message=str(e)),
                ))
                if err is not None:
                    await self._send_error("", err)
            raise
        if self._stopped:  # a connect that finished in the background after on_stop: leave nothing open
            await client.stop()
            raise NYOTTSConnectError("stopped while connecting", None, code="stopped")
        if gen != self._config_gen:
            # update_configs was applied while this socket was opening: it carries the old route. Close it;
            # _ensure_connected opens one with the current config (not an outage)
            self.ten_env.log_info("NYO TTS: config changed while connecting; reopening with the new config")
            await client.stop()
            return False
        self.client = client
        self._conn_up_at = time.monotonic()  # the outage clock resets only once this socket proves itself (rule C)
        self._conn_proven = False
        self._retry_after = None
        self._last_connect_error = None
        self._idle_close = False
        self._connected_ev.set()
        await self.on_connected()
        await self.metrics_connect_delay(client.connect_delay_ms)
        return True

    # ------------------------------------------------- outage clock (rules C, D)
    def _connection_proven(self, now: float) -> bool:
        return self._conn_proven or (self._conn_up_at is not None and now - self._conn_up_at >= STABLE_CONNECTION_S)

    def _note_outage(self) -> None:
        """The socket died or a connect failed. A new outage starts unless the last socket never proved itself
        (stayed up STABLE_CONNECTION_S or completed a request): a socket that opens and dies sooner keeps the clock."""
        now = time.monotonic()
        if self._outage_start is None or self._connection_proven(now):
            self._outage_start = now
            self._outage_err_at = None
        self._conn_up_at = None
        self._conn_proven = False

    def _mark_proven(self) -> None:
        self._conn_proven = True
        self._outage_start = None
        self._outage_err_at = None

    def _rate_limit(self, err: ModuleError) -> ModuleError | None:
        """Rule D: the first NON_FATAL connection error of an outage, then at most one per RECONNECT_ERROR_INTERVAL_S."""
        now = time.monotonic()
        if self._outage_err_at is not None and now - self._outage_err_at < RECONNECT_ERROR_INTERVAL_S:
            self.ten_env.log_warn(f"NYO TTS connection error not reported (rate limit): {err.message}",
                                  category=LOG_CATEGORY_VENDOR)
            return None
        self._outage_err_at = now
        return err

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
            if client is not None:
                self._retire(client)
        return err

    def _retire(self, client: NYOTTSClient) -> None:
        """Close a socket after a FATAL without cancelling its reader: the reader may be the task reporting that
        FATAL (tts_audio_end, tts_error, finish_request); it ends by itself on the closed socket."""
        self._retired.add(client)

        async def close() -> None:
            try:
                await client.stop(cancel_reader=False)
            finally:
                self._retired.discard(client)

        self._spawn(close())

    async def _send_error(self, request_id: str, err: ModuleError) -> None:
        err = self._error_once(err)
        if err is not None:
            await self.send_tts_error(request_id=request_id, error=err)

    def _reconnect_delay(self, attempt: int) -> float:
        # jittered exponential backoff, 0.5 s doubling, capped
        return random.uniform(0.5, 1.0) * min(0.5 * 2 ** (min(attempt, 16) - 1), RECONNECT_BACKOFF_CAP_S)

    def _reconnect_window_s(self) -> float:
        return (self.config.reconnect_window_ms if self.config else 120000) / 1000

    async def _reconnect_loop(self) -> None:
        """Retry until connected; FATAL connect_failed only once the outage is older than reconnect_window_ms."""
        last: Exception | None = None
        attempt = 0
        if self._outage_start is None:
            self._outage_start = time.monotonic()
        window = self._reconnect_window_s()
        try:
            while not (self._stopped or self._fatal):
                if self._usable():
                    return
                attempt += 1
                delay = self._reconnect_delay(attempt)
                ra, self._retry_after = self._retry_after, None
                if ra is not None:  # the gateway said when to come back (429/503)
                    delay = max(delay, min(ra, RETRY_AFTER_CAP_S))
                outage_start = self._outage_start if self._outage_start is not None else time.monotonic()
                remaining = outage_start + window - time.monotonic()
                if remaining <= 0:
                    last = last or self._last_connect_error or NYOTTSConnectError(
                        "connections keep dropping before they prove stable")
                    break
                await asyncio.sleep(min(delay, remaining))
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
                if not self._request_affected():
                    # idle: nobody waits for audio, so this is not FATAL yet; stop retrying in the background. The
                    # next request reconnects; if the outage is still on (older than the window), that is the FATAL.
                    self.ten_env.log_warn(
                        f"NYO TTS: gateway unreachable for more than {window:g} s while idle ({attempt} reconnect "
                        f"attempts, last: {last}); retrying stops until the next request", category=LOG_CATEGORY_VENDOR)
                    return
                await self._send_error(
                    "",
                    ModuleError(
                        message=f"NYO TTS: no usable gateway connection for more than {window:g} s "
                                f"({attempt} reconnect attempts): {last}",
                        module=ModuleType.TTS,
                        code=ModuleErrorCode.FATAL_ERROR,
                        vendor_info=ModuleErrorVendorInfo(
                            vendor=VENDOR_NAME, message=str(last),
                            # a model still missing from /v1/models when the window ends: say so (rule A')
                            code=last.code if getattr(last, "code", None) in _MAYBE_DOWN_CODES
                            else "connect_failed",
                        ),
                    ),
                )
        finally:
            if self._reconnect_task is asyncio.current_task():
                self._reconnect_task = None

    def _request_affected(self) -> bool:
        """A request is waiting for a socket or has work on the gateway (an open TEN request whose segments are all
        spoken, waiting for the LLM's next chunk, is not affected by an outage: its next chunk reconnects)."""
        req = self._req
        return req is not None and req.open and (req.connecting or not req.wire_ids or self._outstanding(req))

    @staticmethod
    def _outstanding(req: _Request) -> bool:
        """The gateway still owes this request something: audio_end of a segment, or the end of the request."""
        return req.end_sent or bool(req.unfinished())

    # ------------------------------------------------------------ requests
    async def request_tts(self, t: TTSTextInput) -> None:
        req = self._req
        if req is not None and req.request_id == t.request_id:
            self._cancel_idle_watch(req)  # a new chunk of the current request: it is not idle-open
        self._tts_busy += 1
        try:
            await self._request_tts(t)
        finally:
            self._tts_busy -= 1
        req = self._req
        if req is not None and req.request_id == t.request_id:
            await self._check_idle_open(req)

    async def _request_tts(self, t: TTSTextInput) -> None:
        if self._stopped:
            self.ten_env.log_debug(f"TTS stopped, skipping request_id: {t.request_id}")
            return
        # Never log the text itself (no prompt/PII in logs): id, length and end flag only.
        self.ten_env.log_info(
            f"Requesting TTS: request ID: {t.request_id}, chars: {len(t.text)}, "
            f"text_input_end: {t.text_input_end}",
            category=LOG_CATEGORY_TRANSCRIPTS,
        )
        req = self._req
        if t.request_id in self._finished or (req is not None and req.request_id == t.request_id
                                              and (not req.open or req.end_sent)):
            await self._ignore_late_chunk(t)
            return
        if req is None or req.request_id != t.request_id:
            if req is not None and req.open:  # base serialises requests; defensive only
                await self._cancel_active(TTSAudioEndReason.INTERRUPTED, finish=True)
            req = await self._begin_request(t)
            if self._pending_config is not None:  # update_configs that arrived mid-request
                # applied after the request exists, so a tts_flush meanwhile ends it (cancel_tts sees it open); the
                # fresh socket is opened by _connect_for_request below, raced against that flush
                cfg, self._pending_config = self._pending_config, None
                await self._apply_config(cfg, checked=self._pending_checked, reconnect=False)
                if not req.open or self._req is not req:
                    return
        if self._fatal or self.config is None:
            # stopped after a FATAL error (already reported once): end the request at its first chunk, final or
            # not, and contact nothing; its later chunks are late chunks (they release the base loop)
            req.end_sent = True
            await self._finalize_request(TTSAudioEndReason.ERROR)
            return
        blank = not t.text.strip()
        if t.text == "" and not t.text_input_end:
            return
        if not t.text_input_end and blank and not (req.seg_text and not req.seg_ended):
            # blank text before any real text of a segment is not sent: the gateway would open the request on it,
            # and an empty end is then answered here without ever telling the gateway (it would hold the socket's
            # worker until its stall timeout)
            return
        if t.text_input_end and blank and not req.sent_text:
            # Nothing to synthesise: end the request here instead of a server round trip
            # (the tts_guarder wants tts_audio_end within 500 ms for empty requests).
            req.end_sent = True
            await self._finalize_request(TTSAudioEndReason.REQUEST_END)
            return
        if t.text_input_end and blank and req.seg_ended:
            await self._end_after_last_segment(req)
            return
        self._disarm_segment(req)  # this chunk goes out: a pending idle end must not cut the segment before it
        resend = req.resend_task
        if resend is not None and not resend.done():
            # an idle-close resend was scheduled but may not hold send_lock yet: let it replay the chunks sent so
            # far first, so this chunk reaches the fresh socket once and after them (asyncio.wait never raises)
            await asyncio.wait({resend})
        try:
            async with req.send_lock:  # a resend after an idle close (below) keeps the request's text in order
                await self._connect_for_request(req)
                if self._stopped or not req.open or self._req is not req:
                    # flushed (or replaced, or stopping) while we waited for the socket: nothing may reach the server
                    self.ten_env.log_debug(f"request {t.request_id} ended while connecting; not sent")
                    return
                if req.seg_ended:
                    # the idle timer ended the segment while we waited: this chunk opens the next one
                    if blank:
                        if t.text_input_end:
                            await self._end_after_last_segment(req)
                        return
                    req.seg_no += 1
                    req.seg_ended = req.seg_text = False
                assert self.client is not None
                client = self.client
                if not blank and t.text[:1].isspace() and req.sent_on is client and self._hard_cut_due(req):
                    # hard cut BEFORE this chunk: a long segment with no cut point after its chunks (a price-only
                    # comma chain "Bigas ₱52," " itlog ₱8," ...: a mark after a digit is never a cut) ends here; the
                    # whitespace this chunk begins with is a word boundary ("₱1," + "250." has none: never cut)
                    await self._end_segment(req, client, "segment_max_ms hard cut before a whitespace chunk")
                    req.seg_no += 1
                    req.seg_ended = req.seg_text = False
                wire = req.wire_id
                text = t.text  # verbatim: a sentence boundary is hinted later (boundary_hint_ms), never inside "3.50"
                self.metrics_add_input_characters(len(t.text))
                if req.sent_ts is None:
                    req.sent_ts = time.perf_counter()
                req.sent_on = client
                if wire not in req.wire_ids:  # before the send: its audio_start may be read before send returns
                    req.wire_ids.append(wire)
                try:
                    await client.send_text(wire, text, t.text_input_end)
                except (ConnectionError, websockets.exceptions.ConnectionClosed):
                    if not await self._handed_to_resend(req, client):
                        raise
                    self.ten_env.log_info(f"request {t.request_id}: send crossed an idle close; the resend carries it")
                self._sent_chunk(req, wire, text, t.text_input_end)
                if not t.text_input_end and self._segment_too_long(req, text):
                    # a long fast reply: end the segment at this sentence end or hard cut (SpeechT5 speaks only at the
                    # end of its request and the gateway aborts one silent for tts_stall_s); the next chunk opens a new one
                    await self._end_segment(req, client, "segment_max_ms reached")
        except NYOTTSConnectError as e:
            await self._fail_request(req, e, status=e.status_code, fatal=bool(self._fatal),
                                     rate_limited=not self._fatal)
            self._schedule_reconnect()
        except Exception as e:
            self.ten_env.log_error(
                f"Error in request_tts for request ID {t.request_id} ({len(t.text)} chars): "
                f"{traceback.format_exc()}",
                category=LOG_CATEGORY_TRANSCRIPTS,
            )
            conn = _is_conn_error(e)
            if conn:
                self._note_outage()
                if req.open and self._req is req:
                    # the socket died under this chunk: its segment is lost, the TEN request goes on (next chunk
                    # reconnects); with text_input_end the request is over
                    if t.text_input_end:
                        req.end_sent = True
                    sock = req.sent_on
                    await self._segment_failed(req, self._rate_limit(ModuleError(
                        message=f"connection lost: {e}", module=ModuleType.TTS, code=ModuleErrorCode.NON_FATAL_ERROR,
                        vendor_info=ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="connection_lost", message=str(e)),
                    )), req.unfinished(), done=True, client=sock)
            else:
                await self._fail_request(req, e, status=None, fatal=False)
            self._schedule_reconnect()

    def _sent_chunk(self, req: _Request, wire: str, text: str, end: bool) -> None:
        req.chunks.append((wire, text, end))
        if text.strip():
            if not req.seg_text:
                req.seg_started = time.monotonic()
            req.sent_text = req.seg_text = True
        req.last_activity = time.monotonic()
        if end:
            req.end_sent = req.seg_ended = True
            self._start_watchdog(req)
        else:
            self._arm_segment(req)
            self._arm_hint(req, text)

    def _segment_age_ms(self, req: _Request) -> float | None:
        """Age of the open segment for segment_max_ms, or None when the rule does not apply."""
        limit = self.config.segment_max_ms if self.config else 0
        if not (limit > 0 and req.open and req.seg_text and not req.seg_ended and req.seg_started is not None):
            return None
        return (time.monotonic() - req.seg_started) * 1000

    def _hard_cut_due(self, req: _Request) -> bool:
        """The open segment is older than segment_max_ms + HARD_CUT_EXTRA_MS: it may be cut without a sentence end."""
        age_ms = self._segment_age_ms(req)
        return age_ms is not None and age_ms >= self.config.segment_max_ms + HARD_CUT_EXTRA_MS

    def _segment_too_long(self, req: _Request, text: str) -> bool:
        age_ms = self._segment_age_ms(req)
        if age_ms is None:
            return False
        if age_ms >= self.config.segment_max_ms and ends_sentence(text):
            return True
        # no sentence end in time (a numbered list, a price at each sentence end, a long comma chain): past
        # segment_max_ms + HARD_CUT_EXTRA_MS cut at a clause mark, a list number or a newline, so SpeechT5 still
        # ends its request before the gateway's stall timer (tts_stall_s, 10 s); a next chunk that begins with
        # whitespace is cut before (request_tts)
        return self._hard_cut_due(req) and hard_cut(text)

    async def _end_segment(self, req: _Request, client: NYOTTSClient, why: str) -> None:
        """end:true for the open segment (caller holds send_lock); the next chunk opens the next segment."""
        wire = req.wire_id
        await client.send_text(wire, "", True)
        req.chunks.append((wire, "", True))
        req.seg_ended = True
        req.last_activity = time.monotonic()
        self.ten_env.log_debug(f"request {req.request_id}: segment {wire} ended ({why})")
        self._start_watchdog(req)

    def _final_reason(self, req: _Request) -> TTSAudioEndReason:
        """How a TEN request that reached its end ends: REQUEST_END, or ERROR when a segment failed and no audio came."""
        return TTSAudioEndReason.ERROR if req.seg_failed and not req.audio_bytes else TTSAudioEndReason.REQUEST_END

    async def _segment_failed(self, req: _Request, err: ModuleError | None, wires: list, done: bool,
                              client: NYOTTSClient | None = None) -> None:
        """A transient failure (admission_rejected, engine_error, synthesis_truncated, a gateway cancel, a lost socket)
        cost these segments, not the TEN request: one NON_FATAL error; the next chunk opens a new segment. ``done``:
        no audio_end will come for them (the socket is gone). A request that already sent text_input_end ends once
        nothing is outstanding (REQUEST_END if any audio was produced, else ERROR)."""
        if not req.open or self._req is not req:
            return
        new = [w for w in wires if w not in req.seg_failed]
        req.seg_failed.update(new)
        if done:
            req.seg_done.update(wires)
            for c in {client, self.client} - {None}:
                for w in wires:
                    c.forget(w)  # frames still in flight on a dying socket must not shift the next segment's timeline
        if req.wire_id in wires:
            req.seg_ended = True  # the gateway ended it: never send to it again
        if new:
            self.ten_env.log_warn(f"NYO TTS request {req.request_id}: segment(s) {new} lost "
                                  f"({err.message if err else 'error not reported (rate limit)'}); the request goes on",
                                  category=LOG_CATEGORY_VENDOR)
            err = self._error_once(err)
            if err is not None:
                await self.send_tts_error(request_id=req.request_id, error=err)
        if not req.open or self._req is not req:
            return
        if req.end_sent:
            if not req.unfinished():
                await self._finalize_request(self._final_reason(req))
            else:
                self._start_watchdog(req)
        else:
            await self._check_idle_open(req)

    async def _handed_to_resend(self, req: _Request, client: NYOTTSClient) -> bool:
        """A send failed on a closing socket. If that close is an idle close the gateway announced (it never saw this
        request), the resend scheduled for it carries this chunk too: True. The reader may still be delivering the
        idle_timeout message, so wait (bounded) for it first."""
        if not req.resending:
            await client.wait_closed(1.0)
        return req.resending and req.open and self._req is req

    async def _end_after_last_segment(self, req: _Request) -> None:
        """text_input_end without text after the idle timer already ended the last segment: the request ends with
        that segment's audio_end (now, if it already arrived)."""
        req.end_sent = True
        if not req.unfinished():
            await self._finalize_request(self._final_reason(req))
        else:
            self._start_watchdog(req)

    # ------------------------------------------------------------ segments
    def _arm_segment(self, req: _Request) -> None:
        """segment_idle_ms: no new chunk for that long and no text_input_end -> end:true for the open segment, so the
        engine speaks what it buffered (its last sentence) now instead of waiting for an end that may never come."""
        idle = self.config.segment_idle_ms if self.config else 0
        if idle <= 0 or req.end_sent or req.seg_ended or not req.seg_text or not req.open:
            return

        async def end_idle(r: _Request, client: NYOTTSClient) -> None:
            await self._end_segment(r, client, f"{idle} ms without a chunk")

        req.seg_timer = self._spawn(self._timer(req, req.chunk_gen, idle / 1000, end_idle))

    def _arm_hint(self, req: _Request, text: str) -> None:
        """boundary_hint_ms: a chunk that ends a sentence where the engine's splitter cuts (hint_splits) and is not
        followed by another chunk within that long gets one " " increment (the engines split at punctuation FOLLOWED
        BY whitespace): the reply's last sentence is spoken now, not only at the idle end. A next chunk in time
        cancels it (it brings its own whitespace)."""
        ms = self.config.boundary_hint_ms if self.config else 0
        if ms <= 0 or req.end_sent or req.seg_ended or not req.seg_text or not req.open or not hint_splits(text):
            return

        async def hint(r: _Request, client: NYOTTSClient) -> None:
            wire = r.wire_id
            await client.send_text(wire, " ", False)
            r.chunks.append((wire, " ", False))

        req.hint_timer = self._spawn(self._timer(req, req.chunk_gen, ms / 1000, hint))

    def _disarm_segment(self, req: _Request) -> None:
        """A chunk is about to go out: the pending idle end and boundary hint are void."""
        req.chunk_gen += 1
        self._cancel_timers(req)

    @staticmethod
    def _cancel_idle_watch(req: _Request) -> None:
        t, req.idle_watch = req.idle_watch, None
        if t is not None and not t.done() and t is not asyncio.current_task():
            t.cancel()

    def _next_request_pending(self, req: _Request) -> bool:
        """The base class buffered a chunk of another request (tts2 _process_input_queue keeps it in
        _pending_messages until finish_request of the current one)."""
        pending = getattr(self, "_pending_messages", None) or {}
        return any(rid != req.request_id for rid in list(pending))

    def _chunk_queued(self, req: _Request) -> bool:
        """A chunk of ``req`` is already in the base input queue (about to reach request_tts)."""
        q = getattr(getattr(self, "input_queue", None), "_queue", None)
        try:
            return any(getattr(t, "request_id", None) == req.request_id for t in list(q or ()))
        except Exception:  # noqa: BLE001  (a private deque: never let it break the request)
            return False

    @staticmethod
    def _owes_nothing(req: _Request) -> bool:
        """No text_input_end yet, every segment's audio_end arrived and no segment is open."""
        return (req.open and not req.end_sent and not req.connecting and not req.unfinished()
                and (not req.seg_text or req.seg_ended))

    def _idle_open(self, req: _Request) -> bool:
        """``req`` owes nothing and nothing can add to it right now: no chunk arriving or queued, no idle-close resend,
        no timer acting. (A boundary hint / idle end still armed for a segment that already ended sends nothing: the
        timers act only on an open segment, which _owes_nothing excludes.)"""
        if self._stopped or self._fatal or self._req is not req or not self._owes_nothing(req):
            return False
        if self._tts_busy or req.resending or req.send_lock.locked() or req.firing:
            return False
        if req.resend_task is not None and not req.resend_task.done():
            return False
        return not self._chunk_queued(req)

    def _idle_open_expired(self, req: _Request) -> bool:
        """idle_open_end_ms (0 = off): the idle-open request got nothing (chunk, audio) for that long."""
        ms = self.config.idle_open_end_ms if self.config else 0
        return ms > 0 and (time.monotonic() - req.last_activity) * 1000 >= ms

    async def _check_idle_open(self, req: _Request) -> None:
        """A TEN request without text_input_end stays open after its segments were spoken (turn-detection graphs
        reset it by flush). But once the base class holds the NEXT request, that one waits for our finish_request:
        end this request now (REQUEST_END) so the next one runs. While it is idle-open and no next request is
        buffered yet, a short-interval watcher checks again (cancelled by a new chunk, the end, a flush or stop)."""
        if self._idle_open(req) and (self._next_request_pending(req) or self._idle_open_expired(req)):
            why = ("the next request is waiting" if self._next_request_pending(req)
                   else f"no chunk for {self.config.idle_open_end_ms} ms (idle_open_end_ms)")
            self.ten_env.log_info(f"request {req.request_id}: every segment spoken, no text_input_end, and {why}; "
                                  f"ending it", category=LOG_CATEGORY_KEY_POINT)
            await self._finalize_request(self._final_reason(req))
            return
        if self._owes_nothing(req) and self._req is req and not self._stopped and (
                req.idle_watch is None or req.idle_watch.done()):
            req.idle_watch = self._spawn(self._watch_idle_open(req))

    async def _watch_idle_open(self, req: _Request) -> None:
        try:
            while not self._stopped and self._req is req and self._owes_nothing(req):
                await asyncio.sleep(IDLE_OPEN_POLL_S)
                if self._idle_open(req) and (self._next_request_pending(req) or self._idle_open_expired(req)):
                    await self._check_idle_open(req)
                    return
        except Exception as e:  # noqa: BLE001
            self.ten_env.log_warn(f"NYO TTS: idle-open watcher of {req.request_id} failed: {e}")
        finally:
            if req.idle_watch is asyncio.current_task():
                req.idle_watch = None

    @staticmethod
    def _cancel_timers(req: _Request) -> None:
        NYOTTSExtension._cancel_idle_watch(req)
        for name in ("seg_timer", "hint_timer"):
            t = getattr(req, name)
            setattr(req, name, None)
            if t is not None and not t.done() and t not in req.firing and t is not asyncio.current_task():
                t.cancel()

    async def _timer(self, req: _Request, gen: int, delay: float, action) -> None:
        """Run ``action(req, client)`` under send_lock after ``delay`` unless a chunk went out meanwhile (``gen``) or
        the open segment ended."""
        await asyncio.sleep(delay)
        task = asyncio.current_task()
        req.firing.add(task)
        try:
            for _ in range(3):  # at most: the socket closes, its resend runs, the action goes to the fresh socket
                async with req.send_lock:
                    if (gen != req.chunk_gen or self._stopped or not req.open or self._req is not req
                            or req.end_sent or req.seg_ended or not req.seg_text):
                        return
                    client = req.sent_on
                    if client is not None and client.is_connected():
                        await action(req, client)
                        return
                    resend = req.resend_task
                # the socket is closing. If that close is an idle close the gateway announced, a resend moves the
                # segment to a fresh socket: let the reader deliver the close (bounded), follow the resend, and act
                # there. Otherwise that socket's close fails the segment and the next chunk opens a new one.
                if client is not None and (resend is None or resend.done()):
                    await client.wait_closed(1.0)
                    resend = req.resend_task
                if resend is not None and not resend.done():
                    await asyncio.wait({resend})
                if req.sent_on is client:
                    return
        except Exception as e:  # noqa: BLE001  (a lost socket is handled by its close)
            self.ten_env.log_warn(f"NYO TTS: segment timer of {req.request_id} failed: {e}")
        finally:
            req.firing.discard(task)

    def _maybe_resend(self, sock: NYOTTSClient | None) -> bool:
        """The gateway closed ``sock`` for idle_timeout. If the open request's text went to it and nothing came back
        (the gateway had not seen the request: it would not have been idle), send that request once more on a fresh
        socket instead of failing it. True if a resend runs (or already ran for this close)."""
        req = self._req_on(sock)
        if req is None:
            return False
        if req.resending:
            return True
        # the segments still owed: all on this socket (a segment on an older socket is done or failed). Any answer
        # means the gateway saw them (it would not have been idle); each segment is resent at most once
        pending = req.unfinished()
        if not pending or any(w in req.replied or w in req.resent_wires for w in pending):
            return False
        req.resent_wires.update(pending)
        req.resending = True
        if sock is not None:
            sock.mark_unusable()
            for w in pending:
                sock.forget(w)
        if sock is self.client:
            self._connected_ev.clear()
        self.ten_env.log_info(f"NYO TTS: idle_timeout raced request {req.request_id}; resending it on a fresh socket",
                              category=LOG_CATEGORY_VENDOR)
        req.resend_task = self._spawn(self._resend(req, pending))
        return True

    async def _resend(self, req: _Request, wires: list) -> None:
        try:
            async with req.send_lock:
                try:
                    await self._connect_for_request(req)
                    if self._stopped or not req.open or self._req is not req:
                        return
                    assert self.client is not None
                    req.sent_on = self.client
                    if req.first_audio_ts is None:
                        req.sent_ts = time.perf_counter()
                    for wire, text, end in list(req.chunks):
                        if wire in wires:  # never a segment already spoken
                            await self.client.send_text(wire, text, end)
                    req.last_activity = time.monotonic()
                finally:
                    req.resending = False
        except NYOTTSConnectError as e:
            if req.open and self._req is req:
                await self._fail_request(req, e, status=e.status_code, fatal=bool(self._fatal),
                                         rate_limited=not self._fatal)
            self._schedule_reconnect()
        except Exception as e:  # noqa: BLE001
            conn = _is_conn_error(e)
            if conn:
                self._note_outage()
            if req.open and self._req is req:
                if conn:
                    # the fresh socket died under the resend: like any lost socket this costs the resent segments
                    # only; the TEN request goes on (its next chunk reconnects)
                    await self._segment_failed(req, self._rate_limit(ModuleError(
                        message=f"connection lost: {e}", module=ModuleType.TTS, code=ModuleErrorCode.NON_FATAL_ERROR,
                        vendor_info=ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="connection_lost", message=str(e)),
                    )), [w for w in wires if w in req.wire_ids], done=True, client=req.sent_on)
                else:
                    await self._fail_request(req, e, status=None, fatal=False)
            self._schedule_reconnect()

    async def _ignore_late_chunk(self, t: TTSTextInput) -> None:
        """A chunk of a request this extension already ended (error mid-stream, or after text_input_end).

        The base input loop re-selects that request as its processing request for this chunk and waits for a
        finish_request that would never come: every later request would stay buffered there. Release it."""
        self.ten_env.log_warn(f"Received a message for finished request_id '{t.request_id}'; ignoring.")
        # only once it is ended here: a duplicate chunk after text_input_end of a request still playing must not
        # release the base loop early (the next request would interrupt it). "Ended here" is the current request
        # once closed (a tts_flush clears _finished but the late chunk of that request still pins the base) or any
        # id in _finished; never the open request.
        req = self._req
        current = req is not None and req.request_id == t.request_id
        if current and req.open:
            return
        if not current and t.request_id not in self._finished:
            return
        if self._processing_request_id == t.request_id:
            reason = (req.end_reason if current and req.end_reason is not None
                      else self._finished.get(t.request_id, TTSAudioEndReason.ERROR))
            await self.finish_request(request_id=t.request_id, reason=reason)

    async def _begin_request(self, t: TTSTextInput) -> _Request:
        req = _Request(request_id=t.request_id)
        self._req = req
        if t.metadata:
            self.session_id = t.metadata.get("session_id", self.session_id)
        await self._setup_recorder(t.request_id)
        return req

    async def _connect_for_request(self, req: _Request) -> None:
        """A usable socket for ``req``: connect now, or wait (bounded by request_timeout_ms) while the background
        reconnect loop works through an outage. Raises NYOTTSConnectError when that fails; returns early, without
        a socket, if the request is finalized (flushed) meanwhile."""
        assert self.config is not None
        req.connecting = True
        try:
            await self._connect_for_request_inner(req)
        finally:
            req.connecting = False

    async def _connect_for_request_inner(self, req: _Request) -> None:
        assert self.config is not None
        deadline = time.monotonic() + self.config.request_timeout_ms / 1000
        while not self._stopped:
            try:
                await self._connect_unless_closed(req)  # returns early if the request was flushed meanwhile
                return
            except NYOTTSConnectError as e:
                if self._fatal or self._stopped or not req.open:
                    if not req.open:
                        return
                    raise
                self._schedule_reconnect()
                loop_task = self._reconnect_task
                remaining = deadline - time.monotonic()
                if loop_task is None or remaining <= 0:
                    raise NYOTTSConnectError(
                        f"not connected within request_timeout_ms ({self.config.request_timeout_ms} ms): {e}",
                        e.status_code, code=e.code) from e
                waiters = [asyncio.ensure_future(self._connected_ev.wait()),
                           asyncio.ensure_future(req.closed.wait())]
                try:
                    await asyncio.wait([*waiters, loop_task], timeout=remaining,
                                       return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for w in waiters:
                        w.cancel()
                if not req.open:
                    return
                if not self._usable() and time.monotonic() >= deadline:
                    raise NYOTTSConnectError(
                        f"not connected within request_timeout_ms ({self.config.request_timeout_ms} ms): {e}",
                        e.status_code, code=e.code) from e

    async def _connect_unless_closed(self, req: _Request) -> bool:
        """_ensure_connected(), raced against the request being finalized (flush): a handshake the gateway holds
        (up to 15 s during an engine swap) must not keep the base input loop, and so the next request, waiting.
        False if the request closed first; the connect then completes in the background for the next request."""
        if self._usable():
            return True
        connect = asyncio.ensure_future(self._ensure_connected())
        closed = asyncio.ensure_future(req.closed.wait())
        try:
            await asyncio.wait([connect, closed], return_when=asyncio.FIRST_COMPLETED)
        finally:
            closed.cancel()
        if connect.done():
            connect.result()  # raises the connect error
            return True
        self._bg_tasks.add(connect)
        connect.add_done_callback(self._background_connect_done)
        return False

    def _background_connect_done(self, task: asyncio.Task) -> None:
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        e = task.exception()
        if e is not None:
            self.ten_env.log_warn(f"NYO TTS background connect failed: {e}")
            if not (self._fatal or self._stopped):
                self._schedule_reconnect()

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
            for wire in req.unfinished() or [req.request_id]:
                await self.client.flush(wire)
        await self._finalize_request(reason, finish=finish)

    async def _fail_request(
        self, req: _Request, e: Exception, status: int | None, fatal: bool, rate_limited: bool = False
    ) -> None:
        if fatal:
            self._fatal = True
        # the real cause (model_not_found, language_not_supported, engine_unavailable, ...) is the vendor code; the
        # HTTP status, if any, stays in the message
        message = str(e)
        if status is not None and f"HTTP {status}" not in message:
            message = f"{message} (HTTP {status})"
        # a websockets ConnectionClosed has a deprecated ``code`` (its close code): never read it
        cause = None if _is_conn_error(e) else getattr(e, "code", None)
        vendor_code = ("connection_lost" if _is_conn_error(e) else str(cause) if cause
                       else str(status) if status is not None
                       else "connect_failed" if isinstance(e, NYOTTSConnectError) else "request_failed")
        err = ModuleError(
            message=message,
            module=ModuleType.TTS,
            code=ModuleErrorCode.FATAL_ERROR if fatal else ModuleErrorCode.NON_FATAL_ERROR,
            vendor_info=ModuleErrorVendorInfo(vendor=VENDOR_NAME, code=vendor_code, message=message),
        )
        if rate_limited and not fatal:
            err = self._rate_limit(err)
        if self.client:
            for wire in req.wire_ids or [req.request_id]:
                self.client.forget(wire)
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
        req.end_reason = reason
        req.closed.set()
        rid = req.request_id
        self._finished[rid] = reason
        while len(self._finished) > STALE_SET_MAX:
            self._finished.popitem(last=False)
        error = self._error_once(error)  # the request still ends (ERROR); the error only if not after a FATAL
        if req.watchdog and req.watchdog is not asyncio.current_task():
            req.watchdog.cancel()
        self._cancel_timers(req)
        if not req.audio_start_sent:  # no audio: close the start/end pair, interval 0
            req.audio_start_sent = True
            await self.send_tts_audio_start(request_id=rid)
        # request_event_interval_ms: tts_audio_start -> tts_audio_end (excludes TTFB), as deepgram_tts does
        interval_ms = (int((time.perf_counter() - req.audio_start_ts) * 1000)
                       if req.audio_start_ts is not None else 0)
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
            for wire in req.wire_ids or [rid]:
                self.client.forget(wire)
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
        """Fails a request the gateway owes something (a segment's audio_end, or the request's end) and that stays
        silent for request_timeout_ms. A request whose segments are all spoken and that waits for the LLM's next
        chunk (no text_input_end yet) owes nothing: the watchdog stops, the next segment end starts it again."""
        assert self.config is not None
        timeout = self.config.request_timeout_ms / 1000
        while req.open:
            if not self._outstanding(req):
                return
            remaining = req.last_activity + timeout - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(remaining)
        if not req.open or self._req is not req or not self._outstanding(req):
            return
        self.ten_env.log_error(
            f"NYO TTS request {req.request_id} silent for {self.config.request_timeout_ms} ms",
            category=LOG_CATEGORY_VENDOR,
        )
        if self.client:
            for wire in req.unfinished() or [req.request_id]:
                await self.client.flush(wire)
        await self._fail_request(
            req, TimeoutError(f"no audio_end within {self.config.request_timeout_ms} ms"),
            status=None, fatal=False,
        )

    # --------------------------------------------------- NRSP -> TEN callbacks
    def _is_current(self, rid: str | None) -> _Request | None:
        """The open request that NRSP request id ``rid`` (the TEN id or one of its segments) belongs to."""
        if self._stopped:  # never touch ten_env once on_stop has begun
            return None
        req = self._req
        if req is None or not req.open or rid is None:
            return None
        return req if rid == req.request_id or rid in req.wire_ids else None

    async def _on_audio_start(self, rid: str, sample_rate: int, msg: dict) -> None:
        req = self._is_current(rid)
        if req is None:
            return
        req.replied.add(rid)
        req.last_activity = time.monotonic()
        # the gateway synthesises one request at a time per socket: this segment starts where the audio so far ends
        req.seg_base_ms = self._audio_ms(req.audio_bytes)
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
        tid = req.request_id
        now = time.perf_counter()
        req.replied.add(rid)
        req.last_activity = time.monotonic()
        if req.first_audio_ts is None:
            req.first_audio_ts = now
            # TTFB: first text written to the socket -> first non-empty audio frame (excludes connect time)
            ttfb_ms = int((now - (req.sent_ts if req.sent_ts is not None else now)) * 1000)
            req.audio_start_sent = True
            req.audio_start_ts = now
            await self.send_tts_audio_start(request_id=tid)
            if not req.open:  # flushed while tts_audio_start was being sent: its audio_end is out already
                return
            await self.send_tts_ttfb_metrics(
                request_id=tid,
                ttfb_ms=ttfb_ms,
                extra_metadata={"model": self.config.model if self.config else ""},
            )
            if not req.open:
                return
        ts = req.seg_base_ms + offset_ms  # one timeline across the request's segments
        expected = self._audio_ms(req.audio_bytes)
        if abs(ts - expected) > 20:  # server-side gap or reorder
            req.gaps += 1
            self.ten_env.log_warn(
                f"NRSP audio gap on {rid}: offset {ts} ms, expected {expected} ms",
                category=LOG_CATEGORY_VENDOR,
            )
        await self._write_dump(tid, pcm)
        if not req.open:  # flushed meanwhile: nothing of it may follow its tts_audio_end
            return
        req.audio_bytes += len(pcm)
        self.metrics_add_recv_audio_chunks(pcm)
        if self._prof_path:
            t0 = time.perf_counter_ns()
            await self.send_tts_audio_data(pcm, ts)
            self._prof.append(time.perf_counter_ns() - t0)
        else:
            await self.send_tts_audio_data(pcm, ts)

    async def _on_audio_end(self, rid: str, reason: str, audio_ms: int, msg: dict) -> None:
        if reason == "end":
            self._mark_proven()  # this socket completed a request: the outage clock may reset (rule C)
        req = self._is_current(rid)
        if req is None:
            return
        req.replied.add(rid)
        tid = req.request_id
        # gateway-measured latencies, exported next to the client-side ones; every
        # finished request also reports the model id it ran on.
        server = {f"server_{k}": msg[k] for k in ("ttfb_ms", "ttlw_ms") if k in msg}
        server["model"] = msg.get("model") or self.config.model
        await self.send_metrics(
            ModuleMetrics(
                id=self.get_uuid(), module=ModuleType.TTS, vendor=self.vendor(),
                metrics=server,
                metadata=self.update_metadata(tid, None),
            ),
            tid,
        )
        if not req.open:
            return
        if self.client:
            self.client.forget(rid)
        req.last_activity = time.monotonic()
        if reason in ("flushed", "interrupted") or _END_REASONS.get(reason) == TTSAudioEndReason.ERROR:
            # our own flush is acknowledged inside the client and never reaches here (its request is closed). This
            # is the gateway's own cancel (engine swap, shutdown drain) or a failed segment (its request-scoped
            # error, reported already, came first): that segment is lost, the TEN request goes on
            msg_text = (f"the gateway ended request {rid!r} itself (audio_end reason {reason!r})"
                        if reason != "error" else f"request {rid!r} failed on the gateway (audio_end reason 'error')")
            await self._segment_failed(req, ModuleError(
                message=msg_text,
                module=ModuleType.TTS,
                code=ModuleErrorCode.NON_FATAL_ERROR,
                vendor_info=ModuleErrorVendorInfo(
                    vendor=VENDOR_NAME, code="server_cancelled" if reason != "error" else "request_failed",
                    message=msg_text),
            ), [rid], done=True)
            return
        ended_here = req.seg_ended  # we ended the current segment (idle timer / segment_max_ms) without the request
        req.seg_done.add(rid)
        if rid == req.wire_id:
            req.seg_ended = True
        if req.unfinished() or (not req.end_sent and (rid != req.wire_id or ended_here)):
            # a segment of a request that goes on (the idle timer ended it; more text or text_input_end follows)
            self.ten_env.log_debug(f"request {tid}: segment {rid} spoken")
            await self._check_idle_open(req)
            return
        await self._finalize_request(self._final_reason(req))

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
        req = self._is_current(msg.get("request_id"))
        if not self.config or not self.config.enable_words or req is None:
            return
        rid, base = req.request_id, req.seg_base_ms
        words = [
            TTSWord(
                word=str(w.get("word", w.get("text", ""))),
                start_ms=base + int(w.get("start_ms", 0)),
                duration_ms=int(w.get("duration_ms", 0)),
            )
            for w in msg.get("words") or []
        ]
        await self.send_tts_text_result(
            TTSTextResult(
                request_id=rid,
                text=str(msg.get("text", "")),
                start_ms=base + int(msg["start_ms"]) if "start_ms" in msg else (words[0].start_ms if words else base),
                duration_ms=int(msg.get("duration_ms", 0)),
                words=words,
                text_result_end=False,
                metadata=self.update_metadata(rid, None),
            )
        )

    def _req_on(self, client: NYOTTSClient | None) -> _Request | None:
        """The open request whose text went to ``client`` (any open request when ``client`` is None). A request
        that has not reached a socket yet is not failed by an old socket's close: it connects on its own."""
        req = self._req
        if req is None or not req.open or self._stopped:
            return None
        if client is not None and req.sent_on is not client:
            return None
        if req.wire_ids and not self._outstanding(req):
            return None  # all its segments are spoken: it waits for the LLM, not for this socket
        return req

    async def _on_error(self, msg: dict[str, Any], client: NYOTTSClient | None = None) -> None:
        rid = msg.get("request_id")
        vendor = msg.get("vendor") or {}
        code = str(msg.get("code", "unknown"))
        fatal = bool(msg.get("fatal", False))
        message = str(vendor.get("message") or msg.get("message") or code)
        if vendor.get("name") and vendor.get("name") != VENDOR_NAME:
            message = f"[{vendor.get('name')}] {message}"
        sock = client or self.client
        if code == IDLE_TIMEOUT_CODE and self._maybe_resend(sock):
            # the open request's text went to this socket unanswered: it is resent on a fresh one
            self._idle_close = True
            return
        if fatal and sock is not None and ((code == IDLE_TIMEOUT_CODE and self._req_on(sock) is None)
                                           or code in RECONNECTABLE_CODES):
            # the gateway closes this socket next: no new request may be written to it (the next one opens a
            # fresh socket at once instead of racing the close frame)
            sock.mark_unusable()
            if sock is self.client:
                self._connected_ev.clear()
        if code == IDLE_TIMEOUT_CODE:
            # the gateway closes sockets idle for SAPIN_IDLE_TIMEOUT_S: not an error; the next request reconnects
            self.ten_env.log_info(f"NYO TTS socket closed by the gateway: idle_timeout ({message})",
                                  category=LOG_CATEGORY_VENDOR)
            self._idle_close = True
            return
        if fatal and code in RECONNECTABLE_CODES:
            # fatal for this socket only (engine swap, engine down, slow consumer): the close that follows is
            # handled by _on_closed, which reconnects; the in-flight request ends with a NON_FATAL error
            self.ten_env.log_warn(f"NYO TTS reconnectable server error {code}: {message}",
                                  category=LOG_CATEGORY_VENDOR)
            if client is None or client is self.client:
                self._note_outage()
            req = self._req_on(client)
            if req is not None:
                err = self._rate_limit(ModuleError(
                    message=message,
                    module=ModuleType.TTS,
                    code=ModuleErrorCode.NON_FATAL_ERROR,
                    vendor_info=ModuleErrorVendorInfo(
                        vendor=VENDOR_NAME, code=str(vendor.get("code") or code), message=message),
                ))
                if req.wire_ids:
                    # the segments on this socket are lost; the TEN request goes on with its next chunk (new socket)
                    await self._segment_failed(req, err, req.unfinished(), done=True, client=client)
                else:
                    await self._finalize_request(TTSAudioEndReason.ERROR, error=err)
            return
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
            req = self._is_current(rid) if rid else None
            if req is None or (client is not None and req.sent_on is not client) or rid in req.seg_failed:
                return  # late text for a request (or a failed segment) already ended; harmless race
            # the gateway refuses the OPEN request (its id already ended on this socket, e.g. reused after a
            # flush): no audio will come, end it now instead of waiting for the watchdog
            await self._finalize_request(TTSAudioEndReason.ERROR, error=ModuleError(
                message=message, module=ModuleType.TTS, code=ModuleErrorCode.NON_FATAL_ERROR,
                vendor_info=ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="stale_request", message=message),
            ))
            return
        seg_req = self._is_current(rid) if (not fatal and rid is not None) else None
        if seg_req is not None and rid in seg_req.wire_ids and code not in _REQUEST_ENDING_CODES:
            # a passing failure of one segment (admission_rejected, engine_error, the engine's own tts_engine_error,
            # synthesis_truncated, text_too_long, any future engine code): it costs that segment only (its
            # audio_end{error} follows and is then a no-op); the TEN request goes on with its next chunk
            await self._segment_failed(seg_req, err, [rid], done=True, client=client)
            return
        if fatal:
            self._fatal = True
        req = self._is_current(rid) if rid else (self._req if self._req and self._req.open else None)
        if req is not None:
            await self._finalize_request(TTSAudioEndReason.ERROR, error=err)
        else:
            await self._send_error(rid or "", err)

    async def _on_closed(self, expected: bool, reason: str, client: NYOTTSClient | None = None) -> None:
        if expected or self._stopped:
            return
        if client is not None and self.client is not None and client is not self.client:
            return  # a replaced socket: nothing of the current state is on it
        self._connected_ev.clear()
        idle, self._idle_close = self._idle_close, False
        if idle and self._maybe_resend(client):
            await self.on_disconnected(code=1001, message="idle_timeout")
            return
        req = self._req_on(client)
        if idle and req is None:
            # idle_timeout with nothing in flight: report the status change only; reconnect on the next request
            await self.on_disconnected(code=1001, message="idle_timeout")
            return
        self._note_outage()
        self.ten_env.log_warn(f"NYO TTS websocket lost: {reason}", category=LOG_CATEGORY_VENDOR)
        vi = ModuleErrorVendorInfo(vendor=VENDOR_NAME, code="connection_lost", message=reason)
        await self.on_disconnected(code=1006, message=reason, vendor_info=vi)
        if req is not None and req.open:
            err = self._rate_limit(ModuleError(
                message=f"connection lost: {reason}",
                module=ModuleType.TTS,
                code=ModuleErrorCode.NON_FATAL_ERROR,
                vendor_info=vi,
            ))
            if req.wire_ids:
                # the segments on this socket are lost; the TEN request goes on (its next chunk reconnects)
                await self._segment_failed(req, err, req.unfinished(), done=True, client=client)
            else:
                await self._finalize_request(TTSAudioEndReason.ERROR, error=err)
        self._schedule_reconnect()

    # ------------------------------------------------------ update_configs
    async def update_configs(self, configs: dict) -> None:
        """cmd `update_configs`: switch voice / language / model between requests.

        The values ride in the websocket URL, so a change takes effect on a fresh socket. The new values are
        validated first (a rejected update changes nothing). Applied right away when idle; during a request
        it is held and applied before the next one, so the running request keeps its route and sample rate.
        A new ``model`` without an explicit ``language`` gets the language the gateway would derive from it.
        The new route is checked against GET /v1/models: a model, language or voice it conclusively does not serve
        rejects the command (the old config stays); an inconclusive answer (gateway unreachable, no TTS model
        listed, no voice catalogue) accepts it.
        """
        assert self.config is not None
        params = configs.get("params", configs) if isinstance(configs, dict) else {}
        if not isinstance(params, dict):
            raise ValueError("update_configs: params must be an object")
        base = self._pending_config or self.config
        cand = base.model_copy(deep=True)
        for name in ("voice", "language", "model"):
            if name in params:
                if not isinstance(params[name], str):
                    raise ValueError(f"update_configs: {name} must be a string")
                setattr(cand, name, params[name])
        if "model" in params and "language" not in params and cand.model != base.model:
            cand.language = language_from_model(cand.model)
        if (cand.voice, cand.language, cand.model) == (base.voice, base.language, base.model):
            return
        cand.validate_config()  # raises before anything is changed
        try:
            verdict = await self._served(cand)
        except NYOTTSConnectError as e:  # 401/403/TLS on the listing: not this command's business; accept
            self.ten_env.log_warn(f"update_configs: /v1/models check inconclusive: {e}")
            verdict = None
        if isinstance(verdict, tuple):
            raise ValueError(f"update_configs rejected ({verdict[0]}): {verdict[1]}")
        self.ten_env.log_info(
            f"update_configs: voice={cand.voice!r} language={cand.language!r} model={cand.model!r}",
            category=LOG_CATEGORY_KEY_POINT,
        )
        if self._req is not None and self._req.open:
            self._pending_config, self._pending_checked = cand, verdict is True
            return
        self._pending_config = None
        await self._apply_config(cand, checked=verdict is True)

    async def _apply_config(self, cand: NYOTTSConfig, checked: bool = False, reconnect: bool = True) -> None:
        assert self.config is not None
        model_changed = cand.model != self.config.model
        for name in ("voice", "language", "model"):
            setattr(self.config, name, getattr(cand, name))
        self._config_gen += 1  # any socket or preflight in flight now belongs to the old config
        if model_changed or not checked:
            # the listing confirmed the new route already, or the next connect checks it
            self._preflight_ok = checked
        self._sample_rate = None  # the new route announces its own rate
        await self._recycle_socket(reconnect)

    async def _recycle_socket(self, reconnect: bool = True) -> None:
        if self.client:
            old, self.client = self.client, None
            self._connected_ev.clear()
            await old.stop()
            await self.on_disconnected(code=0, message="config changed")
        if not reconnect:  # the caller connects (raced against a flush)
            return
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

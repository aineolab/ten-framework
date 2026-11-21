#!/usr/bin/env python3
"""Local NRSP v1 fake server for the TEN addons (and speechd's own tests).

Implements the wire protocol of plan section 3.1 / ``serve/protocol.py`` exactly,
with deterministic fake "models" (canned transcripts, sine-tone audio).

    python3 ten/fake_nrsp_server.py --port 9100          # standalone
    async with FakeNRSPServer() as srv: srv.url           # in-process (pytest)

Endpoints (``Authorization: Bearer <key>``, subprotocol ``nrsp.v1``)
    /v1/asr?model=&language=&sample_rate=
    /v1/tts?model=&voice=&language=

ASR   client -> server : binary PCM16 mono | {config} | {finalize,id} | {close}
      server -> client : {result} {finalize_end} {usage} {error}
TTS   client -> server : {text,request_id,text,end} | {flush,request_id} | {close}
      server -> client : {audio_start,request_id,request_seq,sample_rate}
                         binary [u32 BE request_seq][u32 BE audio_offset_ms][PCM16]
                         {audio_end,request_id,reason,audio_ms} {alignment} {usage} {error}

Fault injection rides on the API key (any key is valid except ``invalid``):
    ``<key>+drop3``     close the socket abruptly after 3 inbound audio frames (first
                        connection with that key only, so reconnect succeeds)
    ``<key>+fatal``     send a fatal error after the first audio frame / text message
    ``<key>+nofinal``   never answer ``finalize`` (exercises finalize_timeout_ms)
    ``<key>+dropfin``   abort the socket on the first ``finalize`` (first connection with
                        that key only, so reconnect succeeds)
    ``<key>+stale3``    after a TTS flush, keep sending 3 frames with the old request_seq
    ``<key>+ttserr``    text containing ERROR_TRIGGER yields a non-fatal engine_error
    ``<key>+rt``        pace TTS audio at real time (default is ~8x faster)
    ``<key>+restart``   first connection with that key: after the first audio frame / text
                        message send fatal ``engine_restarting`` and close 1012 (as the real
                        gateway does on an engine swap); later connections are normal
    ``<key>+idle<N>``   close with fatal ``idle_timeout`` (1001) after N ms with no client
                        message (the real gateway's idle rule, default 300 s there)
Constructor kwargs give the same switches programmatically.
Also: plain-HTTP ``GET /v1/models`` (same auth) lists models (``models=[...]`` overrides);
    ``endpointing_ms`` auto-cut emits final ``utt-N`` with metadata ``{model, finalize_id:
    auto-K, endpoint: auto}`` + ``finalize_end auto-K``; ``ssl_context=``/``--tls-cert`` -> wss.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import re
import ssl
import struct
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

SUBPROTOCOL = "nrsp.v1"
TTS_HDR = struct.Struct(">II")  # [u32 request_seq][u32 audio_offset_ms], big-endian
MAX_ASR_FRAME_BYTES = 64 * 1024
MAX_TTS_TEXT_CHARS = 20_000
INTERIM_HOP_MS = 320
STABLE_AGE_MS = 640
TTS_CHUNK_MS = 85  # one Orpheus coarse frame (2048 samples @ 24 kHz)

ASR_LANGS = {"bcl", "ceb", "fil", "hil", "ilo", "pam", "pag", "tsg", "war", "eng"}
CANNED = {
    "en": "hello world this is a test of the speech gateway",
    "zh": "你好世界 这是 一个 测试",
    "es": "hola mundo esto es una prueba del sistema",
    "fil": "kumusta mundo ito ay isang pagsubok ng sistema",
    "ceb": "kumusta kalibutan kini usa ka pagsulay sa sistema",
    "hil": "kumusta kalibutan ini isa ka pagtilaw sang sistema",
}
KNOWN_VENDORS = {"halohalo", "nvidia", "nyo", "deepgram", "soniox", "elevenlabs", "cartesia", "fw"}


def default_models() -> list[dict]:
    """Model list for GET /v1/models when the server was not given ``models=[...]``."""
    asr = [f"halohalo/asr-{lang}" for lang in sorted(ASR_LANGS | {"en", "zh", "es"})]
    asr += ["nyo/auto", "nvidia/nemotron-3.5-transcribe",
            "fw/whisper-large-v3-pld-fil-norm", "fw/whisper-tiny",
            "fw/flipvox-whisper-base-fil", "fw/whisper-large-v3-pld-hil-norm"]
    tts = [f"halohalo/tts-{lang}" for lang in sorted(ASR_LANGS)] + ["halohalo/tts-speecht5"]
    return ([{"id": m, "object": "model", "kind": "asr"} for m in asr]
            + [{"id": m, "object": "model", "kind": "tts"} for m in tts])


def self_signed_ssl_context(host: str = "127.0.0.1") -> ssl.SSLContext:
    """Server TLS context with a throwaway self-signed certificate (``openssl`` CLI). Clients that verify
    certificates must reject it: this is the "bad cert" case of the transport tests."""
    import subprocess
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp(prefix="fake-nrsp-tls-"))
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-subj", f"/CN={host}", "-keyout", str(d / "key.pem"), "-out", str(d / "cert.pem")],
                   check=True, capture_output=True)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(d / "cert.pem", d / "key.pem")
    return ctx


def pack_tts_frame(seq: int, offset_ms: int, pcm: bytes) -> bytes:
    return TTS_HDR.pack(seq & 0xFFFFFFFF, offset_ms & 0xFFFFFFFF) + pcm


def error_obj(code: str, message: str, fatal: bool = False, vendor: str = "halohalo", **extra: Any) -> dict:
    d = {"type": "error", "code": code, "fatal": fatal,
         "vendor": {"name": vendor, "code": code, "message": message}}
    d.update(extra)
    return d


def _unauthorized() -> Response:
    body = json.dumps(error_obj("unauthorized", "invalid or missing API key", True))
    return Response(401, "Unauthorized", Headers({"Content-Type": "application/json"}), body.encode())


def sine_pcm(sample_rate: int, ms: int, start_sample: int, freq: float = 220.0) -> bytes:
    n = sample_rate * ms // 1000
    return b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * freq * (start_sample + i) / sample_rate)))
        for i in range(n)
    )


@dataclass
class _Opts:
    drop_after: int = 0
    fatal: bool = False
    nofinal: bool = False
    dropfin: bool = False
    stale: int = 0
    ttserr: bool = False
    realtime: bool = False
    restart: bool = False
    idle_ms: int = 0

    @classmethod
    def from_key(cls, key: str) -> "_Opts":
        o = cls()
        for tok in key.split("+")[1:]:
            if m := re.fullmatch(r"drop(\d+)", tok):
                o.drop_after = int(m[1])
            elif m := re.fullmatch(r"stale(\d+)", tok):
                o.stale = int(m[1])
            elif m := re.fullmatch(r"idle(\d+)", tok):
                o.idle_ms = int(m[1])
            elif tok == "restart":
                o.restart = True
            elif tok in ("fatal", "nofinal", "dropfin", "ttserr"):
                setattr(o, tok, True)
            elif tok == "rt":
                o.realtime = True
        return o


@dataclass
class _TTSRequest:
    request_id: str
    seq: int
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    chars: int = 0
    audio_ms: int = 0
    flushed: bool = False
    started: bool = False


class FakeNRSPServer:
    """In-process NRSP v1 server. All state is inspectable for assertions."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, first_audio_delay_ms: int = 15,
                 chunk_interval_ms: int = 10, ms_per_char: int = 60, verbose: bool = False,
                 models: list | None = None, ssl_context: ssl.SSLContext | None = None):
        self.host, self.port = host, port
        self.first_audio_delay_ms = first_audio_delay_ms
        self.chunk_interval_ms = chunk_interval_ms
        self.ms_per_char = ms_per_char
        self.verbose = verbose
        self.models = models          # explicit /v1/models list; also gates websocket models
        self.models_requests = 0      # GET /v1/models hits, for assertions
        self.ssl_context = ssl_context
        self.connections: list[dict[str, Any]] = []  # one record per accepted socket
        self.log: list[tuple[str, dict]] = []        # (direction/kind, payload)
        self._dropped_keys: set[str] = set()
        self._restarted_keys: set[str] = set()
        self._server = None
        self._req_counter = 0

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> "FakeNRSPServer":
        self._server = await serve(
            self._handler, self.host, self.port,
            subprotocols=[SUBPROTOCOL],  # type: ignore[list-item]
            process_request=self._process_request,
            max_size=1 << 20,
            ssl=self.ssl_context,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> "FakeNRSPServer":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    @property
    def url(self) -> str:
        scheme = "wss" if self.ssl_context else "ws"
        return f"{scheme}://{self.host}:{self.port}"

    def received(self, kind: str) -> list[dict]:
        return [p for k, p in self.log if k == f"rx:{kind}"]

    # ------------------------------------------------------------- handshake
    def _process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        u = urlparse(request.path)
        if u.path == "/v1/models" and request.headers.get("Upgrade", "").lower() != "websocket":
            return self._models_http(request)
        if u.path not in ("/v1/asr", "/v1/tts"):
            return conn.respond(404, "unknown endpoint\n")
        auth = request.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if not key or key == "invalid" or key.split("+")[0] == "invalid":
            return _unauthorized()
        return None

    def _models_http(self, request: Request) -> Response:
        self.models_requests += 1
        auth = request.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if not key or key == "invalid" or key.split("+")[0] == "invalid":
            return _unauthorized()
        data = self.models if self.models is not None else default_models()
        body = json.dumps({"object": "list", "data": data, "protocol": SUBPROTOCOL}, ensure_ascii=False)
        return Response(200, "OK", Headers({"Content-Type": "application/json"}), body.encode())

    # --------------------------------------------------------------- session
    async def _handler(self, ws: ServerConnection) -> None:
        u = urlparse(ws.request.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        key = ws.request.headers.get("Authorization", "")[7:]
        rec = {"path": u.path, "query": q, "key": key, "subprotocol": ws.subprotocol,
               "opened_at": time.time(), "frames": 0}
        self.connections.append(rec)
        if ws.subprotocol != SUBPROTOCOL:
            await self._send(ws, error_obj("unsupported_version", "offer subprotocol nrsp.v1", True))
            await ws.close(1002)
            return
        model = q.get("model", "")
        if not self._model_ok(u.path, model):
            await self._send(ws, error_obj("unknown_model", f"unknown model {model!r}", True))
            await ws.close(1008)
            return
        opts = _Opts.from_key(key)
        rec["last_in"] = time.monotonic()
        idle_task = asyncio.create_task(self._idle_watch(ws, rec, opts.idle_ms)) if opts.idle_ms else None
        try:
            if u.path == "/v1/asr":
                await _AsrSession(self, ws, q, opts, rec).run()
            else:
                await _TtsSession(self, ws, q, opts, rec).run()
        except Exception as e:  # ConnectionClosed and friends end the session
            if self.verbose:
                print(f"[fake-nrsp] session ended: {type(e).__name__}: {e}")
        finally:
            if idle_task is not None:
                idle_task.cancel()

    async def _idle_watch(self, ws: ServerConnection, rec: dict, idle_ms: int) -> None:
        with contextlib.suppress(Exception):
            while True:
                await asyncio.sleep(idle_ms / 4000)
                if (time.monotonic() - rec["last_in"]) * 1000 > idle_ms:
                    await self._send(ws, error_obj("idle_timeout", f"no client message for {idle_ms} ms", True))
                    await ws.close(1001)
                    return

    async def _maybe_restart(self, ws: ServerConnection, opts: "_Opts", rec: dict) -> bool:
        """``+restart``: fatal engine_restarting + close 1012, once per key. True if it fired."""
        if not opts.restart or rec["key"] in self._restarted_keys:
            return False
        self._restarted_keys.add(rec["key"])
        await self._send(ws, error_obj("engine_restarting", "engine swap in progress", True))
        await ws.close(1012)
        return True

    def _model_ok(self, path: str, model: str) -> bool:
        if self.models is not None:  # explicit list: websocket model must be in it
            ids = {m if isinstance(m, str) else m.get("id", "") for m in self.models}
            return model in ids
        if not model or "/" not in model and model != "nyo/auto":
            return False
        vendor, _, name = model.partition("/")
        if vendor not in KNOWN_VENDORS or not name:
            return False
        if path == "/v1/asr" and vendor == "halohalo" and name.startswith("asr-"):
            return name[4:] in ASR_LANGS | {"en", "zh", "es"}  # en/zh/es: guarder configs only
        return True

    async def _send(self, ws: ServerConnection, obj: dict) -> None:
        self.log.append(("tx:" + obj["type"], obj))
        await ws.send(json.dumps(obj, ensure_ascii=False))


# ===================================================================== ASR
class _AsrSession:
    def __init__(self, srv: FakeNRSPServer, ws: ServerConnection, q: dict, opts: _Opts, rec: dict):
        self.srv, self.ws, self.opts, self.rec = srv, ws, opts, rec
        self.language = q.get("language", "")
        self.model = q.get("model", "")
        self.sample_rate = int(q.get("sample_rate", 16000))
        self.interim = True
        self.endpointing_ms = 0
        self.total_ms = 0          # audio received on this socket
        self.utt_start_ms = 0
        self.utt_ms = 0
        self.utt_id = 0
        self.auto_id = 0           # auto-endpoint cuts so far (finalize_id auto-<k>)
        self.next_interim_ms = INTERIM_HOP_MS
        self.silence_ms = 0
        self.speech_seen = False
        self.frames = 0
        self.last_usage_ms = 0

    # ---- helpers
    def _lang_key(self) -> str:
        lang = self.language or self.model.rsplit("-", 1)[-1]
        lang = lang.split("-")[0]
        return lang if lang in CANNED else ("en" if lang in ("", "eng", "auto") else "fil")

    def _backend(self, final: bool) -> str:
        return "fake-whisper" if final else "fake-ctc"

    def _words(self, n: int) -> list[str]:
        text = CANNED[self._lang_key()]
        parts = text.split()
        return parts[: max(1, min(n, len(parts)))]

    def _result(self, final: bool) -> dict:
        n_words = max(1, self.utt_ms // 400)
        words = self._words(n_words)
        per = max(1, self.utt_ms // max(1, len(words)))
        wl = [{"word": w, "start_ms": self.utt_start_ms + i * per, "duration_ms": per,
               "stable": final or (self.utt_ms - (i + 1) * per) >= STABLE_AGE_MS}
              for i, w in enumerate(words)]
        return {"type": "result", "id": f"u{self.utt_id}", "text": " ".join(words), "final": final,
                "start_ms": self.utt_start_ms, "duration_ms": self.utt_ms,
                "language": self.language or self._lang_key(), "words": wl,
                "backend": self._backend(final), "metadata": {"model": self.model}}

    async def _emit_final(self, auto: bool = False) -> bool:
        if not self.speech_seen:
            return False
        res = self._result(True)
        if auto:  # D21 server-side endpointing: utt-N id + finalize metadata, then finalize_end
            self.auto_id += 1
            res["id"] = f"utt-{self.utt_id}"
            res["metadata"].update({"finalize_id": f"auto-{self.auto_id}", "endpoint": "auto"})
        await self.srv._send(self.ws, res)
        if auto:
            await self.srv._send(self.ws, {"type": "finalize_end", "id": f"auto-{self.auto_id}"})
        self.utt_id += 1
        self.utt_start_ms = self.total_ms
        self.utt_ms = 0
        self.next_interim_ms = INTERIM_HOP_MS
        self.speech_seen = False
        return True

    async def _usage(self) -> None:
        await self.srv._send(self.ws, {"type": "usage", "audio_ms": self.total_ms, "credits": 0.0})
        self.last_usage_ms = self.total_ms

    # ---- main loop
    async def run(self) -> None:
        async for raw in self.ws:
            self.rec["last_in"] = time.monotonic()
            if isinstance(raw, bytes):
                await self._on_audio(raw)
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self.srv._send(self.ws, error_obj("bad_message", "invalid JSON"))
                continue
            self.srv.log.append(("rx:" + str(msg.get("type")), msg))
            t = msg.get("type")
            if t == "config":
                self.language = msg.get("language") or self.language
                self.interim = bool(msg.get("interim", True))
                self.endpointing_ms = int(msg.get("endpointing_ms") or 0)
                self.sample_rate = int(msg.get("sample_rate") or self.sample_rate)
            elif t == "finalize":
                if self.opts.nofinal:
                    continue
                if self.opts.dropfin and self.rec["key"] not in self.srv._dropped_keys:
                    self.srv._dropped_keys.add(self.rec["key"])
                    self.ws.transport.abort()  # abrupt close, no close frame
                    return
                await self._emit_final()
                await self.srv._send(self.ws, {"type": "finalize_end", "id": msg.get("id")})
            elif t == "close":
                await self._usage()
                await self.ws.close(1000)
                return
            else:
                await self.srv._send(self.ws, error_obj("bad_message", f"unknown type {t!r}"))

    async def _on_audio(self, data: bytes) -> None:
        if len(data) > MAX_ASR_FRAME_BYTES:
            await self.srv._send(self.ws, error_obj("frame_too_large", f"{len(data)} bytes", True))
            await self.ws.close(1009)
            return
        if len(data) % 2:
            await self.srv._send(self.ws, error_obj("bad_message", "PCM16 frame has odd length"))
            return
        self.frames += 1
        self.rec["frames"] = self.frames
        self.srv.log.append(("rx:audio", {"bytes": len(data)}))
        ms = len(data) * 1000 // (self.sample_rate * 2)
        self.total_ms += ms
        silent = not any(data)
        if silent:
            self.silence_ms += ms
        else:
            self.silence_ms = 0
            self.speech_seen = True
            self.utt_ms += ms
        if self.frames == 1 and await self.srv._maybe_restart(self.ws, self.opts, self.rec):
            return
        if self.opts.fatal and self.frames == 1:
            await self.srv._send(self.ws, error_obj("engine_error", "injected fatal error", True, vendor="fake-vendor"))
            await self.ws.close(1011)
            return
        if self.opts.drop_after and self.frames == self.opts.drop_after \
                and self.rec["key"] not in self.srv._dropped_keys:
            self.srv._dropped_keys.add(self.rec["key"])
            self.ws.transport.abort()  # abrupt close, no close frame
            return
        if self.interim and self.speech_seen and self.utt_ms >= self.next_interim_ms and not silent:
            self.next_interim_ms += INTERIM_HOP_MS
            await self.srv._send(self.ws, self._result(False))
        if self.endpointing_ms and self.speech_seen and self.silence_ms >= self.endpointing_ms:
            await self._emit_final(auto=True)
        if self.total_ms - self.last_usage_ms >= 10_000:
            await self._usage()


# ===================================================================== TTS
class _TtsSession:
    def __init__(self, srv: FakeNRSPServer, ws: ServerConnection, q: dict, opts: _Opts, rec: dict):
        self.srv, self.ws, self.opts, self.rec = srv, ws, opts, rec
        model = q.get("model", "")
        self.sample_rate = 16000 if re.search(r"speecht5|fallback", model) else 24000
        self.requests: dict[str, _TTSRequest] = {}
        self.dead: set[str] = set()  # flushed / finished request ids
        self.write_lock = asyncio.Lock()

    async def run(self) -> None:
        try:
            async for raw in self.ws:
                self.rec["last_in"] = time.monotonic()
                if isinstance(raw, bytes):
                    await self.srv._send(self.ws, error_obj("bad_message", "TTS takes no binary input"))
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self.srv._send(self.ws, error_obj("bad_message", "invalid JSON"))
                    continue
                self.srv.log.append(("rx:" + str(msg.get("type")), msg))
                t = msg.get("type")
                if t == "text":
                    await self._on_text(msg)
                elif t == "flush":
                    await self._on_flush(msg)
                elif t == "close":
                    await self.ws.close(1000)
                    return
                else:
                    await self.srv._send(self.ws, error_obj("bad_message", f"unknown type {t!r}"))
        finally:
            for r in self.requests.values():
                if r.task:
                    r.task.cancel()

    async def _on_text(self, msg: dict) -> None:
        rid = msg.get("request_id")
        if not isinstance(rid, str) or not rid or not isinstance(msg.get("text", ""), str):
            await self.srv._send(self.ws, error_obj("bad_message", "text needs request_id and text"))
            return
        if rid in self.dead:
            await self.srv._send(self.ws, error_obj("stale_request", f"request {rid} is finished or flushed",
                                                    request_id=rid))
            return
        if not self.requests and await self.srv._maybe_restart(self.ws, self.opts, self.rec):
            return
        if self.opts.fatal and not self.requests:
            await self.srv._send(self.ws, error_obj("engine_error", "injected fatal error", True,
                                                    vendor="fake-vendor", request_id=rid))
            await self.ws.close(1011)
            return
        req = self.requests.get(rid)
        if req is None:
            self.srv._req_counter += 1
            req = _TTSRequest(rid, self.srv._req_counter)
            self.requests[rid] = req
            req.task = asyncio.create_task(self._synth(req))
        req.chars += len(msg.get("text", ""))
        if req.chars > MAX_TTS_TEXT_CHARS:
            await self.srv._send(self.ws, error_obj("text_too_long", "request text too long", request_id=rid))
            return
        req.queue.put_nowait((msg.get("text", ""), bool(msg.get("end", False))))

    async def _send(self, obj: dict) -> None:
        async with self.write_lock:
            await self.srv._send(self.ws, obj)

    async def _send_frame(self, req: _TTSRequest, seq: int, offset_ms: int, pcm: bytes) -> None:
        async with self.write_lock:
            self.srv.log.append(("tx:audio", {"request_id": req.request_id, "seq": seq, "offset_ms": offset_ms}))
            await self.ws.send(pack_tts_frame(seq, offset_ms, pcm))

    async def _synth(self, req: _TTSRequest) -> None:
        sr = self.sample_rate
        samples = 0
        try:
            while True:
                text, end = await req.queue.get()
                if self.opts.ttserr and "ERROR_TRIGGER" in text:
                    await self._send(error_obj("engine_error", "injected engine error", request_id=req.request_id))
                    self.dead.add(req.request_id)
                    return
                ms_total = len(text.strip()) * self.srv.ms_per_char
                if ms_total and not req.started:
                    req.started = True
                    await asyncio.sleep(self.srv.first_audio_delay_ms / 1000)
                    await self._send({"type": "audio_start", "request_id": req.request_id,
                                      "request_seq": req.seq, "sample_rate": sr})
                if ms_total:
                    start_ms = req.audio_ms
                    words = text.split()
                    emitted = 0
                    while emitted < ms_total:
                        step = min(TTS_CHUNK_MS, ms_total - emitted)
                        pcm = sine_pcm(sr, step, samples)
                        samples += len(pcm) // 2
                        await self._send_frame(req, req.seq, req.audio_ms, pcm)
                        req.audio_ms += step
                        emitted += step
                        pace = step if self.opts.realtime else self.srv.chunk_interval_ms
                        await asyncio.sleep(pace / 1000)
                    if words:
                        per = max(1, ms_total // len(words))
                        await self._send({"type": "alignment", "request_id": req.request_id, "text": text,
                                          "start_ms": start_ms, "duration_ms": ms_total,
                                          "words": [{"word": w, "start_ms": start_ms + i * per, "duration_ms": per}
                                                    for i, w in enumerate(words)]})
                if end:
                    await self._send({"type": "audio_end", "request_id": req.request_id, "reason": "end",
                                      "audio_ms": req.audio_ms})
                    await self._send({"type": "usage", "request_id": req.request_id,
                                      "audio_ms": req.audio_ms, "credits": 0.0})
                    self.dead.add(req.request_id)
                    return
        except asyncio.CancelledError:
            pass

    async def _on_flush(self, msg: dict) -> None:
        rid = msg.get("request_id")
        req = self.requests.get(rid)
        if req is None or req.flushed:
            return
        req.flushed = True
        if req.task:
            req.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await req.task
        self.dead.add(rid)
        await self._send({"type": "audio_end", "request_id": rid, "reason": "flushed", "audio_ms": req.audio_ms})
        for i in range(self.opts.stale):  # misbehaving / in-flight frames with the old seq
            await self._send_frame(req, req.seq, req.audio_ms + i, sine_pcm(self.sample_rate, 20, 0))


# ===================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tls-cert", metavar="PATH", help="PEM certificate chain (enables wss://)")
    ap.add_argument("--tls-key", metavar="PATH", help="PEM private key (default: from --tls-cert)")
    a = ap.parse_args()

    async def _run() -> None:
        ssl_context = None
        if a.tls_cert:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(a.tls_cert, a.tls_key)
        srv = await FakeNRSPServer(a.host, a.port, verbose=a.verbose, ssl_context=ssl_context).start()
        print(f"fake NRSP v1 server on {srv.url}  (key 'invalid' -> 401)", flush=True)
        await asyncio.Future()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":
    main()

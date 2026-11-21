"""QA round 4 fixes: oversized text increments are split, the gateway's own cancel is an ERROR, the end of
reconnect_window_ms while idle is not FATAL, and the handshake rejection body's code is used."""
import asyncio
import json
from unittest.mock import MagicMock

from ten_ai_base.message import ModuleErrorCode, TTSAudioEndReason
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response
from websockets.protocol import State

from nyo_tts_python.const import TEXT_FRAME_MAX_BYTES
from nyo_tts_python.extension import _Request
from nyo_tts_python.tts_client import NYOTTSClient, split_text

from .helpers import make_config, make_extension, run, text_input, wait_for

FATAL = int(ModuleErrorCode.FATAL_ERROR.value)
NON_FATAL = int(ModuleErrorCode.NON_FATAL_ERROR.value)


class _WS:
    state = State.OPEN

    def __init__(self):
        self.sent = []

    async def send(self, m):
        self.sent.append(m)


def test_split_text_is_lossless_and_bounded():
    t = "kumusta ka " * 1000 + "x" * 5000
    parts = split_text(t, 700)
    assert "".join(parts) == t and all(len(p) <= 700 for p in parts) and len(parts) > 1
    assert split_text("short", 700) == ["short"]


def test_oversized_text_increment_goes_out_as_several_text_messages():
    async def go():
        c = NYOTTSClient(make_config(), MagicMock())
        c.websocket = _WS()
        text = "ñ😀 " * 9000                                   # ~ 9000 * 20 encoded bytes: far over 64 KiB
        await c.send_text("r", text, True)
        msgs = [json.loads(m) for m in c.websocket.sent]
        assert len(msgs) > 1 and all(len(m) <= TEXT_FRAME_MAX_BYTES for m in c.websocket.sent)
        assert "".join(m["text"] for m in msgs) == text
        assert [m["end"] for m in msgs] == [False] * (len(msgs) - 1) + [True]
        assert {m["request_id"] for m in msgs} == {"r"}
        c.websocket.sent.clear()
        await c.send_text("q", "hello", False)                  # small: one message, unchanged
        assert [json.loads(m) for m in c.websocket.sent] == [
            {"type": "text", "request_id": "q", "text": "hello", "end": False}]

    run(go())


def test_gateway_cancel_is_error_not_interrupted():
    async def go():
        for reason in ("interrupted", "flushed"):
            ext, rec = make_extension()
            ext._req = _Request("r", wire_ids=["r"], seg_text=True)
            ext._req.end_sent = True
            await ext._on_audio_end("r", reason, 0, {})
            # round 8: the segment is lost (NON_FATAL server_cancelled); the request had text_input_end and no audio
            assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.ERROR
            err = rec.of("error")[-1]["error"]
            assert int(err.code) == NON_FATAL and err.vendor_info.code == "server_cancelled"

    run(go())


def test_window_end_while_idle_is_not_fatal_next_request_is():
    async def go():
        ext, rec = make_extension("ws://127.0.0.1:1", reconnect_window_ms=300)   # nothing listens
        ext._reconnect_delay = lambda attempt: 0.01
        ext._schedule_reconnect()
        await asyncio.sleep(0.05)
        await wait_for(lambda: ext._reconnect_task is None, timeout=5, what="loop end")
        assert rec.of("error") == [] and not ext._fatal          # idle: no FATAL, retrying stopped
        await asyncio.wait_for(ext.request_tts(text_input("a", "kumusta", end=True)), 5)
        sent = [e["error"] for e in rec.of("error")] + [f["error"] for f in rec.of("finish") if f["error"]]
        assert [(int(e.code), e.vendor_info.code) for e in sent] == [(FATAL, "connect_failed")]
        assert rec.of("audio_end")[-1]["request_id"] == "a"
        assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.ERROR
        ext._stopped = True

    run(go())


def test_window_end_while_idle_then_gateway_back_recovers():
    async def go():
        from fake_nrsp_server import FakeNRSPServer
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url, reconnect_window_ms=300)
            import time
            ext._outage_start = time.monotonic() - 10                 # an outage older than the window, idle
            await ext.request_tts(text_input("a", "kumusta", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5)
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END and not ext._fatal
            ext._stopped = True
            await ext.client.stop()

    run(go())


def _rejecting(body: dict, status: int, models):
    async def reject(conn, request):
        if request.path.startswith("/v1/models"):
            return Response(200, "OK", Headers({"Content-Type": "application/json"}),
                            json.dumps({"object": "list", "data": models}).encode())
        return Response(status, "Bad", Headers({"Content-Type": "application/json"}), json.dumps(body).encode())

    return serve(lambda ws: None, "127.0.0.1", 0, process_request=reject)


def test_handshake_body_unknown_voice_is_fatal_at_once():
    body = {"type": "error", "code": "unknown_voice", "fatal": True, "message": "unknown voice; served: default"}
    models = [{"id": "halohalo/tts-hil", "kind": "tts", "language": "hil", "voices": []}]   # "*": listing can't tell

    async def go():
        async with _rejecting(body, 400, models) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            ext, rec = make_extension(url, model="halohalo/tts-hil", voice="x")
            await ext.request_tts(text_input("r", "kumusta", end=True))
            sent = [e["error"] for e in rec.of("error")] + [f["error"] for f in rec.of("finish") if f["error"]]
            assert [(int(e.code), e.vendor_info.code) for e in sent] == [(FATAL, "unknown_voice")]
            assert ext._fatal and rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.ERROR
            ext._stopped = True

    run(go())


def test_handshake_body_unknown_model_with_inconclusive_listing_is_retried_with_that_code():
    body = {"type": "error", "code": "unknown_model", "fatal": True, "message": "unknown model"}

    async def go():
        async with _rejecting(body, 404, []) as server:               # empty listing: inconclusive
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            ext, rec = make_extension(url, reconnect_window_ms=120000)
            ext._preflight_ok = True
            ext._reconnect_delay = lambda attempt: 0.05
            try:
                await ext._ensure_connected()
            except Exception as e:  # noqa: BLE001
                assert getattr(e, "code", None) == "model_not_found"
            assert not ext._fatal
            errs = [(int(e["error"].code), e["error"].vendor_info.code) for e in rec.of("error")]
            assert errs == [(NON_FATAL, "model_not_found")]
            ext._stopped = True
            ext._cancel_reconnect_task()

    run(go())

"""QA probes for nyo_tts_python. Each test asserts the CORRECT behaviour; a failure demonstrates a defect.

Real-gateway tests need serve/ running locally (mock engines) at NYO_QA_GW, e.g.
SAPIN_AUTH=off SAPIN_MIN_AVAILABLE_GIB=0 SAPIN_IDLE_TIMEOUT_S=3 uvicorn serve.gateway:create_app --factory --port 9137
"""
import asyncio
import json
import os
import socket
from urllib.parse import urlparse

import pytest
from ten_ai_base.message import TTSAudioEndReason

from fake_nrsp_server import FakeNRSPServer
from .helpers import harness, make_extension, run, text_input, wait_for

GW = os.environ.get("NYO_QA_GW", "ws://127.0.0.1:9137")


def _gw_up() -> bool:
    u = urlparse(GW)
    try:
        socket.create_connection((u.hostname, u.port), 0.3).close()
        return True
    except OSError:
        return False


real_gw = pytest.mark.skipif(not _gw_up(), reason="real gateway not running")


@real_gw
def test_qa_real_gateway_idle_timeout_must_not_kill_extension():
    """Gateway closes an idle socket with fatal idle_timeout (nrsp-v1 s.1). The next request must still work."""
    async def go():
        ext, rec = make_extension(GW, model="halohalo/tts-fil")
        try:
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "Kumusta ka.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="r1 end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            await asyncio.sleep(5)  # > SAPIN_IDLE_TIMEOUT_S=3: the gateway sends fatal idle_timeout and closes 1001
            errs = [e["error"] for e in rec.of("error")]
            rec.events.clear()
            await ext.request_tts(text_input("r2", "Mabuhay.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="r2 end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END, (
                f"extension dead after idle_timeout: fatal={ext._fatal} earlier errors={[(e.code, e.message) for e in errs]}")
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())


def test_qa_engine_restarting_fatal_must_not_kill_extension():
    """Gateway engine swap sends fatal engine_restarting + close 1012, documented as 'reconnect'."""
    async def go():
        async with harness() as (srv, ext, rec):
            await ext._ensure_connected()
            client = ext.client
            await client._on_text(json.dumps({"type": "error", "code": "engine_restarting", "fatal": True,
                                                  "message": "swap", "vendor": {"name": "halohalo",
                                                  "code": "engine_restarting", "message": "swap"}}))
            if client.websocket is not None:
                await client.websocket.close(1012)
            await asyncio.sleep(0.2)
            rec.events.clear()
            await ext.request_tts(text_input("r2", "after swap", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5)
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END, f"fatal={ext._fatal}"

    run(go())


def test_qa_flush_while_connecting_must_not_send_text_afterwards():
    """tts_flush arrives while request_tts is still awaiting the (re)connect: text must not reach the server."""
    async def go():
        async with harness() as (srv, ext, rec):
            gate, real_preflight = asyncio.Event(), ext._preflight

            async def slow_preflight():  # hold the connect open deterministically
                await gate.wait()
                await real_preflight()

            ext._preflight = slow_preflight
            task = asyncio.create_task(ext.request_tts(text_input("r1", "first sentence ")))
            await wait_for(lambda: "connecting" in rec.names(), what="connecting")
            await ext.cancel_tts()  # base on_data(tts_flush) runs concurrently with the queue loop
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.INTERRUPTED
            gate.set()
            await task
            await asyncio.sleep(0.2)
            sent = [m["request_id"] for m in srv.received("text")]
            assert "r1" not in sent, f"flushed request r1 was sent to the server after its flush: {sent}"
            assert rec.of("error") == [] and [f["error"] for f in rec.of("finish")] in ([], [None])
            rec.events.clear()
            await ext.request_tts(text_input("r2", "next ", end=True))  # the socket opened meanwhile is usable
            await wait_for(lambda: rec.of("audio_end"))
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END

    run(go())


@real_gw
def test_qa_real_gateway_flush_while_connecting_stalls_next_request():
    """Same race against the real FIFO gateway: the zombie non-final request blocks the next one."""
    async def go():
        ext, rec = make_extension(GW, model="halohalo/tts-fil")
        gate, real_preflight = asyncio.Event(), ext._preflight

        async def slow_preflight():
            await gate.wait()
            await real_preflight()

        ext._preflight = slow_preflight
        try:
            task = asyncio.create_task(ext.request_tts(text_input("r1", "Unang pangungusap ")))
            await wait_for(lambda: "connecting" in rec.names(), what="connecting")
            await ext.cancel_tts()
            gate.set()
            await task
            rec.events.clear()
            t0 = asyncio.get_event_loop().time()
            await ext.request_tts(text_input("r2", "Mabuhay.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=15, what="r2 end")
            dt = asyncio.get_event_loop().time() - t0
            errs = [(e["request_id"], e["error"].message) for e in rec.of("error")]
            assert dt < 2, f"r2 took {dt:.1f} s, reason={rec.of('audio_end')[0]['reason']}, ttfb={rec.of('ttfb')}, errors={errs}"
        finally:
            ext._stopped = True
            if ext.client:
                await ext.client.stop()

    run(go())


def test_qa_update_configs_mid_request_keeps_frame_rate():
    """update_configs during a 16 kHz request must not re-tag that request's remaining frames as 24 kHz."""
    async def go():
        async with harness(model="halohalo/tts-speecht5", server_kw={"chunk_interval_ms": 30}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "x" * 40, end=True))
            await wait_for(lambda: rec.of("audio"), what="first audio")
            await ext.update_configs({"params": {"voice": "other"}})
            await wait_for(lambda: rec.of("audio_end"), what="end")
            rates = {a["rate"] for a in rec.of("audio")}
            assert rates == {16000}, f"frames of one 16 kHz request tagged with rates {rates}"

    run(go())


def test_qa_event_interval_starts_at_audio_start_not_request_start():
    """request_event_interval_ms = tts_audio_start -> tts_audio_end (skill common.md); must exclude TTFB."""
    async def go():
        async with harness(server_kw={"first_audio_delay_ms": 400, "ms_per_char": 1}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "abcdefghij", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5)
            ttfb = rec.of("ttfb")[0]["ms"]
            interval = rec.of("audio_end")[0]["interval_ms"]
            assert interval < ttfb, f"interval {interval} ms includes TTFB {ttfb} ms"

    run(go())


@real_gw
def test_qa_real_gateway_basic_flush_next():
    async def go():
        ext, rec = make_extension(GW, model="halohalo/tts-fil")
        try:
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "Ito ay isang mahabang pangungusap para sa pagsubok ng daloy. " * 4))
            await wait_for(lambda: rec.of("audio"), timeout=5)
            await ext.cancel_tts()
            n_after = len(rec.of("audio"))
            await ext.request_tts(text_input("b", "Mabuhay kayong lahat.", end=True))
            await wait_for(lambda: len(rec.of("audio_end")) == 2, timeout=5)
            ends = rec.of("audio_end")
            assert [e["reason"] for e in ends] == [TTSAudioEndReason.INTERRUPTED, TTSAudioEndReason.REQUEST_END]
            ia = rec.names().index("audio_end")
            ib = [i for i, (n, kw) in enumerate(rec.events) if n == "audio_start" and kw["request_id"] == "b"][0]
            assert all(n != "audio" for n, _ in rec.events[ia + 1:ib]), "audio leaked between flush and b"
            b_bytes = sum(a["n"] for a in rec.of("audio")[n_after:])
            assert ends[1]["duration_ms"] == b_bytes * 1000 // (24000 * 2)
            assert [e for e in rec.of("error")] == []
        finally:
            ext._stopped = True
            if ext.client:
                await ext.client.stop()

    run(go())


@real_gw
def test_qa_real_gateway_engine_swap_mid_request_recovers():
    """POST /v1/admin/swap kills live TTS sessions with fatal engine_restarting + 1012; the next request must work."""
    import urllib.request

    async def go():
        ext, rec = make_extension(GW, model="halohalo/tts-fil")
        ext._reconnect_delay = lambda attempt: 0.05
        try:
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "Ito ay isang mahabang pangungusap para sa pagsubok. " * 6, end=True))
            await wait_for(lambda: rec.of("audio"), timeout=5)
            u = urlparse(GW)
            req = urllib.request.Request(f"http://{u.hostname}:{u.port}/v1/admin/swap?engine=tts&spec=mock",
                                         method="POST")
            await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=20).read())
            await wait_for(lambda: rec.of("audio_end"), timeout=10, what="a ends")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.ERROR
            assert int(rec.of("finish")[0]["error"].code) == 1000, rec.of("finish")[0]["error"]
            assert not ext._fatal
            rec.events.clear()
            await ext.request_tts(text_input("b", "Mabuhay.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=20, what="b ends")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert all(int(e["error"].code) != -1000 for e in rec.of("error"))
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

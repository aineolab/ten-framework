"""Regression tests: reconnectable server errors, idle close, time-based reconnect budget, Retry-After,
requests during an outage, flush while connecting, TTFB/interval origins, update_configs semantics."""
import asyncio
import json
import socket
import time
from unittest.mock import MagicMock

from ten_ai_base.message import ModuleErrorCode, TTSAudioEndReason
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from fake_nrsp_server import FakeNRSPServer
from nyo_tts_python import const
from nyo_tts_python.tts_client import NYOTTSClient, NYOTTSConnectError

from .helpers import harness, make_config, make_extension, run, text_input, wait_for

FATAL = int(ModuleErrorCode.FATAL_ERROR.value)
NON_FATAL = int(ModuleErrorCode.NON_FATAL_ERROR.value)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _all_errors(rec):
    return [e["error"] for e in rec.of("error")] + [f["error"] for f in rec.of("finish") if f["error"]]


def _srv_error(code, fatal=True):
    return json.dumps({"type": "error", "code": code, "fatal": fatal, "message": code,
                       "vendor": {"name": "halohalo", "code": code, "message": code}})


# ------------------------------------------------------------ error classes
def test_engine_restarting_ends_the_request_non_fatal_and_reconnects():
    async def go():
        async with harness(key="k+restart") as (srv, ext, rec):
            ext._reconnect_delay = lambda attempt: 0.01
            await ext.request_tts(text_input("r1", "kumusta ", end=True))
            await wait_for(lambda: rec.of("audio_end"), what="r1 end")
            # round 8: the segment is lost (one NON_FATAL); REQUEST_END if r1 produced any audio, else ERROR
            want = TTSAudioEndReason.REQUEST_END if rec.of("audio") else TTSAudioEndReason.ERROR
            assert rec.of("audio_end")[0]["reason"] == want
            errs = _all_errors(rec)
            assert len(errs) == 1
            err = errs[0]
            assert int(err.code) == NON_FATAL and err.vendor_info.code == "engine_restarting"
            assert not ext._fatal
            await wait_for(lambda: len(srv.connections) == 2 and ext.client and ext.client.is_connected(),
                           what="background reconnect")
            rec.events.clear()
            await ext.request_tts(text_input("r2", "mabuhay ", end=True))
            await wait_for(lambda: rec.of("audio_end"), what="r2 end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert all(int(e.code) != FATAL for e in _all_errors(rec))

    run(go())


def test_reconnectable_fatal_codes_while_idle_emit_no_error():
    async def go():
        for code in sorted(const.RECONNECTABLE_CODES):
            async with harness() as (srv, ext, rec):
                ext._reconnect_delay = lambda attempt: 0.01
                await ext._ensure_connected()
                client = ext.client
                await client._on_text(_srv_error(code))
                await client.websocket.close(1011)
                await wait_for(lambda: len(srv.connections) == 2 and ext.client is not client
                               and ext.client.is_connected(), what=f"{code}: reconnect")
                assert rec.of("error") == [] and not ext._fatal, code

    run(go())


def test_other_fatal_codes_stay_fatal():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.client._on_text(_srv_error("frame_too_large"))
            assert ext._fatal and [int(e.code) for e in _all_errors(rec)] == [FATAL]

    run(go())


def test_idle_timeout_is_silent_and_reconnects_lazily():
    async def go():
        async with harness(key="k+idle200") as (srv, ext, rec):
            await ext._ensure_connected()
            await wait_for(lambda: ext.client is not None and not ext.client.is_connected(), what="idle close")
            await asyncio.sleep(0.3)
            assert len(srv.connections) == 1                      # lazy: no background reconnect
            assert rec.of("error") == [] and not ext._fatal
            assert rec.of("disconnected")[-1]["message"] == "idle_timeout"
            await ext.request_tts(text_input("r", "kumusta ", end=True))
            await wait_for(lambda: rec.of("audio_end"), what="end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert len(srv.connections) == 2 and rec.of("error") == []

    run(go())


def test_idle_timeout_during_a_request_is_handled_like_connection_lost():
    async def go():
        # the fake counts only client messages, so a long real-time request goes idle on the wire
        async with harness(key="k+idle150+rt") as (srv, ext, rec):
            ext._reconnect_delay = lambda attempt: 0.01
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "x" * 40, end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="end")
            # round 8: the segment is lost (one NON_FATAL connection_lost); it had audio -> REQUEST_END
            assert rec.of("audio") and rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            errs = _all_errors(rec)
            assert len(errs) == 1
            err = errs[0]
            assert int(err.code) == NON_FATAL and err.vendor_info.code == "connection_lost"
            await wait_for(lambda: len(srv.connections) == 2, what="background reconnect")
            assert not ext._fatal

    run(go())


# ------------------------------------------------------- reconnect budget
def test_outage_longer_than_five_attempts_but_inside_the_window_recovers():
    async def go():
        port = _free_port()
        ext, rec = make_extension(f"ws://127.0.0.1:{port}", reconnect_window_ms=10000)
        ext._reconnect_delay = lambda attempt: 0.02
        ext._schedule_reconnect()
        await asyncio.sleep(0.4)                                  # ~15 refused attempts
        async with FakeNRSPServer(port=port):
            await wait_for(lambda: ext.client is not None and ext.client.is_connected(), timeout=5)
            assert rec.of("error") == [] and not ext._fatal
            await ext.client.stop()

    run(go())


def test_retry_after_of_a_503_is_honoured():
    async def reject(conn, request):
        return Response(503, "Service Unavailable", Headers({"Retry-After": "1"}), b"{}")

    async def go():
        async with serve(lambda ws: None, "127.0.0.1", 0, process_request=reject) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            client = NYOTTSClient(make_config(url), MagicMock())
            try:
                await client.start()
                raise AssertionError("connect should fail")
            except NYOTTSConnectError as e:
                assert e.status_code == 503 and e.retry_after == 1.0
            ext, rec = make_extension(url)
            ext._reconnect_delay = lambda attempt: 0.01
            calls = []
            real = ext._ensure_connected

            async def counted():
                calls.append(time.monotonic())
                await real()

            ext._ensure_connected = counted
            ext._preflight_ok = True
            ext._retry_after = 0.3                                # as left by a rejected connect
            t0 = time.monotonic()
            ext._schedule_reconnect()
            await wait_for(lambda: calls, what="first retry")
            assert calls[0] - t0 >= 0.29
            ext._stopped = True
            ext._cancel_reconnect_task()

    run(go())


def test_connect_timeout_covers_the_gateway_swap_queue():
    assert const.CONNECT_TIMEOUT_S >= 15 + 5
    assert make_config().reconnect_window_ms == 120000


def test_request_during_an_outage_waits_for_the_reconnect():
    async def go():
        port = _free_port()
        ext, rec = make_extension(f"ws://127.0.0.1:{port}", request_timeout_ms=5000)
        ext._reconnect_delay = lambda attempt: 0.05
        task = asyncio.create_task(ext.request_tts(text_input("r", "kumusta ", end=True)))
        await asyncio.sleep(0.3)
        assert not task.done() and rec.of("audio_end") == []      # waiting, not failed
        async with FakeNRSPServer(port=port):
            await task
            await wait_for(lambda: rec.of("audio_end"), what="end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert rec.of("error") == []
            ext._stopped = True
            await ext.client.stop()

    run(go())


def test_request_during_an_outage_fails_non_fatal_after_request_timeout():
    async def go():
        ext, rec = make_extension("ws://127.0.0.1:1", request_timeout_ms=300)
        ext._reconnect_delay = lambda attempt: 0.02
        t0 = time.monotonic()
        await asyncio.wait_for(ext.request_tts(text_input("r", "kumusta ", end=True)), 3)
        assert 0.25 <= time.monotonic() - t0 < 2
        assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.ERROR
        assert int(rec.of("finish")[0]["error"].code) == NON_FATAL and not ext._fatal
        ext._stopped = True
        ext._cancel_reconnect_task()

    run(go())


def test_flush_while_waiting_out_an_outage_sends_nothing_and_reports_nothing():
    async def go():
        port = _free_port()
        ext, rec = make_extension(f"ws://127.0.0.1:{port}", request_timeout_ms=5000)
        ext._reconnect_delay = lambda attempt: 0.02
        task = asyncio.create_task(ext.request_tts(text_input("r", "kumusta ")))
        await asyncio.sleep(0.2)
        await ext.cancel_tts()
        await asyncio.wait_for(task, 1)                           # released by the flush, no hang
        async with FakeNRSPServer(port=port) as srv:
            await wait_for(lambda: ext.client is not None and ext.client.is_connected(), timeout=5)
            await asyncio.sleep(0.1)
            assert srv.received("text") == []
            assert rec.of("error") == [] and [f["error"] for f in rec.of("finish")] == []
            assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.INTERRUPTED]
            ext._stopped = True
            await ext.client.stop()

    run(go())


# ------------------------------------------------------------- metrics
def test_ttfb_excludes_connect_time_and_interval_starts_at_audio_start():
    async def go():
        async with harness(server_kw={"first_audio_delay_ms": 30, "chunk_interval_ms": 20}) as (srv, ext, rec):
            real = ext._preflight

            async def slow():
                await asyncio.sleep(0.3)
                await real()

            ext._preflight = slow
            await ext.request_tts(text_input("r", "abcdefghij", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5)
            assert rec.of("ttfb")[0]["ms"] < 250                  # 300 ms of connecting is not TTFB
            end = rec.of("audio_end")[0]
            n_chunks = len(rec.of("audio"))
            # interval ~ streaming time (chunks x 20 ms pacing), far below TTFB + connect
            assert 0 < end["interval_ms"] < 20 * n_chunks + 200

    run(go())


def test_no_audio_request_reports_zero_interval():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext.request_tts(text_input("r", "  ", end=True))
            assert rec.of("audio_end")[0]["interval_ms"] == 0

    run(go())


# --------------------------------------------------------- update_configs
def test_update_configs_mid_request_is_applied_before_the_next_request():
    async def go():
        async with harness(model="halohalo/tts-ceb", server_kw={"chunk_interval_ms": 30}) as (srv, ext, rec):
            await ext.request_tts(text_input("a", "x" * 30, end=True))
            await wait_for(lambda: rec.of("audio"), what="audio")
            await ext.update_configs({"params": {"voice": "ana"}})
            assert ext.config.voice == "" and len(srv.connections) == 1   # running request untouched
            await wait_for(lambda: rec.of("audio_end"))
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            await ext.request_tts(text_input("b", "two ", end=True))
            await wait_for(lambda: len(rec.of("audio_end")) == 2)
            assert ext.config.voice == "ana" and srv.connections[-1]["query"]["voice"] == "ana"

    run(go())


def test_rejected_update_changes_nothing():
    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await ext._ensure_connected()
            for bad in ({"model": "noslash"}, {"voice": 7}, {"params": "x"}):
                try:
                    await ext.update_configs(bad)
                    raise AssertionError(f"{bad} accepted")
                except ValueError:
                    pass
            assert (ext.config.model, ext.config.voice, ext.config.language) == ("halohalo/tts-ceb", "", "ceb")
            assert len(srv.connections) == 1

    run(go())


def test_model_change_derives_language_resets_preflight_and_keeps_explicit_language():
    async def go():
        async with harness(model="halohalo/tts-fil", language="fil") as (srv, ext, rec):
            await ext._ensure_connected()
            n_models = srv.models_requests
            await ext.update_configs({"params": {"model": "halohalo/tts-ceb"}})
            assert ext.config.language == "ceb" and srv.connections[-1]["query"]["language"] == "ceb"
            assert srv.models_requests == n_models + 1            # the new model was checked
            await ext.update_configs({"params": {"model": "halohalo/tts-war", "language": "fil"}})
            assert ext.config.language == "fil" and srv.connections[-1]["query"]["model"] == "halohalo/tts-war"

    run(go())

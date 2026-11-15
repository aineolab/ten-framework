"""Transport policy (plan 4.6) and the /v1/models preflight for nyo_tts_python."""
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from ten_ai_base.message import ModuleErrorCode

from fake_nrsp_server import FakeNRSPServer, self_signed_ssl_context
from nyo_tts_python import transport
from nyo_tts_python.tts_client import NYOTTSConnectError

from .helpers import make_config, make_extension, run, text_input, wait_for

FATAL = int(ModuleErrorCode.FATAL_ERROR.value)


@contextlib.asynccontextmanager
async def tls_harness(**params):
    async with FakeNRSPServer(ssl_context=self_signed_ssl_context()) as srv:
        ext, rec = make_extension(srv.url, **params)
        try:
            yield srv, ext, rec
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()


async def _connect_expect_fatal(ext):
    with pytest.raises(NYOTTSConnectError) as ei:
        await ext._ensure_connected()
    return ei.value


@pytest.mark.parametrize("url", ["ws://speech-host.local:9100", "ws://example.com"])
def test_ws_to_remote_host_is_refused(url):
    with pytest.raises(transport.TransportError) as ei:
        make_config(url)
    assert ei.value.code == "insecure_url"


def test_allow_insecure_and_loopback_are_accepted():
    make_config("ws://speech-host.local:9100", allow_insecure=True)
    make_config("ws://localhost:9200")
    make_config("wss://gateway.example.net")


def test_insecure_url_is_a_fatal_module_error_at_init():
    from ten_ai_base.tts2 import AsyncTTS2BaseExtension

    async def go():
        ext, rec = make_extension()
        env = MagicMock()
        env.get_property_to_json = AsyncMock(return_value=(
            '{"params": {"api_key": "k", "base_url": "ws://speech-host.local:9100"}}', None))
        orig = AsyncTTS2BaseExtension.on_init
        AsyncTTS2BaseExtension.on_init = AsyncMock()
        try:
            ext.ten_env = env
            await ext.on_init(env)
        finally:
            AsyncTTS2BaseExtension.on_init = orig
        err = rec.of("error")[0]["error"]
        assert err.code == FATAL and err.vendor_info.code == "insecure_url"

    run(go())


def test_self_signed_cert_is_fatal_tls_error():
    async def go():
        async with tls_harness() as (srv, ext, rec):
            e = await _connect_expect_fatal(ext)
            assert e.code == "tls_error" and ext._fatal
            err = rec.of("error")[0]["error"]
            assert err.code == FATAL and err.vendor_info.code == "tls_error"
            assert "tls_verify" in err.vendor_info.message
            assert len(srv.connections) == 0

    run(go())


def test_self_signed_cert_on_the_websocket_is_fatal_tls_error():
    async def go():
        async with tls_harness() as (srv, ext, rec):
            ext._preflight_ok = True
            e = await _connect_expect_fatal(ext)
            assert e.code == "tls_error" and ext._fatal
            assert rec.of("error")[0]["error"].vendor_info.code == "tls_error"

    run(go())


def test_tls_verify_false_accepts_a_self_signed_cert():
    async def go():
        async with tls_harness(tls_verify=False) as (srv, ext, rec):
            await ext._ensure_connected()
            assert ext.client.is_connected() and srv.models_requests == 1
            await ext.request_tts(text_input("r", "kumusta", end=True))
            await wait_for(lambda: "audio_end" in rec.names(), what="audio_end")

    run(go())


def test_unlisted_model_is_fatal_model_not_found():
    async def go():
        async with FakeNRSPServer(models=[{"id": "halohalo/tts-fil"}]) as srv:
            ext, rec = make_extension(srv.url, model="halohalo/tts-ceb")
            e = await _connect_expect_fatal(ext)
            assert e.code == "model_not_found" and ext._fatal
            err = rec.of("error")[0]["error"]
            assert err.code == FATAL and err.vendor_info.code == "model_not_found"
            assert "halohalo/tts-fil" in err.vendor_info.message  # lists what is available
            assert len(srv.connections) == 0

    run(go())


def test_listed_model_passes_preflight_once():
    async def go():
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url, model="halohalo/tts-fil")
            await ext._ensure_connected()
            await ext.client.stop()
            await ext._ensure_connected()  # reconnect: no second /v1/models round trip
            assert ext._preflight_ok and srv.models_requests == 1 and len(srv.connections) == 2
            await ext.client.stop()

    run(go())


def test_asr_model_is_not_accepted_for_tts():
    """Review F3: kind is checked, not only the id."""
    async def go():
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url, model="fw/whisper-tiny")
            e = await _connect_expect_fatal(ext)
            assert e.code == "model_not_found" and "asr model" in str(e)
            assert len(srv.connections) == 0

    run(go())


def test_exactly_one_fatal_then_requests_end_quietly():
    """Review F1: one FATAL, then no reconnects, no server traffic and no further errors."""
    async def go():
        async with tls_harness() as (srv, ext, rec):
            await _connect_expect_fatal(ext)
            for i in range(5):
                await ext.request_tts(text_input(f"r{i}", "kumusta", end=True))
            await ext._on_error({"type": "error", "code": "engine_error", "fatal": True,
                                 "vendor": {"name": "halohalo", "message": "again"}})
            errors = rec.of("error")
            assert len(errors) == 1 and errors[0]["error"].vendor_info.code == "tls_error"
            assert [f["error"] for f in rec.of("finish")] == [None] * 5
            assert len(rec.of("audio_end")) == 5 and ext._reconnect_task is None
            assert len(srv.connections) == 0 and srv.models_requests == 0  # TLS failed before HTTP

    run(go())


def test_server_vendor_is_named_in_the_message_not_the_vendor_field():
    async def go():
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url)
            await ext._on_error({"type": "error", "code": "engine_error", "fatal": False,
                                 "vendor": {"name": "halohalo", "code": "busy", "message": "busy"}})
            vi = rec.of("error")[0]["error"].vendor_info
            assert (vi.vendor, vi.code, vi.message) == ("nyo", "busy", "[halohalo] busy")

    run(go())


def test_reconnect_loop_ends_in_exactly_one_fatal():
    """ten review F4: a gateway that never comes back ends the reconnect loop with one FATAL connect_failed (as
    the ASR twin does), after which requests end with ERROR and contact nothing."""
    from ten_ai_base.message import TTSAudioEndReason

    async def go():
        ext, rec = make_extension("ws://127.0.0.1:1")          # nothing listens: every attempt is refused
        ext._reconnect_delay = lambda attempt: 0.0
        ext._schedule_reconnect()
        await wait_for(lambda: rec.of("error"), timeout=20, what="fatal after the last attempt")
        await wait_for(lambda: ext._reconnect_task is None, what="loop end")
        errors = rec.of("error")
        assert len(errors) == 1
        err = errors[0]["error"]
        assert int(err.code) == int(ModuleErrorCode.FATAL_ERROR.value)
        assert err.vendor_info.vendor == "nyo" and err.vendor_info.code == "connect_failed"
        assert "5 reconnect attempts failed" in err.message
        assert ext._fatal
        ext._schedule_reconnect()                                # no new loop after the FATAL
        assert ext._reconnect_task is None
        await ext.request_tts(text_input("after", "kumusta", end=True))
        assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.ERROR
        assert len(rec.of("error")) == 1 and rec.of("finish")[-1]["error"] is None

    run(go())


def test_reconnect_loop_stops_quietly_when_a_retry_succeeds():
    async def go():
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url)
            ext._reconnect_delay = lambda attempt: 0.0
            ext._schedule_reconnect()
            await wait_for(lambda: ext._reconnect_task is None and ext.client is not None, what="reconnected")
            assert rec.of("error") == [] and not ext._fatal and len(srv.connections) == 1
            await ext.client.stop()

    run(go())


def test_request_during_the_last_reconnect_attempt_still_gives_one_fatal():
    """architect r3 F2: a request that arrives while the last reconnect attempt is in flight (its own lazy
    connect fails too) must not add a second FATAL: one FATAL connect_failed, the request ends with ERROR."""
    import asyncio
    from ten_ai_base.message import TTSAudioEndReason

    async def go():
        ext, rec = make_extension("ws://127.0.0.1:1")
        ext._reconnect_delay = lambda attempt: 0.0
        real_connect, calls = ext._connect, []
        req_task = []

        async def connect():
            calls.append(1)
            if len(calls) == 5 and not req_task:                  # the loop's last attempt
                req_task.append(asyncio.create_task(ext.request_tts(text_input("mid", "kumusta", end=True))))
                await asyncio.sleep(0.05)                          # the request's own connect runs meanwhile
            await real_connect()

        ext._connect = connect
        ext._schedule_reconnect()
        await wait_for(lambda: ext._reconnect_task is None and req_task and req_task[0].done(), timeout=20,
                       what="loop end and request end")
        fatal = int(ModuleErrorCode.FATAL_ERROR.value)
        sent = [e["error"] for e in rec.of("error")] + [f["error"] for f in rec.of("finish") if f["error"]]
        assert [int(e.code) for e in sent].count(fatal) == 1, [(e.code, e.message) for e in sent]
        assert rec.of("audio_end")[-1]["request_id"] == "mid"
        assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.ERROR

    run(go())


def test_fatal_stop_task_is_kept_alive_until_done():
    """ten r3 F3: the background client.stop() started on a FATAL is strongly referenced until it finishes."""
    import asyncio
    from unittest.mock import MagicMock
    from ten_ai_base.message import ModuleError, ModuleType

    async def go():
        ext, rec = make_extension()
        gate = asyncio.Event()
        client = MagicMock()

        async def slow_stop():
            await gate.wait()

        client.stop = slow_stop
        ext.client = client
        await ext._send_error("", ModuleError(message="x", module=ModuleType.TTS, code=ModuleErrorCode.FATAL_ERROR))
        assert len(ext._bg_tasks) == 1
        gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(ext._bg_tasks) == 0

    run(go())

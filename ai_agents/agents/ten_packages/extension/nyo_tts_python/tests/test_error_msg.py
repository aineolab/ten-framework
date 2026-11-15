import pytest
"""Vendor errors, fatal vs non-fatal, connection loss, watchdog."""
import asyncio

from ten_ai_base.message import TTSAudioEndReason

from nyo_tts_python.tts_client import NYOTTSConnectError

from .helpers import harness, make_extension, run, text_input, wait_for

FATAL, NON_FATAL = -1000, 1000


def test_invalid_key_is_fatal_401_with_vendor_info():
    async def go():
        async with harness(key="invalid") as (srv, ext, rec):
            try:
                await ext._ensure_connected()
            except NYOTTSConnectError as e:
                assert e.status_code == 401
            else:
                raise AssertionError("expected a connect error")
            err = rec.of("error")[0]["error"]
            assert err.code == FATAL and err.vendor_info.vendor == "nyo" and err.vendor_info.code == "401"
            assert ext._fatal is True
            assert rec.of("disconnected")[0]["code"] == 401

    run(go())


def test_connect_refused_is_not_fatal():
    async def go():
        ext, rec = make_extension("ws://127.0.0.1:1")
        try:
            await ext._ensure_connected()
        except NYOTTSConnectError as e:
            assert e.status_code is None
        assert ext._fatal is False
        assert rec.of("error") == []  # request_tts surfaces it per request instead

    run(go())


def test_request_on_dead_server_finishes_with_nonfatal_error():
    async def go():
        ext, rec = make_extension("ws://127.0.0.1:1")
        ext._schedule_reconnect = lambda: None
        await ext.request_tts(text_input("r", "hello", end=True))
        assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.ERROR
        fin = rec.of("finish")[0]
        assert fin["reason"] == TTSAudioEndReason.ERROR and fin["error"].code == NON_FATAL

    run(go())


def test_request_scoped_server_error_finishes_request_with_vendor_info():
    async def go():
        async with harness(key="k+ttserr") as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "please ERROR_TRIGGER now", end=True))
            await wait_for(lambda: "finish" in rec.names())
            fin = rec.of("finish")[0]
            assert fin["error"].code == NON_FATAL
            assert fin["error"].vendor_info.code == "engine_error"
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.ERROR
            assert ext._fatal is False
            # connection survives and the next request works
            rec.events.clear()
            await ext.request_tts(text_input("r2", "fine", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END

    run(go())


def test_fatal_server_error_marks_extension_fatal():
    async def go():
        async with harness(key="k+fatal") as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "boom", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert rec.of("finish")[0]["error"].code == FATAL
            assert ext._fatal is True

    run(go())


def test_connection_lost_mid_request_errors_the_request_and_reconnects_lazily():
    async def go():
        async with harness(server_kw={"chunk_interval_ms": 50}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "x" * 100))
            await wait_for(lambda: "audio" in rec.names())
            ext.client.websocket.transport.abort()  # yank the cable
            await wait_for(lambda: "finish" in rec.names(), what="request failed")
            fin = rec.of("finish")[0]
            assert fin["error"].vendor_info.code == "connection_lost"
            assert fin["error"].code == NON_FATAL
            rec.events.clear()
            await ext.request_tts(text_input("r2", "again", end=True))  # lazy reconnect inside
            await wait_for(lambda: "finish" in rec.names())
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert len(srv.connections) == 2

    run(go())


def test_watchdog_fails_a_silent_request():
    async def go():
        async with harness(request_timeout_ms=150, server_kw={"first_audio_delay_ms": 5000}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "never answered", end=True))
            await wait_for(lambda: "finish" in rec.names(), timeout=2, what="watchdog")
            fin = rec.of("finish")[0]
            assert fin["error"].code == NON_FATAL and "no audio_end" in fin["error"].message
            await wait_for(lambda: srv.received("flush"), what="flush at server")
            assert srv.received("flush") == [{"type": "flush", "request_id": "r"}]

    run(go())


def test_on_init_with_empty_key_emits_fatal_minus_1000():
    from unittest.mock import AsyncMock, MagicMock
    from ten_ai_base.tts2 import AsyncTTS2BaseExtension

    async def go():
        ext, rec = make_extension()
        env = MagicMock()
        env.get_property_to_json = AsyncMock(
            return_value=('{"params": {"api_key": "", "base_url": "wss://nyo.example.test"}}', None))
        orig = AsyncTTS2BaseExtension.on_init
        AsyncTTS2BaseExtension.on_init = AsyncMock()
        try:
            ext.ten_env = env
            await ext.on_init(env)
        finally:
            AsyncTTS2BaseExtension.on_init = orig
        err = rec.of("error")[0]["error"]
        assert err.code == FATAL and err.vendor_info.code == "invalid_config"

    run(go())


def test_missing_base_url_is_fatal_and_never_connects():
    """Review item 1: no default host. A missing base_url fails at init with no preflight or socket."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import nyo_tts_python.transport as transport
    import nyo_tts_python.tts_client as tts_client
    from ten_ai_base.tts2 import AsyncTTS2BaseExtension

    async def go():
        ext, rec = make_extension()
        env = MagicMock()
        env.get_property_to_json = AsyncMock(return_value=('{"params": {"api_key": "k"}}', None))
        orig = AsyncTTS2BaseExtension.on_init
        AsyncTTS2BaseExtension.on_init = AsyncMock()
        try:
            ext.ten_env = env
            with patch.object(transport, "fetch_models", AsyncMock()) as models, \
                    patch.object(tts_client.websockets, "connect", AsyncMock()) as connect:
                await ext.on_init(env)
                ext.client = None
                with pytest.raises(NYOTTSConnectError):
                    await ext._ensure_connected()
        finally:
            AsyncTTS2BaseExtension.on_init = orig
        err = rec.of("error")[0]["error"]
        assert err.code == FATAL and err.vendor_info.code == "invalid_config"
        assert "base_url is required" in err.message
        models.assert_not_awaited()
        connect.assert_not_awaited()

    run(go())

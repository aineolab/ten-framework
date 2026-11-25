import pytest
import asyncio
from unittest.mock import AsyncMock
from nyo_asr_python.extension import NYOASRExtension

from .helpers import make_extension, run

FATAL = -1000
NON_FATAL = 1000


def test_nonfatal_server_error_keeps_vendor_info():
    ext = make_extension()
    run(ext.on_error("rate limited", "admission_rejected", False, "deepgram", "429"))
    err, vi = ext.send_asr_error.await_args.args
    assert err.code == NON_FATAL and err.module == "asr"
    assert (vi.vendor, vi.code, vi.message) == ("nyo", "429", "[deepgram] rate limited")  # review F5
    assert ext._fatal is False


def test_fatal_server_error_stops_reconnects():
    ext = make_extension()
    run(ext.on_error("bad key", "unauthorized", True, "nyo", "unauthorized"))
    err, vi = ext.send_asr_error.await_args.args
    assert err.code == FATAL and vi.code == "unauthorized"
    assert ext._fatal is True


def test_connect_401_is_fatal_and_reports_http_status():
    ext = make_extension()
    e = Exception("server rejected WebSocket connection: HTTP 401")
    e.status_code = 401
    run(ext._report_connect_failure(e))
    err, vi = ext.send_asr_error.await_args.args
    assert err.code == FATAL and vi.vendor == "nyo" and vi.code == "401"
    assert ext._fatal is True
    ext.on_disconnected.assert_awaited_once()


def test_connect_refused_is_nonfatal():
    ext = make_extension()
    run(ext._report_connect_failure(ConnectionRefusedError("refused")))
    err, vi = ext.send_asr_error.await_args.args
    assert err.code == NON_FATAL and vi.code == "connect_failed"
    assert ext._fatal is False


def test_fatal_status_set():
    ext = make_extension()
    assert ext._is_fatal_status(401) and ext._is_fatal_status(403)
    assert not ext._is_fatal_status(503) and not ext._is_fatal_status(None)


def test_404_and_400_are_fatal_only_when_models_conclusively_miss_the_model():
    """QA r2 rule B: a 400/404 handshake re-checks GET /v1/models; FATAL only on a conclusive miss."""
    from nyo_asr_python.recognition import NRSPConnectError
    for verdict, fatal in (("inconclusive", False), ("ok", False), ("bad", True), ("auth", True)):
        ext = make_extension()
        ext._models_verdict = AsyncMock(return_value=(verdict, "model_not_found", "not served"))
        for _ in range(5):
            got, _, _ = run(ext._classify_connect_failure(NRSPConnectError("404", 404)))
            assert got is fatal
    ext = make_extension()
    ext._models_verdict = AsyncMock()
    assert run(ext._classify_connect_failure(NRSPConnectError("503", 503)))[0] is False
    ext._models_verdict.assert_not_awaited()  # only 400/404 re-check the listing


def test_invalid_config_emits_fatal_with_vendor_info():
    from unittest.mock import AsyncMock, MagicMock

    async def go():
        ext = make_extension()
        env = MagicMock()
        env.get_property_to_json = AsyncMock(
            return_value=('{"params": {"api_key": "", "base_url": "wss://nyo.example.test"}}', None))
        # Skip the base-class on_init (needs a live runtime) by pre-seeding what it sets.
        from ten_ai_base.asr import AsyncASRBaseExtension
        orig = AsyncASRBaseExtension.on_init
        AsyncASRBaseExtension.on_init = AsyncMock()
        try:
            ext.ten_env = env
            ext.config = None
            await ext.on_init(env)
        finally:
            AsyncASRBaseExtension.on_init = orig
        err, vi = ext.send_asr_error.await_args.args
        assert err.code == FATAL and vi.code == "invalid_config" and "API key" in err.message
        assert ext._init_failed is True

    run(go())


def test_exactly_one_fatal_then_silence_and_no_audio():
    """Review F1: after the first FATAL the extension stops: no second FATAL, no further errors, no audio."""
    from unittest.mock import AsyncMock, MagicMock

    async def go():
        ext = make_extension()
        ext._emit_connection_transition = AsyncMock()
        rec = MagicMock()
        rec.is_connected.return_value = True
        rec.close = AsyncMock()
        ext.recognition, ext._live = rec, True
        await ext.on_error("unsupported_language:en", "unsupported_language:en", True, "halohalo", "x")
        for _ in range(20):  # what a gateway that reopens the engine keeps sending (round 1: 21 FATAL)
            await ext.on_error("again", "unsupported_language:en", True, "halohalo", "x")
            await ext.on_error("engine session lost", "engine_unavailable", False, "halohalo", "y")
        assert ext.send_asr_error.await_count == 1
        assert ext.send_asr_error.await_args.args[0].code == FATAL
        assert ext.recognition is None and not ext.is_connected()
        assert await ext.send_audio(MagicMock(), "s") is False
        await asyncio.sleep(0)
        rec.close.assert_awaited_once()
        await ext.start_connection()  # a later start does nothing either
        assert ext.send_asr_error.await_count == 1

    run(go())


def test_missing_base_url_is_fatal_and_never_connects():
    """Review item 1: no default host. A missing base_url fails at init with no preflight or socket."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import nyo_asr_python.recognition as recognition
    import nyo_asr_python.transport as transport

    async def go():
        ext = make_extension()
        env = MagicMock()
        env.get_property_to_json = AsyncMock(return_value=('{"params": {"api_key": "k"}}', None))
        env.send_data = AsyncMock()  # the base start_connection wrapper reports a status change
        from ten_ai_base.asr import AsyncASRBaseExtension
        orig = AsyncASRBaseExtension.on_init
        AsyncASRBaseExtension.on_init = AsyncMock()
        try:
            ext.ten_env = env
            ext.config = None
            with patch.object(transport, "fetch_models", AsyncMock()) as models, \
                    patch.object(recognition.websockets, "connect", AsyncMock()) as connect:
                await ext.on_init(env)
                await ext.start_connection()
        finally:
            AsyncASRBaseExtension.on_init = orig
        err, vi = ext.send_asr_error.await_args.args
        assert err.code == FATAL and vi.code == "invalid_config" and err.message == "base_url is required"
        models.assert_not_awaited()
        connect.assert_not_awaited()

    run(go())


def test_build_url_refuses_an_empty_base_url():
    from nyo_asr_python.recognition import NYOASRRecognition

    rec = NYOASRRecognition.__new__(NYOASRRecognition)
    rec.config = {"api_key": "k", "base_url": ""}
    with pytest.raises(ValueError, match="base_url is required"):
        rec.build_url()

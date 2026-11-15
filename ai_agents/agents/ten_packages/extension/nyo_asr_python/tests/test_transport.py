"""Transport policy (plan 4.6), /v1/models preflight and server-side endpointing (C1, D21)."""
import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from fake_nrsp_server import FakeNRSPServer, self_signed_ssl_context
from nyo_asr_python import transport
from nyo_asr_python.reconnect_manager import ReconnectManager
from nyo_asr_python.recognition import NYOASRRecognition

from .helpers import make_config, make_extension, run
from .test_protocol import Frame, session, wait_for

SILENCE_20MS = b"\x00\x00" * 320


@contextlib.asynccontextmanager
async def tls_session(**params):
    async with FakeNRSPServer(ssl_context=self_signed_ssl_context()) as srv:
        ext = make_extension(base_url=srv.url, api_key="nyo-test-key", **params)
        ext._emit_connection_transition = AsyncMock()
        ext.reconnect_manager = ReconnectManager(base_delay=0.02, max_delay=0.05, max_attempts=2)
        try:
            yield srv, ext
        finally:
            await ext.stop_connection()


# ------------------------------------------------------------------ URL policy
@pytest.mark.parametrize("url", ["ws://speech-host.local:9100", "ws://100.64.0.11:9200", "ws://example.com"])
def test_ws_to_remote_host_is_refused(url):
    with pytest.raises(transport.TransportError) as ei:
        make_config(base_url=url).validate_config()
    assert ei.value.code == "insecure_url" and "allow_insecure" in str(ei.value)


@pytest.mark.parametrize("url", ["ws://127.0.0.1:9200", "ws://localhost:9200", "ws://[::1]:9200",
                                 "wss://gateway.example.net"])
def test_loopback_ws_and_any_wss_are_allowed(url):
    make_config(base_url=url).validate_config()


def test_allow_insecure_permits_remote_ws():
    make_config(base_url="ws://speech-host.local:9100", allow_insecure=True).validate_config()


def test_insecure_url_is_a_fatal_module_error_at_init():
    from ten_ai_base.asr import AsyncASRBaseExtension

    async def go():
        ext = make_extension()
        env = MagicMock()
        env.get_property_to_json = AsyncMock(return_value=(
            '{"params": {"api_key": "k", "base_url": "ws://speech-host.local:9100"}}', None))
        orig = AsyncASRBaseExtension.on_init
        AsyncASRBaseExtension.on_init = AsyncMock()
        try:
            ext.ten_env = env
            await ext.on_init(env)
        finally:
            AsyncASRBaseExtension.on_init = orig
        err, vi = ext.send_asr_error.await_args.args
        assert err.code == -1000 and vi.vendor == "nyo" and vi.code == "insecure_url"
        assert ext._init_failed

    run(go())


# ------------------------------------------------------------------------ TLS
def test_self_signed_cert_is_fatal_tls_error_before_any_websocket():
    async def go():
        async with tls_session() as (srv, ext):
            await ext.start_connection()
            err, vi = ext.send_asr_error.await_args.args
            assert err.code == -1000 and vi.code == "tls_error"
            assert "certificate" in vi.message.lower() and "tls_verify" in vi.message
            assert ext._fatal and len(srv.connections) == 0

    run(go())


def test_self_signed_cert_on_the_websocket_is_also_fatal_tls_error():
    async def go():
        async with tls_session() as (srv, ext):
            ext._preflight_ok = True  # only the websocket handshake is checked here
            await ext.start_connection()
            err, vi = ext.send_asr_error.await_args.args
            assert err.code == -1000 and vi.code == "tls_error"
            assert ext._fatal and ext._reconnect_task is None

    run(go())


def test_tls_verify_false_accepts_a_self_signed_cert():
    async def go():
        async with tls_session(tls_verify=False) as (srv, ext):
            await ext.start_connection()
            assert ext.is_connected() and not ext._fatal
            assert srv.models_requests == 1 and srv.connections[0]["path"] == "/v1/asr"

    run(go())


# ------------------------------------------------------------------ preflight
def test_listed_model_passes_preflight_once():
    async def go():
        async with session(model="fw/whisper-large-v3-pld-fil-norm") as (srv, ext):
            await ext.start_connection()
            assert ext.is_connected() and ext._preflight_ok and srv.models_requests == 1
            assert srv.connections[0]["query"]["model"] == "fw/whisper-large-v3-pld-fil-norm"

    run(go())


def test_unreachable_models_endpoint_is_not_fatal():
    async def go():
        ext = make_extension(base_url="ws://127.0.0.1:1")
        ext._emit_connection_transition = AsyncMock()
        ext.reconnect_manager = ReconnectManager(base_delay=0.02, max_delay=0.05, max_attempts=1)
        await ext.start_connection()
        assert not ext._fatal and not ext._preflight_ok
        _, vi = ext.send_asr_error.await_args.args
        assert vi.code == "connect_failed"  # the websocket path reports it, non-fatal
        await ext.stop_connection()

    run(go())


# ---------------------------------------------------- C1 / D21 auto-endpointing
def test_server_side_endpointing_gives_one_final_per_utterance_without_finalize():
    async def go():
        async with session(endpointing_ms=200, language="fil") as (srv, ext):
            await ext.start_connection()
            for _ in range(2):  # two utterances: 400 ms speech, 300 ms silence each
                for _ in range(20):
                    await ext.send_audio(Frame(), "s")
                for _ in range(15):
                    await ext.send_audio(Frame(SILENCE_20MS), "s")
            await wait_for(lambda: [o for o in ext.order if o == ("result", True)].__len__() >= 2,
                           what="two auto finals")
            await asyncio.sleep(0.05)
            assert [o for o in ext.order if o[0] == "result" and o[1]] == [("result", True)] * 2
            assert ext.send_asr_finalize_end.await_count == 0  # auto-K ids are not TEN finalizes
            assert [p["id"] for k, p in srv.log if k == "tx:finalize_end"] == ["auto-1", "auto-2"]
            assert srv.received("finalize") == []
            cfg = srv.received("config")[0]
            assert cfg["endpointing_ms"] == 200

    run(go())


def test_default_config_sends_endpointing_ms_1000():
    async def go():
        async with session() as (srv, ext):
            await ext.start_connection()
            await wait_for(lambda: srv.received("config"), what="config")
            assert srv.received("config")[0]["endpointing_ms"] == 1000

    run(go())


def test_auto_finalize_end_does_not_complete_a_pending_finalize():
    async def go():
        rec = NYOASRRecognition("k", MagicMock(), MagicMock(), {}, AsyncMock())
        ev = asyncio.Event()
        rec._finalize_events["fin-1"] = ev
        await rec._dispatch({"type": "finalize_end", "id": "auto-1"})
        assert not ev.is_set()
        await rec._dispatch({"type": "finalize_end", "id": "fin-1"})
        assert ev.is_set()

    run(go())


def test_server_model_is_reported_once_per_change_as_vendor_metric():
    async def go():
        cb = AsyncMock()
        rec = NYOASRRecognition("k", MagicMock(), MagicMock(), {}, cb)
        for m in ("whisper-large-v3-pld-fil-norm", "whisper-large-v3-pld-fil-norm", "whisper-tiny"):
            await rec._dispatch({"type": "result", "text": "x", "final": True, "metadata": {"model": m}})
        models = [c.args[0]["model"] for c in cb.on_usage.await_args_list if "model" in c.args[0]]
        assert models == ["whisper-large-v3-pld-fil-norm", "whisper-tiny"]

        ext = make_extension()
        await ext.on_usage({"model": "whisper-tiny", "server_ttfw_ms": 12.5})
        ext.send_vendor_metrics.assert_awaited_once_with({"server_ttfw_ms": 12.5, "model": "whisper-tiny"})

    run(go())


def test_asr_info_model_prefers_the_server_model():
    async def go():
        async with session(model="halohalo/asr-fil", language="fil") as (srv, ext):
            await ext._emit_asr_result({"text": "kumusta", "metadata": {"model": "whisper-tiny"}}, final=True)
            r = ext.send_asr_result.await_args.args[0]
            assert r.metadata["asr_info"]["model"] == "whisper-tiny"

    run(go())


def test_live_frames_wait_until_the_keep_buffer_is_replayed():
    """Regression (asr_guarder test_dump on the real gateway): the base class must not drain its Keep queue
    while on_open is replaying it, or buffered and live frames interleave out of order."""
    from nyo_asr_python.extension import NYOASRExtension

    async def go():
        ext = make_extension()
        ext.ten_env = MagicMock()
        ext.reconnect_manager = None
        ext.audio_timeline = MagicMock()
        ext.audio_timeline.get_total_user_audio_duration.return_value = 0
        sent, seen_connected = [], []
        rec = MagicMock()
        rec.is_connected.return_value = True
        rec.bytes_per_ms = 32

        async def send(data):
            seen_connected.append(ext.is_connected())
            sent.append(data[0])
            await asyncio.sleep(0)
        rec.send_audio_frame = send
        ext.recognition = rec
        for i in range(5):
            ext.buffered_frames.put_nowait(Frame(bytes([i, 1]) * 320))
        await NYOASRExtension.on_open(ext)
        assert sent == [0, 1, 2, 3, 4] and seen_connected == [False] * 5
        assert ext.is_connected() and ext.buffered_frames.empty()

    run(go())


def test_preflight_checks_kind_and_language():
    """Review F3: a TTS id, or a language the model does not list, fails at start like an unknown id."""
    entries = [{"id": "fw/whisper-large-v3-pld-fil-norm", "kind": "asr", "languages": ["fil"]},
               {"id": "halohalo/tts-fil", "kind": "tts", "language": "fil"}]
    ok = transport.check_model(entries, "fw/whisper-large-v3-pld-fil-norm", "asr", "fil", "ws://127.0.0.1:9200")
    assert ok is None
    code, msg = transport.check_model(entries, "fw/whisper-large-v3-pld-fil-norm", "asr", "en", "ws://127.0.0.1:9200")
    assert code == "language_not_supported" and "'en'" in msg and "fil" in msg
    code, msg = transport.check_model(entries, "halohalo/tts-fil", "asr", "fil", "ws://127.0.0.1:9200")
    assert code == "model_not_found" and "tts model" in msg and "halohalo/tts-fil" not in msg.split("asr models:")[1]
    code, msg = transport.check_model(entries, "fw/x", "tts", None, "ws://127.0.0.1:9200")
    assert code == "model_not_found" and msg.endswith("tts models: halohalo/tts-fil")


def test_unsupported_language_is_fatal_at_start():
    async def go():
        models = [{"id": "fw/whisper-large-v3-pld-fil-norm", "kind": "asr", "languages": ["fil"]}]
        async with FakeNRSPServer(models=models) as srv:
            ext = make_extension(base_url=srv.url, api_key="k", model="fw/whisper-large-v3-pld-fil-norm",
                                 language="en")
            ext._emit_connection_transition = AsyncMock()
            await ext.start_connection()
            err, vi = ext.send_asr_error.await_args.args
            assert err.code == -1000 and vi.code == "language_not_supported"
            assert ext._fatal and len(srv.connections) == 0
            await ext.stop_connection()

    run(go())


def test_interim_then_late_final_after_timeout_promotion_is_dropped():
    """ten review 1: promote on timeout -> interim of the same utterance -> its late real final: one final only."""
    async def go():
        ext = make_extension()
        ext.config.finalize_timeout_ms = 10
        ext._last_interim = {"id": "utt-3", "text": "kumusta", "final": False}
        ext.last_finalize_timestamp = 1
        ext.recognition = MagicMock()                    # a real timeout: the socket is still open (ten F3)
        ext.recognition.is_connected.return_value = True
        await ext._on_finalize_timeout("fin-1", ext.recognition)
        await ext.on_result({"id": "utt-3", "text": "kumusta ka", "final": False})
        await ext.on_result({"id": "utt-3", "text": "kumusta ka na", "final": True})
        finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
        assert len(finals) == 1 and finals[0].metadata["asr_info"]["promoted_on_timeout"]
        await ext.on_result({"id": "utt-4", "text": "salamat", "final": True})  # next utterance is emitted
        assert len([c for c in ext.send_asr_result.await_args_list if c.args[0].final]) == 2

    run(go())


def test_results_are_logged_without_transcript_text():
    """Review item 2: no transcripts in logs, at any level."""
    async def go():
        env = MagicMock()
        rec = NYOASRRecognition("k", MagicMock(), env, {}, AsyncMock())
        await rec._dispatch({"type": "result", "id": "u1", "final": True,
                             "text": "ang PIN ko ay 4821", "words": [{"word": "PIN"}]})
        logged = " ".join(str(c) for c in env.mock_calls)
        assert "vendor_result: result id=u1 final=True chars=18" in logged
        assert "4821" not in logged and "PIN" not in logged

    run(go())

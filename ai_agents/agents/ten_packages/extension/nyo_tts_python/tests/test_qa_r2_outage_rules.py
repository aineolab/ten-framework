"""QA round 2: shared rules A (model-missing decision), C (the outage clock resets only on a proven connection)
and D (NON_FATAL connection errors are rate-limited within one outage); update_configs checks /v1/models."""
import asyncio
import time

import pytest
from ten_ai_base.message import ModuleErrorCode, TTSAudioEndReason
from websockets.asyncio.server import serve

from fake_nrsp_server import FakeNRSPServer
from nyo_tts_python.extension import served_verdict

from .helpers import harness, make_extension, run, text_input, wait_for

FATAL = int(ModuleErrorCode.FATAL_ERROR.value)
NON_FATAL = int(ModuleErrorCode.NON_FATAL_ERROR.value)

LIVE = [  # the shape the live gateway lists (trimmed)
    {"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"},
    {"id": "halohalo/tts-fil", "kind": "tts", "language": "fil", "voices": ["FIL_0051", "FIL_0078"],
     "default_voice": "FIL_0078", "voice_details": [{"id": "FIL_0051"}, {"id": "FIL_0078"}]},
    {"id": "halohalo/tts-hil", "kind": "tts", "language": "hil", "voices": ["default"]},
]


# ------------------------------------------------------------------ rule A
def test_rule_a_verdicts():
    assert served_verdict(None, "halohalo/tts-fil", "", "") is None
    assert served_verdict([], "halohalo/tts-fil", "", "") is None
    assert served_verdict([{"id": "x/y", "kind": "asr"}], "halohalo/tts-x", "", "") is None  # no tts model listed
    # a swap drops the halohalo/ TTS ids while other namespaces stay listed: not proof (transport.lists_kind)
    assert served_verdict([{"id": "other/tts", "kind": "tts"}], "halohalo/tts-x", "", "") is None
    assert served_verdict([{"id": "halohalo/tts-fil"}], "halohalo/tts-x", "", "")[0] == "model_not_found"
    assert served_verdict(LIVE, "halohalo/tts-fil", "", "fil") is True
    assert served_verdict(LIVE, "halohalo/tts-ceb", "", "")[::2] == ("model_not_found", False)   # rule A': maybe down
    assert served_verdict(LIVE, "halohalo/asr-fil", "", "")[::2] == ("model_not_found", True)    # listed, wrong kind
    assert served_verdict(LIVE, "halohalo/tts-fil", "", "ceb")[2] is False          # that engine may be down
    assert served_verdict(LIVE, "halohalo/tts-fil", "nobody", "")[2] is True
    assert "asr model" in served_verdict(LIVE, "halohalo/asr-fil", "", "")[1]
    # language: any language a TTS model declares (``language`` string or ``languages`` list) is routable
    assert served_verdict(LIVE, "halohalo/tts-fil", "", "hil") is True
    assert served_verdict(LIVE, "halohalo/tts-fil", "", "ceb")[0] == "language_not_supported"
    lst = [{"id": "m/t", "kind": "tts", "languages": ["en", "fil"]}]
    assert served_verdict(lst, "m/t", "", "fil") is True and served_verdict(lst, "m/t", "", "ja")[0] == "language_not_supported"
    # voice: against the catalogue (case-insensitive, default aliases pass), else the route's allow-list (QA r4)
    assert served_verdict(LIVE, "halohalo/tts-fil", "fil_0051", "") is True
    assert served_verdict(LIVE, "halohalo/tts-fil", "alloy", "") is True
    assert served_verdict(LIVE, "halohalo/tts-fil", "nobody", "")[0] == "unknown_voice"
    assert served_verdict(LIVE, "halohalo/tts-hil", "nobody", "")[::2] == ("unknown_voice", True)  # allow-list
    assert served_verdict(LIVE, "halohalo/tts-hil", "alloy", "")[0] == "unknown_voice"  # no aliases without a catalogue
    assert served_verdict(LIVE, "halohalo/tts-hil", "default", "") is True
    assert served_verdict(LIVE, "halohalo/tts-hil", "Default", "")[0] == "unknown_voice"  # exact, as pick_voice
    open_list = [{"id": "halohalo/tts-hil", "kind": "tts", "language": "hil", "voices": []}]
    assert served_verdict(open_list, "halohalo/tts-hil", "anyone", "") is True    # "*" only: listed as []


# ------------------------------------------------------------------ rule C
def test_rule_c_flapping_connections_do_not_reset_the_outage_clock():
    async def accept_then_close(ws):
        await ws.close(1011)

    async def go():
        async with serve(accept_then_close, "127.0.0.1", 0, subprotocols=["nrsp.v1"]) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            ext, rec = make_extension(url, reconnect_window_ms=800)
            ext._preflight_ok = True
            ext._reconnect_delay = lambda attempt: 0.05
            await ext._begin_request(text_input("r", "x"))    # a request waits (idle, the window end is not FATAL)
            t0 = time.monotonic()
            await ext._ensure_connected()                    # opens fine, then dies at once, again and again
            try:
                await wait_for(lambda: ext._fatal, timeout=4, what="fatal after the window")
            finally:
                ext._stopped = True
                ext._cancel_reconnect_task()
                if ext.client:
                    await ext.client.stop()
            assert 0.7 <= time.monotonic() - t0 < 3
            assert [e["error"].vendor_info.code for e in rec.of("error")] == ["connect_failed"]

    run(go())


def test_rule_c_a_completed_request_or_10s_uptime_resets_the_clock():
    async def go():
        async with harness() as (srv, ext, rec):
            ext._outage_start = time.monotonic() - 100            # left over from an earlier outage
            await ext._ensure_connected()
            assert ext._outage_start is not None                  # an open socket alone proves nothing yet
            await ext.request_tts(text_input("r", "kumusta ", end=True))
            await wait_for(lambda: rec.of("audio_end"))
            assert ext._outage_start is None                      # completed a request
            ext._outage_start = time.monotonic() - 100
            ext._conn_proven, ext._conn_up_at = False, time.monotonic() - 11   # up for > 10 s
            ext._note_outage()
            assert time.monotonic() - ext._outage_start < 1       # a new outage starts now

    run(go())


# ------------------------------------------------------------------ rule D
def test_rule_d_connection_errors_are_rate_limited_within_one_outage():
    async def go():
        ext, rec = make_extension()
        for i in range(4):                                        # four drops within 10 s, one outage
            await ext._begin_request(text_input(f"r{i}", "x"))
            await ext._on_closed(False, "boom")
        ext._stopped = True
        ext._cancel_reconnect_task()
        assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.ERROR] * 4
        errs = [f["error"] for f in rec.of("finish") if f["error"]]
        assert len(errs) == 1 and int(errs[0].code) == NON_FATAL
        ext._outage_err_at -= 11                                  # 10 s later: one more may be reported
        ext._stopped = False
        await ext._begin_request(text_input("r9", "x"))
        await ext._on_closed(False, "boom")
        ext._stopped = True
        ext._cancel_reconnect_task()
        assert len([f for f in rec.of("finish") if f["error"]]) == 2

    run(go())


# ------------------------------------------------------- update_configs (finding 2)
def test_update_configs_rejects_a_route_the_listing_does_not_serve_and_keeps_the_old_config():
    async def go():
        async with harness(model="halohalo/tts-fil", server_kw={"models": LIVE}) as (srv, ext, rec):
            await ext._ensure_connected()
            before = (ext.config.model, ext.config.voice, ext.config.language)
            for params in ({"voice": "nobody"}, {"model": "halohalo/tts-ceb"}, {"language": "ceb"}):
                with pytest.raises(ValueError, match="rejected"):
                    await ext.update_configs({"params": params})
                assert (ext.config.model, ext.config.voice, ext.config.language) == before
                assert ext._pending_config is None and not ext._fatal
            await ext.update_configs({"params": {"voice": "FIL_0051"}})       # served: applied
            assert ext.config.voice == "FIL_0051" and srv.connections[-1]["query"]["voice"] == "FIL_0051"
            with pytest.raises(ValueError, match="unknown_voice"):                  # not in hil's allow-list
                await ext.update_configs({"params": {"model": "halohalo/tts-hil", "voice": "anyone"}})
            assert ext.config.model == "halohalo/tts-fil" and ext.config.voice == "FIL_0051"
            await ext.update_configs({"params": {"model": "halohalo/tts-hil", "voice": "default"}})
            assert ext.config.voice == "default"

    run(go())


def test_update_configs_is_accepted_when_the_listing_is_unreachable():
    async def go():
        ext, rec = make_extension("ws://127.0.0.1:1")
        ext._reconnect_delay = lambda attempt: 0.05
        await ext.update_configs({"params": {"voice": "nobody"}})
        assert ext.config.voice == "nobody" and not ext._fatal
        ext._stopped = True
        ext._cancel_reconnect_task()

    run(go())


def test_rule_a_prime_unlisted_language_retries_then_fatal_language_not_supported():
    """A language no listed TTS model declares may be that language's engine being down: NON_FATAL, retried,
    FATAL with the same code only when the window ends; requests meanwhile end NON_FATAL, never hang."""
    async def go():
        async with FakeNRSPServer(models=LIVE) as srv:
            ext, rec = make_extension(srv.url, model="halohalo/tts-fil", language="ceb",
                                      reconnect_window_ms=800, request_timeout_ms=300)
            ext._reconnect_delay = lambda attempt: 0.05
            await asyncio.wait_for(ext.request_tts(text_input("r", "maayong aga", end=True)), 3)
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.ERROR and not ext._fatal
            await asyncio.sleep(1.0)                      # the window ends while idle: no FATAL (QA r4)
            assert not ext._fatal
            await asyncio.wait_for(ext.request_tts(text_input("r2", "maayong aga", end=True)), 3)
            await wait_for(lambda: ext._fatal, timeout=4, what="FATAL once a request is affected")
            codes = [(int(e["error"].code), e["error"].vendor_info.code) for e in rec.of("error")]
            assert codes[0] == (NON_FATAL, "language_not_supported") and codes[-1] == (FATAL, "language_not_supported")
            assert len(srv.connections) == 0
            ext._stopped = True

    run(go())

"""QA round 9 (fixed): an in-band engine error costs one segment, not the whole TEN request.

The Orpheus and SpeechT5 modules report a failed synthesis in-band as {"type":"error","code":"tts_engine_error"}
(modules/tts/engine.py:405, modules/tts_speecht5/engine.py:168, modules/tts/server.py:89). serve/remote.py:417 keeps
that code and serve/gateway.py:1363 puts it on the wire unchanged as a non-fatal, request-scoped error, followed by
audio_end{error}. TRANSIENT_SEGMENT_CODES only lists the gateway's own "engine_error", so this one takes the generic
path in _on_error: the TEN request ends with tts_audio_end(ERROR) at once and every later chunk of the reply is a
late chunk (not spoken) - although the README says an engine error costs that segment only. Now any non-fatal
error naming a segment costs that segment; stale_request and the route errors still end the request.
"""
import asyncio

from ten_ai_base.message import TTSAudioEndReason

from .helpers import harness, run, text_input, wait_for


def _wire(srv):
    return [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]


def _engine_error(rid):
    # exactly what serve/gateway.py:1363 sends for an engine item {"type":"error","code":"tts_engine_error",...}
    return {"type": "error", "code": "tts_engine_error", "fatal": False, "message": "tts engine error",
            "vendor": {"name": "halohalo", "code": "tts_engine_error", "message": "tts engine error"},
            "request_id": rid}


def test_qa_r9_module_engine_error_costs_one_segment_not_the_reply():
    async def go():
        async with harness(segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una."))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1 spoken")
            await ext.request_tts(text_input("r", " Ikalawa."))          # segment r~2
            await ext._on_error(_engine_error("r~2"), ext.client)       # the engine failed on segment 2
            await ext._on_audio_end("r~2", "error", 0, {})               # the gateway's audio_end{error}
            await asyncio.sleep(0.05)
            # expected (README "A failed segment is not a failed reply"): one NON_FATAL error, the request goes on
            assert ext._req.open and not rec.of("audio_end"), rec.names()
            errs = rec.of("error")
            assert [e["error"].vendor_info.code for e in errs] == ["tts_engine_error"], errs
            assert int(errs[0]["error"].code) == 1000
            await ext.request_tts(text_input("r", " Ikatlo.", end=True))  # the rest of the reply is spoken
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert ("r~3", " Ikatlo.", True) in _wire(srv), _wire(srv)
            assert len(rec.of("error")) == 1 and rec.names().count("audio_start") == 1

    run(go())


def test_qa_r9_route_error_on_a_segment_still_ends_the_request():
    async def go():
        async with harness(segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una."))
            err = _engine_error("r")
            err["code"] = err["vendor"]["code"] = "unsupported_voice"
            await ext._on_error(err, ext.client)
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.ERROR and not ext._req.open

    run(go())

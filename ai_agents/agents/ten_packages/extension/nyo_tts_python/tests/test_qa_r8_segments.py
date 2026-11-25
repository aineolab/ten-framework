"""QA round 8 (fixed): segment_max_ms, the idle-close resend for any segment, transient segment failures.

* segment_max_ms: a segment open that long ends at the next sentence-final chunk (not after a digit), so SpeechT5
  (which speaks only at the end of its request) stays under the gateway's tts_stall_s (10 s) on long fast replies;
* got_reply is per segment: an idle close that crosses the first text of segment 2 resends that segment only;
* a transient segment failure (admission_rejected, engine_error, synthesis_truncated, gateway cancel, lost socket)
  costs that segment: one NON_FATAL error, the TEN request goes on; REQUEST_END if any audio, else ERROR.
Fake NRSP server (in-process), plus one real-engine check (SpeechT5) when the stack on 9150 runs."""
import asyncio
import os
import socket

import pytest
from ten_ai_base.message import TTSAudioEndReason

from .helpers import harness, make_extension, run, text_input, wait_for


def _wire(srv):
    return [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]


def test_qa_r8_segment_max_ends_a_long_segment_at_a_sentence_end():
    async def go():
        async with harness(segment_idle_ms=5000, segment_max_ms=300, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            for c in ("Una.", " Ikalawa,"):
                await ext.request_tts(text_input("r", c))
                await asyncio.sleep(0.2)
            await ext.request_tts(text_input("r", " noong ₱3."))  # >= 300 ms open, but after a digit: no end
            await ext.request_tts(text_input("r", "50 iyon."))         # a sentence end: the segment ends here
            await ext.request_tts(text_input("r", " Ikaapat."))         # opens r~2
            await ext.request_tts(text_input("r", "", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            assert _wire(srv) == [("r", "Una.", False), ("r", " Ikalawa,", False), ("r", " noong ₱3.", False),
                                  ("r", "50 iyon.", False), ("r", "", True),
                                  ("r~2", " Ikaapat.", False), ("r~2", "", True)], _wire(srv)
            end = rec.of("audio_end")[0]
            assert end["reason"] == TTSAudioEndReason.REQUEST_END and rec.names().count("audio_start") == 1
            assert not rec.of("error")

    run(go())


def test_qa_r8_idle_close_crossing_segment_2_resends_segment_2_only():
    async def go():
        async with harness(segment_idle_ms=150, server_kw={"first_audio_delay_ms": 300}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una."))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1 spoken")
            await ext.request_tts(text_input("r", " Ikalawa po."))  # segment 2: no reply yet (300 ms delay)
            sock = ext.client
            # the gateway's idle close crossed segment 2's first text (it never saw it)
            await ext._on_error({"type": "error", "code": "idle_timeout", "fatal": True}, sock)
            await sock.websocket.close()
            await wait_for(lambda: ext.client is not sock and ext._req.sent_on is ext.client, timeout=3,
                           what="resend on a fresh socket")
            await ext.request_tts(text_input("r", "", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert not rec.of("error") and not rec.of("finish_error")
            texts = [(w[0], w[1]) for w in _wire(srv)]
            assert texts.count(("r", "Una.")) == 1, texts             # segment 1 is never spoken twice
            assert texts.count(("r~2", " Ikalawa po.")) == 2, texts   # segment 2 resent once

    run(go())


def test_qa_r8_admission_rejected_on_a_segment_costs_that_segment_only():
    async def go():
        async with harness(segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una."))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1 spoken")
            await ext.request_tts(text_input("r", " Ikalawa."))
            await ext._on_error({"type": "error", "code": "admission_rejected", "fatal": False, "request_id": "r~2",
                                 "message": "no headroom"}, ext.client)
            await ext._on_audio_end("r~2", "error", 0, {})          # the gateway's audio_end{error} that follows
            await ext._on_error({"type": "error", "code": "stale_request", "fatal": False, "request_id": "r~2",
                                 "message": "late text"}, ext.client)  # text that crossed it: ignored
            errs = rec.of("error")
            assert [e["error"].vendor_info.code for e in errs] == ["admission_rejected"]
            assert ext._req.open and not rec.of("audio_end")
            await ext.request_tts(text_input("r", " Ikatlo.", end=True))  # next chunk: segment 3
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert ("r~3", " Ikatlo.", True) in _wire(srv) and len(rec.of("error")) == 1

    run(go())


def test_qa_r8_socket_lost_while_a_segment_plays_the_request_goes_on():
    async def go():
        async with harness(key="nyo-test-key+rt", segment_idle_ms=150) as (srv, ext, rec):
            ext._reconnect_delay = lambda attempt: 0.01
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Mahabang unang pangungusap po ito."))
            await wait_for(lambda: len(rec.of("audio")) > 2, timeout=3, what="segment 1 playing")
            ext.client.websocket.transport.abort()
            await wait_for(lambda: rec.of("error"), timeout=3, what="NON_FATAL connection_lost")
            assert rec.of("error")[0]["error"].vendor_info.code == "connection_lost" and ext._req.open
            await ext.request_tts(text_input("r", " Ikalawa.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=8, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert rec.names().count("audio_start") == 1 and len(rec.of("error")) == 1
            ts = [a["ts"] for a in rec.of("audio")]
            assert ts == sorted(ts), ts  # one timeline across the lost and the new segment

    run(go())


# ------------------------------------------------------------------- real engine (SpeechT5, halohalo/tts-hil)
GW = "ws://127.0.0.1:9150"
KEY_FILE = os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key")


def _up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", 9150), 0.3).close()
        return os.path.exists(KEY_FILE)
    except OSError:
        return False


@pytest.mark.skipif(not _up(), reason="real stack on 9150 not running")
def test_qa_r8_real_hil_long_fast_reply_stays_under_the_gateway_stall_timer():
    """A 14 s fast reply (a chunk every 0.4 s, never a 1 s gap) on hil: cut into segments of about segment_max_ms
    (each NRSP request stays under the gateway's 10 s stall timer for an engine that speaks only at the end of its
    request, as SpeechT5 does), all spoken, one TEN request. (The 9150 stack's hil adapter streams per sentence, so
    it does not reproduce the stall itself; the cut is what this checks.)"""
    async def go():
        key = open(KEY_FILE).read().strip()
        ext, rec = make_extension(GW, api_key=key, model="halohalo/tts-hil")
        try:
            await ext._ensure_connected()
            for i in range(35):
                await ext.request_tts(text_input("tts-request-1", f" Pangungusap bilang {'isa dalawa tatlo'.split()[i % 3]}."))
                await asyncio.sleep(0.4)
            await ext.request_tts(text_input("tts-request-1", "", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=120, what="tts_audio_end")
            assert not rec.of("error"), [e["error"].message for e in rec.of("error")]
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert len(ext._req.wire_ids) >= 3  # cut into segments of about segment_max_ms
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

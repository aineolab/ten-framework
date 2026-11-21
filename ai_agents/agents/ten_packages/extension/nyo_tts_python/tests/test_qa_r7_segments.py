"""QA round 7: a TEN request without text_input_end (voice-assistant-with-turn-detection) is spoken.

* sentence-boundary hint: a sentence-final increment goes out with one trailing space (engines split at once);
* idle segmentation (segment_idle_ms): no new chunk for that long -> end:true for the open NRSP request, the next
  chunk opens `<id>~2`; one tts_audio_start, one timeline, summed duration, tts_audio_end only on text_input_end,
  flush or error; the watchdog never fires while the request just waits for the LLM;
* after a FATAL the first chunk (final or not) ends the request at once;
* a send that crosses an announced idle close hands over to the resend instead of failing the request.
Fake NRSP server (in-process)."""
import asyncio

import websockets
from ten_ai_base.message import TTSAudioEndReason

from nyo_tts_python.extension import ends_sentence

from .helpers import harness, run, text_input, wait_for

FAKE_CHUNK_MS = 85


def _wire(srv):
    return [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]


def test_ends_sentence():
    for yes in ("Salamat po.", 'Talaga?"', "Ano?)", "Oo!", "好。", "x…", "nyolab."):
        assert ends_sentence(yes), yes
    # never after a digit ("₱3." of "₱3.50"), never without final punctuation or with whitespace after it
    for no in ("Ang presyo ay ₱3.", "Bilang 2!", "Kumusta,", "Salamat po. ", "Salamat po.\n", "walang tuldok", ""):
        assert not ends_sentence(no), no


def test_qa_r7_segments_one_tts_request_one_timeline():
    async def go():
        async with harness(segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Kumusta po.", end=False))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1 ended and spoken")
            await ext.request_tts(text_input("r", " Paalam na.", end=False))
            await wait_for(lambda: "r~2" in ext._req.seg_done, timeout=3, what="segment 2 ended and spoken")
            assert not rec.of("audio_end") and ext._req.open
            await ext.request_tts(text_input("r", "", end=True))  # the end: the last segment is already spoken
            await wait_for(lambda: rec.of("audio_end"), timeout=2, what="tts_audio_end")
            assert _wire(srv) == [("r", "Kumusta po.", False), ("r", "", True),
                                  ("r~2", " Paalam na.", False), ("r~2", "", True)], _wire(srv)
            names = rec.names()
            assert names.count("audio_start") == 1 and names.count("audio_end") == 1 and names.count("ttfb") == 1
            assert names.count("finish") == 1 and not rec.of("error")
            end = rec.of("audio_end")[0]
            # (11 + 10) stripped chars * 60 ms in the fake, both segments in one request
            assert end["reason"] == TTSAudioEndReason.REQUEST_END and end["duration_ms"] == 21 * 60, end
            ts = [a["ts"] for a in rec.of("audio")]
            assert ts[0] == 0 and ts == sorted(ts) and len(set(ts)) == len(ts), ts
            # segment 2 continues where segment 1 ends (660 ms), not at 0
            assert 660 in ts
            assert all(b - a <= FAKE_CHUNK_MS for a, b in zip(ts, ts[1:])), ts

    run(go())


def test_qa_r7_end_inside_a_segment_ends_it_and_the_request():
    async def go():
        async with harness(segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una.", end=False))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1")
            await ext.request_tts(text_input("r", " Ikalawa", end=False))
            await ext.request_tts(text_input("r", " at huli.", end=True))  # before the idle timer
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            assert _wire(srv) == [("r", "Una.", False), ("r", "", True), ("r~2", " Ikalawa", False),
                                  ("r~2", " at huli.", True)], _wire(srv)
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert rec.names().count("audio_start") == 1

    run(go())


def test_qa_r7_slow_llm_never_times_the_request_out():
    async def go():
        async with harness(segment_idle_ms=100, request_timeout_ms=300) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Sandali lang po.", end=False))
            await asyncio.sleep(1.2)  # 4x request_timeout_ms without a chunk: the LLM is slow, nothing is owed
            assert ext._req.open and not rec.of("audio_end") and not rec.of("error"), rec.names()
            assert "r" in ext._req.seg_done
            await ext.request_tts(text_input("r", " Heto na.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END

    run(go())


def test_qa_r7_segment_idle_off_keeps_one_nrsp_request():
    async def go():
        async with harness(segment_idle_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Kumusta po.", end=False))
            await asyncio.sleep(0.4)
            await ext.request_tts(text_input("r", " Paalam.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            # one NRSP request; the pause after the sentence end got the delayed boundary hint (boundary_hint_ms)
            assert _wire(srv) == [("r", "Kumusta po.", False), ("r", " ", False), ("r", " Paalam.", True)]

    run(go())


def test_qa_r7_flush_mid_segment_cancels_it_and_drops_later_chunks():
    async def go():
        async with harness(key="nyo-test-key+rt", segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Isa.", end=False))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1")
            await ext.request_tts(text_input("r", " Mahabang pangungusap na ito po.", end=False))
            n0 = len(rec.of("audio"))
            await wait_for(lambda: len(rec.of("audio")) > n0 + 1, timeout=3, what="segment 2 playing")
            await ext.cancel_tts()  # tts_flush mid segment 2
            ends = rec.of("audio_end")
            assert [e["reason"] for e in ends] == [TTSAudioEndReason.INTERRUPTED]
            await asyncio.sleep(0.05)
            assert [m["request_id"] for m in srv.received("flush")] == ["r~2"]
            n_audio = len(rec.of("audio"))
            await ext.request_tts(text_input("r", " Late chunk.", end=False))  # the aborted LLM's next sentence
            await asyncio.sleep(0.5)
            assert len(rec.of("audio")) == n_audio and len(rec.of("audio_end")) == 1
            assert [w[0] for w in _wire(srv)] == ["r", "r", "r~2"]  # nothing after the flush
            assert rec.names().count("audio_start") == 1

    run(go())


def test_qa_r7_error_on_segment_2_costs_that_segment_only():
    """(round 8) a transient engine_error on a segment: one NON_FATAL error, the TEN request goes on."""
    async def go():
        async with harness(key="nyo-test-key+ttserr", segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una.", end=False))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1")
            await ext.request_tts(text_input("r", " ERROR_TRIGGER dito.", end=False))
            await wait_for(lambda: rec.of("error"), timeout=3, what="NON_FATAL error")
            errs = rec.of("error")
            assert len(errs) == 1 and errs[0]["request_id"] == "r"
            assert errs[0]["error"].vendor_info.code == "engine_error" and int(errs[0]["error"].code) != -1000
            assert ext._req.open and not rec.of("audio_end")
            await asyncio.sleep(0.4)  # no idle end for the failed segment, nothing else emitted
            assert ("r~2", "", True) not in _wire(srv) and not rec.of("audio_end")
            await ext.request_tts(text_input("r", " Ikatlo.", end=True))  # the next chunk: a new segment
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            end = rec.of("audio_end")[0]
            assert end["request_id"] == "r" and end["reason"] == TTSAudioEndReason.REQUEST_END
            assert end["duration_ms"] == (4 + 7) * 60  # segments 1 and 3
            assert [w for w in _wire(srv) if w[0] == "r~3"] == [("r~3", " Ikatlo.", True)]
            assert rec.names().count("audio_start") == 1 and rec.names().count("finish") == 1
            assert len(rec.of("error")) == 1 and not rec.of("finish_error")

    run(go())


def test_qa_r7_only_segment_fails_without_audio_ends_with_error():
    async def go():
        async with harness(key="nyo-test-key+ttserr") as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "ERROR_TRIGGER lang.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.ERROR
            assert len(rec.of("error")) == 1 and rec.names().count("finish") == 1

    run(go())


def test_qa_r7_stale_request_on_a_segment_ends_the_ten_request():
    async def go():
        async with harness(segment_idle_ms=150) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una.", end=False))
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="segment 1")
            ext._req.seg_no = 0  # force the next segment onto the finished wire id "r" (the gateway: stale_request)
            await ext.request_tts(text_input("r", " Ikalawa.", end=False))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            end = rec.of("audio_end")[0]
            assert end["request_id"] == "r" and end["reason"] == TTSAudioEndReason.ERROR
            assert rec.of("finish_error")[0]["error"].vendor_info.code == "stale_request"

    run(go())


def test_qa_r7_after_fatal_first_non_final_chunk_ends_the_request():
    async def go():
        async with harness() as (srv, ext, rec):
            ext._fatal = ext._fatal_sent = True  # a FATAL was reported earlier
            await ext.request_tts(text_input("r", "Kumusta po.", end=False))
            assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.ERROR]
            assert [f["request_id"] for f in rec.of("finish")] == ["r"]  # base loop released
            await ext.request_tts(text_input("r", " Paalam.", end=False))  # late chunk: nothing more
            assert len(rec.of("audio_end")) == 1 and not rec.of("error") and not srv.received("text")

    run(go())


def test_qa_r7_send_crossing_idle_close_hands_over_to_the_resend():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext._ensure_connected()
            old = ext.client
            real_send = old.send_text

            async def crossing(rid, text, end):
                # the gateway announced idle_timeout (reader -> _maybe_resend) and closed before this write
                assert ext._maybe_resend(old)
                raise websockets.exceptions.ConnectionClosedError(None, None)

            old.send_text = crossing
            await ext.request_tts(text_input("r", "Kumusta po.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END, rec.names()
            assert not rec.of("error") and not rec.of("finish_error")
            assert ext.client is not old and _wire(srv) == [("r", "Kumusta po.", True)]
            old.send_text = real_send

    run(go())

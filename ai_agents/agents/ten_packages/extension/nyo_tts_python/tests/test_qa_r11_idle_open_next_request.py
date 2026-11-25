"""QA round 11: an idle-open TEN request ends once the base class holds the next request.

A TEN request without text_input_end stays open after all its segments were spoken (turn-detection graphs reset it
by flush). The base class (tts2 _process_input_queue) buffers chunks of another request id in _pending_messages
until finish_request of the current one, so an idle-open request that is never ended blocked the next request for
good (tts_guarder test_append_input_without_text_input_end passed only through its 5 s frame timeout). Now, as soon
as the request is idle-open (every segment's audio_end received, nothing owed, no chunk arriving, no timer that
would send more) AND another request id is pending in the base, it ends with tts_audio_end(REQUEST_END) and
finish_request. Never while audio is owed or a chunk of it is arriving. Fake NRSP server (in-process)."""
import asyncio

from ten_ai_base.message import TTSAudioEndReason
from ten_ai_base.tts2 import RequestState

from .helpers import harness, run, text_input, wait_for


def _wire(srv):
    return [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]


def _as_base(ext, rid):
    ext._processing_request_id = rid  # as the base input loop has it
    ext.request_states[rid] = RequestState.PROCESSING


def _pend(ext, rid, text="Susunod po."):
    ext._pending_messages.setdefault(rid, []).append(text_input(rid, text))


def test_qa_r11_idle_open_request_ends_when_the_next_one_is_already_buffered():
    async def go():
        async with harness(segment_idle_ms=150, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            await ext.request_tts(text_input("r1", "Kumusta po."))
            _pend(ext, "r2")  # buffered by the base while r1 is processing
            # r1's segment is still owed: it must not end before its audio_end
            await asyncio.sleep(0.05)
            assert not rec.of("audio_end")
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end of r1")
            end = rec.of("audio_end")[0]
            assert end["request_id"] == "r1" and end["reason"] == TTSAudioEndReason.REQUEST_END, end
            assert end["duration_ms"] == 11 * 60 and end["interval_ms"] > 0, end  # every frame, normal interval
            assert [f["request_id"] for f in rec.of("finish")] == ["r1"]
            assert rec.of("finish")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            names = rec.names()
            assert names.index("audio_end") > max(i for i, n in enumerate(names) if n == "audio")
            assert not rec.of("error")
            assert _wire(srv) == [("r1", "Kumusta po.", False), ("r1", "", True)], _wire(srv)
            assert ext._req.end_reason == TTSAudioEndReason.REQUEST_END and not ext._req.open

    run(go())


def test_qa_r11_watcher_ends_it_when_the_next_request_arrives_later():
    async def go():
        async with harness(segment_idle_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            await ext.request_tts(text_input("r1", "Una po."))
            await wait_for(lambda: "r1" in ext._req.seg_done, timeout=3, what="segment spoken")
            await asyncio.sleep(0.3)
            # idle-open with nothing buffered: it stays open (the graph may still send more or flush)
            assert ext._req.open and not rec.of("audio_end")
            assert ext._req.idle_watch is not None and not ext._req.idle_watch.done()
            _pend(ext, "r2")
            await wait_for(lambda: rec.of("audio_end"), timeout=1, what="tts_audio_end once r2 is buffered")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert rec.of("audio_end")[0]["duration_ms"] == 7 * 60
            assert ext._req.idle_watch is None

    run(go())


def test_qa_r11_never_ends_while_a_segment_is_owed_or_a_chunk_is_arriving():
    async def go():
        async with harness(segment_idle_ms=150, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            await ext.request_tts(text_input("r1", "Una po."))
            await wait_for(lambda: "r1" in ext._req.seg_done, timeout=3, what="segment 1 spoken")
            # a new chunk of r1 cancels the watcher and opens segment ~2; r2 gets buffered meanwhile
            await ext.request_tts(text_input("r1", " Ikalawa po."))
            assert ext._req.idle_watch is None
            _pend(ext, "r2")
            await asyncio.sleep(0.1)
            assert not rec.of("audio_end"), "ended while segment ~2 was owed"
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end after segment ~2")
            end = rec.of("audio_end")[0]
            assert end["reason"] == TTSAudioEndReason.REQUEST_END and end["duration_ms"] == (7 + 11) * 60, end
            assert _wire(srv) == [("r1", "Una po.", False), ("r1", "", True),
                                  ("r1~2", " Ikalawa po.", False), ("r1~2", "", True)], _wire(srv)

    run(go())


def test_qa_r11_a_queued_chunk_of_the_request_keeps_it_open():
    async def go():
        async with harness(segment_idle_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            await ext.request_tts(text_input("r1", "Una po."))
            await wait_for(lambda: "r1" in ext._req.seg_done, timeout=3, what="segment spoken")
            await ext.input_queue.put(text_input("r1", " Pa."))  # about to reach request_tts
            _pend(ext, "r2")
            await asyncio.sleep(0.25)
            assert ext._req.open and not rec.of("audio_end")
            t = await ext.input_queue.get()
            await ext.request_tts(t)  # the base loop delivers it: segment ~2
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end after segment ~2")
            assert rec.of("audio_end")[0]["duration_ms"] == (7 + 3) * 60
            assert ("r1~2", " Pa.", False) in _wire(srv)

    run(go())


def test_qa_r11_a_segment_ended_by_segment_max_ends_without_waiting_for_its_void_timers():
    async def go():
        # segment_max_ms ends the segment at the sentence end; its idle end and hint (1 s) are still armed but void
        async with harness(segment_idle_ms=1000, boundary_hint_ms=1000, segment_max_ms=5) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            _pend(ext, "r2")
            await ext.request_tts(text_input("r1", "Una"))
            await asyncio.sleep(0.02)
            await ext.request_tts(text_input("r1", " po."))
            assert ext._req.seg_ended
            t0 = asyncio.get_event_loop().time()
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            assert asyncio.get_event_loop().time() - t0 < 0.9
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert rec.of("audio_end")[0]["duration_ms"] == 6 * 60
            await asyncio.sleep(0.1)
            assert _wire(srv) == [("r1", "Una", False), ("r1", " po.", False), ("r1", "", True)], _wire(srv)

    run(go())


def test_qa_r11_flush_and_stop_cancel_the_watcher():
    async def go():
        async with harness(segment_idle_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            await ext.request_tts(text_input("r1", "Una po."))
            await wait_for(lambda: ext._req.idle_watch is not None, timeout=3, what="idle-open watcher")
            watch = ext._req.idle_watch
            await ext.cancel_tts()  # tts_flush
            await asyncio.sleep(0)
            assert watch.done() and ext._req.idle_watch is None
            _pend(ext, "r2")
            await asyncio.sleep(0.2)
            assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.INTERRUPTED]

        async with harness(segment_idle_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            await ext.request_tts(text_input("r1", "Una po."))
            await wait_for(lambda: ext._req.idle_watch is not None, timeout=3, what="idle-open watcher")
            watch = ext._req.idle_watch
            await ext.on_stop(ext.ten_env)
            assert watch.done()
            assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.INTERRUPTED]

    run(go())


def test_qa_r11_request_with_text_input_end_is_unchanged():
    async def go():
        async with harness(segment_idle_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            _pend(ext, "r2")
            await ext.request_tts(text_input("r1", "Una po.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            await asyncio.sleep(0.2)
            assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.REQUEST_END]
            assert rec.names().count("finish") == 1
            assert _wire(srv) == [("r1", "Una po.", True)], _wire(srv)

    run(go())


def test_qa_r11_idle_open_end_ms_ends_the_last_request():
    """idle_open_end_ms (opt-in, default 0): an idle-open request that no next request and no flush follow (the
    tts_guarder's last request) ends with REQUEST_END after that long without a chunk. Off, it stays open."""
    async def go():
        async with harness(segment_idle_ms=100, boundary_hint_ms=0, idle_open_end_ms=300) as (srv, ext, rec):
            await ext._ensure_connected()
            _as_base(ext, "r1")
            await ext.request_tts(text_input("r1", "Una po."))
            await wait_for(lambda: "r1" in ext._req.seg_done, timeout=3, what="segment spoken")
            await asyncio.sleep(0.1)
            assert not rec.of("audio_end")
            await wait_for(lambda: rec.of("audio_end"), timeout=1, what="tts_audio_end after idle_open_end_ms")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert rec.of("audio_end")[0]["duration_ms"] == 7 * 60

        async with harness(segment_idle_ms=100, boundary_hint_ms=0) as (srv, ext, rec):  # default: off
            assert ext.config.idle_open_end_ms == 0
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "Una po."))
            await wait_for(lambda: "r1" in ext._req.seg_done, timeout=3, what="segment spoken")
            await asyncio.sleep(0.5)
            assert ext._req.open and not rec.of("audio_end")

    run(go())

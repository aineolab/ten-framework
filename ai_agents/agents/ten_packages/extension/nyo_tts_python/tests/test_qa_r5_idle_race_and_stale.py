"""QA round 5.

idle_timeout race: the gateway's idle close crosses the first text of a new request (the gateway never saw it).
The request must be resent once on a fresh socket and complete, not fail with connection_lost.

stale_request for the OPEN request (an id reused after a tts_flush on the same socket): end it at once with ERROR and
a NON_FATAL stale_request error instead of waiting request_timeout_ms for the watchdog.
"""
import asyncio
import time

import fake_nrsp_server
from ten_ai_base.message import ModuleErrorCode, TTSAudioEndReason

from .helpers import harness, run, text_input, wait_for

NON_FATAL = int(ModuleErrorCode.NON_FATAL_ERROR.value)


def test_qa_r5_idle_close_racing_first_text_resends_on_fresh_socket():
    async def go():
        real_on_text = fake_nrsp_server._TtsSession._on_text
        raced = []

        async def on_text(self, msg):
            if not raced:  # the idle fatal left the gateway before this text arrived: it is never processed
                raced.append(msg)
                await self.srv._send(self.ws, fake_nrsp_server.error_obj("idle_timeout", "no client message", True))
                await self.ws.close(1001)
                return
            await real_on_text(self, msg)

        fake_nrsp_server._TtsSession._on_text = on_text
        try:
            async with harness() as (srv, ext, rec):
                await ext._ensure_connected()
                await ext.request_tts(text_input("r1", "Mabuhay ka.", end=True))
                await wait_for(lambda: rec.of("audio_end"), timeout=5, what="audio_end")
                end = rec.of("audio_end")[-1]
                assert end["reason"] == TTSAudioEndReason.REQUEST_END, (end, rec.of("error"), rec.of("finish_error"))
                assert end["duration_ms"] > 0
                assert not rec.of("error") and not rec.of("finish_error")
                tts = [c for c in srv.connections if c["path"] == "/v1/tts"]
                assert len(tts) == 2, f"expected a fresh socket for the resend, got {len(tts)}"
                texts = srv.received("text")
                # once on the closed socket (never processed), once on the fresh one: resent exactly once
                assert [t["request_id"] for t in texts] == ["r1", "r1"] and texts[1]["end"] is True
                assert len(rec.of("audio_start")) == 1
        finally:
            fake_nrsp_server._TtsSession._on_text = real_on_text

    run(go())


def test_qa_r5_idle_close_after_reply_still_fails_request():
    """Not the race: the request already got audio, so the close fails it as before (no resend)."""
    async def go():
        async with harness(server_kw={"first_audio_delay_ms": 5, "ms_per_char": 200}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "Mabuhay ka sa lahat ng araw.", end=True))
            await wait_for(lambda: rec.of("audio"), timeout=5, what="audio")
            sock = ext.client
            await ext._on_error({"type": "error", "code": "idle_timeout", "fatal": True}, sock)
            await sock.websocket.close()
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="audio_end")
            # round 8: the segment is lost (one NON_FATAL connection_lost); audio was produced -> REQUEST_END
            assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.REQUEST_END
            assert [e["error"].vendor_info.code for e in rec.of("error")] == ["connection_lost"]
            assert len(srv.received("text")) == 1  # not resent

    run(go())


def test_qa_r5_stale_request_for_open_request_ends_it_now():
    async def go():
        async with harness(request_timeout_ms=10000) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "Mabuhay.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="first audio_end")
            await ext.request_tts(text_input("x", "Oo.", end=True))
            await wait_for(lambda: len(rec.of("audio_end")) == 2, timeout=5, what="audio_end of x")
            ext._finished.clear()  # what a tts_flush does: the graph may reuse the id
            rec.events.clear()
            t0 = time.monotonic()
            await ext.request_tts(text_input("a", "Muli.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="audio_end of the refused request")
            assert time.monotonic() - t0 < 2
            end = rec.of("audio_end")[-1]
            assert end["request_id"] == "a" and end["reason"] == TTSAudioEndReason.ERROR
            errs = rec.of("finish_error")
            assert errs and int(errs[0]["error"].code) == NON_FATAL
            assert errs[0]["error"].vendor_info.code == "stale_request"
            assert not ext._fatal
            # the extension keeps working
            rec.events.clear()
            await ext.request_tts(text_input("b", "Salamat.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="audio_end of b")
            assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.REQUEST_END

    run(go())


def test_qa_r5_stale_request_for_finished_id_is_ignored():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "Mabuhay.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5)
            await ext.request_tts(text_input("b", "Salamat po.", end=False))
            await ext._on_error({"type": "error", "code": "stale_request", "fatal": False, "request_id": "a"},
                                ext.client)
            await ext.request_tts(text_input("b", "", end=True))
            await wait_for(lambda: len(rec.of("audio_end")) == 2, timeout=5)
            assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.REQUEST_END
            assert not rec.of("error") and not rec.of("finish_error")

    run(go())

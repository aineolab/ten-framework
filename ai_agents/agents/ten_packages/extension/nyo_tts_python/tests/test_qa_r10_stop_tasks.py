"""QA round 10 (fixed): on_stop cancels and awaits the extension's background tasks (segment timers, resends,
background connects) with a short bound, so none outlives the extension."""
import asyncio
import time

from ten_ai_base.message import TTSAudioEndReason
from ten_ai_base.tts2 import RequestState

from .helpers import harness, run, text_input


def test_qa_r10_on_stop_cancels_background_tasks():
    async def go():
        async with harness(segment_idle_ms=5000, boundary_hint_ms=5000) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una po."))  # arms the idle-end and hint timers (5 s)
            ext._processing_request_id = "r"  # as the base input loop has it
            ext.request_states["r"] = RequestState.PROCESSING
            stuck = ext._spawn(asyncio.sleep(30))  # e.g. a background connect held by the gateway
            assert len([t for t in ext._bg_tasks if not t.done()]) >= 3
            t0 = time.monotonic()
            await ext.on_stop(ext.ten_env)
            assert time.monotonic() - t0 < 1.5
            assert stuck.done() and not [t for t in ext._bg_tasks if not t.done()]
            assert ext.client is None
            # the open request was ended by the base flush in on_stop: one tts_audio_end
            assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.INTERRUPTED]

    run(go())

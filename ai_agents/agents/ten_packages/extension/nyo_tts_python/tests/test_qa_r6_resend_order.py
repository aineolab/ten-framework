"""QA round 6 (fixed): idle_timeout resend vs the next chunk of the same request. The base input loop may deliver
the next chunk before the spawned resend task has taken req.send_lock; that chunk must wait for the resend, so the
fresh socket sees c1, c2 exactly once and in order (it used to see c2, c1, c2 and answer stale_request)."""
import asyncio

from ten_ai_base.message import TTSAudioEndReason

from .helpers import harness, run, text_input, wait_for


def test_qa_r6_chunk_racing_resend_task_keeps_order():
    async def go():
        async with harness(server_kw={"first_audio_delay_ms": 400}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "Unang pangungusap. "))
            sock = ext.client
            # the gateway's idle close crossed c1 (no reply yet); the input loop is ready with c2 in the same tick
            await ext._on_error({"type": "error", "code": "idle_timeout", "fatal": True}, sock)
            await ext.request_tts(text_input("r1", "Ikalawa.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=8, what="audio_end")
            texts = [t["text"] for t in srv.received("text")]
            # c1 on the idle socket, then c1, c2 once on the fresh one
            assert texts == ["Unang pangungusap. ", "Unang pangungusap. ", "Ikalawa."], texts  # verbatim
            end = rec.of("audio_end")[-1]
            assert end["reason"] == TTSAudioEndReason.REQUEST_END, (end, rec.of("finish_error"))
            assert not rec.of("error") and not rec.of("finish_error")

    run(go())

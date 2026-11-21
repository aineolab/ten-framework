"""QA round 2: after a reconnectable fatal error (engine_restarting) ends request A, the base class releases
request B right away (finish_request). B must be synthesised on a fresh socket, not written to the dying one."""
import asyncio

from ten_ai_base.message import TTSAudioEndReason

from .helpers import harness, run, text_input, wait_for


def test_qa_r2_request_released_by_the_restart_error_must_succeed():
    async def go():
        async with harness(key="k+restart") as (srv, ext, rec):
            ext._reconnect_delay = lambda attempt: 0.01
            await ext._ensure_connected()
            orig_finish = ext.finish_request
            tasks = []

            async def finish(request_id, reason=None, error=None):
                await orig_finish(request_id, reason=reason, error=error)
                if request_id == "a":  # the base puts the buffered next request on the queue here
                    tasks.append(asyncio.create_task(ext.request_tts(text_input("b", "mabuhay ", end=True))))
                await asyncio.sleep(0)  # TEN's send_data yields to the loop

            ext.finish_request = finish
            await ext.request_tts(text_input("a", "kumusta ", end=True))
            await wait_for(lambda: len(rec.of("audio_end")) == 2, timeout=10, what="a and b end")
            await asyncio.gather(*tasks)
            ends = {e["request_id"]: e["reason"] for e in rec.of("audio_end")}
            b_err = [f["error"] for f in rec.of("finish") if f["request_id"] == "b" and f["error"]]
            assert ends["a"] == TTSAudioEndReason.ERROR
            assert ends["b"] == TTSAudioEndReason.REQUEST_END and not b_err, (
                f"b (released by a's engine_restarting) failed: reason={ends['b']}, "
                f"errors={[(e.vendor_info.code, e.message) for e in b_err]}, connections={len(srv.connections)}")

    run(go())

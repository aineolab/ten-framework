"""QA round 2: a flush during the request's own first connect (the gateway holds opens <=15 s during a swap)
ends the request at once, but request_tts keeps the base input loop blocked until the handshake finishes,
so the next request cannot start. Fixed: the first connect is raced against the flush."""
import asyncio
import time

from fake_nrsp_server import FakeNRSPServer
from ten_ai_base.message import TTSAudioEndReason

from .helpers import make_extension, run, text_input, wait_for


def test_qa_r2_flush_during_slow_handshake_releases_the_input_loop():
    async def go():
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url)
            ext._preflight_ok = True
            from nyo_tts_python import tts_client
            real_connect = tts_client.websockets.connect

            async def slow_connect(*a, **k):  # gateway queueing the open during an engine swap
                await asyncio.sleep(3)
                return await real_connect(*a, **k)

            tts_client.websockets.connect = slow_connect
            try:
                task = asyncio.create_task(ext.request_tts(text_input("a", "kumusta ")))
                await wait_for(lambda: "connecting" in rec.names())
                await ext.cancel_tts()
                t0 = time.monotonic()
                await task
                held = time.monotonic() - t0
                # the handshake completes in the background: the next request finds the socket ready
                await wait_for(lambda: ext.client is not None and ext.client.is_connected(), timeout=6)
                assert [e["reason"] for e in rec.of("audio_end")] == [TTSAudioEndReason.INTERRUPTED]
                assert rec.of("error") == []
            finally:
                tts_client.websockets.connect = real_connect
                ext._stopped = True
                if ext.client:
                    await ext.client.stop()
            assert srv.received("text") == []
            assert held < 0.5, f"input loop held {held:.1f} s after the flush (next request blocked)"

    run(go())

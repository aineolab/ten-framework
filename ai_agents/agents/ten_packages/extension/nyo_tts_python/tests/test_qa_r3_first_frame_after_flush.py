"""QA round 3: _on_audio awaits send_tts_audio_start (a ten_env.send_data round trip) and then sends the TTFB metric and
the first PCM frame without re-checking req.open. A tts_flush that lands during that await emits
tts_audio_end(INTERRUPTED) and the frame still goes out after it. Fixed: re-checked after each await."""
import asyncio

from fake_nrsp_server import FakeNRSPServer

from .helpers import make_extension, run, text_input, wait_for


def test_qa_r3_first_frame_sent_after_interrupted_audio_end():
    async def go():
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url)
            orig = ext.send_tts_audio_start
            flushed = []

            async def slow_start(request_id, turn_id=-1, extra_metadata=None):
                await orig(request_id)
                if not flushed:
                    flushed.append(1)
                    asyncio.ensure_future(ext.cancel_tts())  # tts_flush handled while send_data is in flight
                    await asyncio.sleep(0.01)

            ext.send_tts_audio_start = slow_start
            try:
                await ext.request_tts(text_input("a", "kumusta mundo", end=True))
                await wait_for(lambda: rec.of("audio_end"), timeout=5)
                await asyncio.sleep(0.3)
                names = rec.names()
                end_at = names.index("audio_end")
                late = [n for n in names[end_at + 1:] if n in ("audio", "ttfb")]
                assert not late, f"after tts_audio_end(INTERRUPTED): {late}; order {names}"
            finally:
                ext._stopped = True
                if ext.client:
                    await ext.client.stop()

    run(go())

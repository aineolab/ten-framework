"""QA round 3: update_configs during request a is deferred; request_tts(b) applies it first (_apply_config ->
_recycle_socket -> stop old socket + full reconnect) BEFORE _begin_request(b). A tts_flush that lands during that
reconnect finds self._req = a (closed), so cancel_tts does nothing; request_tts then begins b anyway and its text
and audio go out after tts_flush_end. The reconnect here is also not raced against the flush, so the base input
loop stays blocked for the whole handshake. Fixed: b begins before the config is applied and its fresh socket is
opened by _connect_for_request, raced against the flush."""
import asyncio
import time

from fake_nrsp_server import FakeNRSPServer

from .helpers import make_extension, run, text_input, wait_for


def test_qa_r3_flush_during_deferred_config_reconnect_leaks_b():
    async def go():
        async with FakeNRSPServer() as srv:
            ext, rec = make_extension(srv.url)
            from nyo_tts_python import tts_client
            real_connect = tts_client.websockets.connect
            try:
                await ext.request_tts(text_input("a", "kumusta "))           # a open on socket 1
                await ext.update_configs({"params": {"voice": "maria"}})      # deferred (a is open)
                assert ext._pending_config is not None
                await ext.request_tts(text_input("a", "mundo", end=True))
                await wait_for(lambda: rec.of("audio_end"), timeout=5, what="a ended")

                async def slow_connect(*a, **k):  # WAN handshake / gateway queueing the open
                    await asyncio.sleep(1.5)
                    return await real_connect(*a, **k)

                tts_client.websockets.connect = slow_connect
                task = asyncio.create_task(ext.request_tts(text_input("b", "kumusta mundo", end=True)))
                await wait_for(lambda: rec.names().count("connecting") >= 2, what="recycle started")
                await ext.cancel_tts()                      # base: tts_flush for b (state PROCESSING)
                t0 = time.monotonic()
                n_after_flush = len(rec.events)
                await task
                held = time.monotonic() - t0
                await asyncio.sleep(1.0)
                after = rec.events[n_after_flush:]
                b_audio = [e for n, e in after if n == "audio"]
                b_texts = [p for p in srv.received("text") if p.get("request_id") == "b"]
                assert held < 0.5, f"input loop held {held:.1f} s after the flush"
                assert [e["reason"] for e in rec.of("audio_end")][-1].name == "INTERRUPTED"
                assert ext.config.voice == "maria"  # the deferred update is applied all the same
                assert not b_texts and not b_audio, (
                    f"flushed request b still reached the gateway ({len(b_texts)} text msgs) and produced "
                    f"{len(b_audio)} audio frames after tts_flush; input loop held {held:.1f} s")
            finally:
                tts_client.websockets.connect = real_connect
                ext._stopped = True
                ext._cancel_reconnect_task()
                if ext.client:
                    await ext.client.stop()

    run(go())

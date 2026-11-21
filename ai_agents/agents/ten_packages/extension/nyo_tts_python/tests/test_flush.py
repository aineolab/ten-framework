"""cancel_tts: NRSP flush + stale request_seq drop, never a reconnect."""
from ten_ai_base.message import TTSAudioEndReason

from .helpers import harness, run, text_input, wait_for


def test_cancel_sends_flush_and_interrupted_audio_end_without_reconnect():
    async def go():
        async with harness(server_kw={"chunk_interval_ms": 40}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "a long sentence that keeps streaming for a while "))
            await wait_for(lambda: "audio" in rec.names(), what="first audio")
            await ext.cancel_tts()
            assert rec.of("audio_end")[-1]["reason"] == TTSAudioEndReason.INTERRUPTED
            await wait_for(lambda: srv.received("flush"), what="flush at server")
            assert srv.received("flush") == [{"type": "flush", "request_id": "r1"}]
            assert len(srv.connections) == 1, "cancel must not reconnect"
            assert "finish" not in rec.names(), "base class owns finish_request after a flush"

    run(go())


def test_audio_after_cancel_is_dropped_by_stale_seq():
    async def go():
        async with harness(key="k+stale3", server_kw={"chunk_interval_ms": 40}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "streaming audio that will be cut off midway " * 2))
            await wait_for(lambda: "audio" in rec.names())
            await ext.cancel_tts()
            await wait_for(lambda: any(k == "tx:audio_end" for k, _ in srv.log), what="server flush ack")
            # 3 misbehaving frames with the old request_seq follow the server's flush ack
            await wait_for(lambda: ext.client.stale_frames_dropped >= 1 or
                           sum(1 for k, _ in srv.log if k == "tx:audio") > 0, what="stale frames")
            n_audio = rec.names().count("audio")
            import asyncio
            await asyncio.sleep(0.3)
            assert rec.names().count("audio") == n_audio, "frames after cancel leaked into the graph"
            assert ext.client.stale_frames_dropped >= 3

    run(go())


def test_new_request_works_right_after_cancel_on_same_socket():
    async def go():
        async with harness(key="k+stale3", server_kw={"chunk_interval_ms": 30}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "first sentence is long enough to be interrupted " * 2))
            await wait_for(lambda: "audio" in rec.names())
            await ext.cancel_tts()
            rec.events.clear()
            await ext.request_tts(text_input("r2", "second ", end=False))
            await ext.request_tts(text_input("r2", "", end=True))
            await wait_for(lambda: "finish" in rec.names(), what="second request finish")
            assert rec.of("audio_start")[0]["request_id"] == "r2"
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert len(srv.connections) == 1
            # only r2's audio was forwarded, with r2's own seq
            assert rec.of("audio_end")[0]["duration_ms"] == 6 * 60

    run(go())


def test_cancel_without_open_request_is_a_noop():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext.cancel_tts()
            assert rec.events == []

    run(go())


def test_cancel_before_any_audio_still_closes_the_start_end_pair():
    async def go():
        async with harness(server_kw={"first_audio_delay_ms": 500}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "slow to start"))
            await ext.cancel_tts()
            assert rec.names()[-2:] == ["audio_start", "audio_end"]
            assert rec.of("audio_end")[0]["duration_ms"] == 0

    run(go())


def test_cancel_latency_is_local_and_fast():
    async def go():
        import time
        async with harness(server_kw={"chunk_interval_ms": 40}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r1", "x" * 200))
            await wait_for(lambda: "audio" in rec.names())
            t0 = time.perf_counter()
            await ext.cancel_tts()
            assert (time.perf_counter() - t0) * 1000 < 50  # no reconnect, no round trip

    run(go())

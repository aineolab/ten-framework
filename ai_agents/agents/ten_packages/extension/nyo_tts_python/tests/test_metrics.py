from .helpers import harness, run, text_input, wait_for


def test_ttfb_is_reported_once_per_request_with_model():
    async def go():
        async with harness(model="halohalo/tts-ceb", server_kw={"first_audio_delay_ms": 80}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "metrics please ", end=True))
            await wait_for(lambda: "finish" in rec.names())
            t = rec.of("ttfb")
            assert len(t) == 1
            assert 70 <= t[0]["ms"] < 1000  # includes the fake's 80 ms think time
            assert t[0]["extra"] == {"model": "halohalo/tts-ceb"}
            names = rec.names()
            assert names.index("audio_start") < names.index("ttfb") < names.index("audio")

    run(go())


def test_ttfb_counts_from_first_text_not_from_connect():
    async def go():
        import asyncio
        async with harness(server_kw={"first_audio_delay_ms": 10}) as (srv, ext, rec):
            await ext._ensure_connected()
            await asyncio.sleep(0.3)  # idle connection must not inflate TTFB
            await ext.request_tts(text_input("r", "hi ", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert rec.of("ttfb")[0]["ms"] < 250

    run(go())


def test_usage_metrics_sent_on_normal_completion_only():
    async def go():
        async with harness(server_kw={"chunk_interval_ms": 40}) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "short ", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert len(rec.of("usage")) == 1
            rec.events.clear()
            await ext.request_tts(text_input("b", "y" * 200))
            await wait_for(lambda: "audio" in rec.names())
            await ext.cancel_tts()
            assert rec.of("usage") == []

    run(go())


def test_connect_delay_metric_reported():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext._ensure_connected()
            ext.metrics_connect_delay.assert_awaited_once()
            assert ext.metrics_connect_delay.await_args.args[0] >= 0

    run(go())


def test_client_counters_track_frames():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext.request_tts(text_input("a", "counting frames ", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert ext.client.frames_received == len(rec.of("audio")) > 0
            assert ext.client.recv_ns_total > 0

    run(go())


def test_server_side_ttfb_ttlw_are_exported_when_present():
    from nyo_tts_python.extension import _Request
    from .helpers import make_extension

    async def go():
        ext, rec = make_extension()
        ext._req = _Request("r")
        await ext._on_audio_end("r", "end", 0, {"ttfb_ms": 96.4, "ttlw_ms": 306.5})
        assert rec.of("metrics")[0]["metrics"] == {"server_ttfb_ms": 96.4, "server_ttlw_ms": 306.5,
                                                   "model": ext.config.model}
        ext._req = _Request("q")
        await ext._on_audio_end("q", "end", 0, {"model": "halohalo/tts-fil"})  # old servers: model only
        assert rec.of("metrics")[1]["metrics"] == {"model": "halohalo/tts-fil"}

    run(go())


def test_server_usage_is_forwarded_as_metrics():
    """ten review F5: the server's `usage` (after audio_end) reaches TEN metrics."""
    from .helpers import harness, text_input, wait_for

    async def go():
        async with harness() as (srv, ext, rec):
            await ext.request_tts(text_input("u1", "usage please", end=True))
            await wait_for(lambda: any("server_audio_ms" in m["metrics"] for m in rec.of("metrics")),
                           what="usage metrics")
            m = next(m["metrics"] for m in rec.of("metrics") if "server_audio_ms" in m["metrics"])
            assert m["server_audio_ms"] > 0 and m["credits"] == 0.0

    run(go())


def test_periodic_session_usage_without_request_id_is_forwarded():
    from .helpers import make_extension

    async def go():
        ext, rec = make_extension()
        await ext._on_usage({"type": "usage", "audio_ms": 1200, "chars": 40, "credits": 0})  # gateway tick
        await ext._on_usage({"type": "usage"})                                               # nothing to report
        assert [m["metrics"] for m in rec.of("metrics")] == [
            {"server_audio_ms": 1200, "server_chars": 40, "credits": 0}]

    run(go())

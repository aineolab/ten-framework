"""Request lifecycle through the real client against the fake NRSP server."""
import pytest
from ten_ai_base.message import TTSAudioEndReason

from .helpers import harness, run, text_input, wait_for


async def _speak(ext, rec, rid="r1", chunks=("Kumusta ", "ka na?"), end_text=""):
    for c in chunks:
        await ext.request_tts(text_input(rid, c))
    await ext.request_tts(text_input(rid, end_text, end=True))
    await wait_for(lambda: "finish" in rec.names(), what="finish_request")


def test_streaming_increments_produce_start_audio_end_in_order():
    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await ext._ensure_connected()
            await _speak(ext, rec)
            names = rec.names()
            assert names.index("audio_start") < names.index("audio") < names.index("audio_end") < names.index("finish")
            assert names.count("audio_start") == 1 and names.count("audio_end") == 1
            end = rec.of("audio_end")[0]
            assert end["reason"] == TTSAudioEndReason.REQUEST_END
            # (7 + 6) stripped chars * 60 ms in the fake
            assert end["duration_ms"] == 780
            assert rec.of("finish")[0]["error"] is None
            # all increments + the end marker reached the server as NRSP `text`
            texts = srv.received("text")
            assert [t["text"] for t in texts] == ["Kumusta ", "ka na?", ""]
            assert [t["end"] for t in texts] == [False, False, True]
            assert all(t["request_id"] == "r1" for t in texts)

    run(go())


def test_sample_rate_follows_server_audio_start_24k_orpheus():
    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await _speak(ext, rec)
            assert {e["rate"] for e in rec.of("audio")} == {24000}
            assert ext.synthesize_audio_sample_rate() == 24000

    run(go())


def test_sample_rate_follows_server_audio_start_16k_speecht5_fallback():
    async def go():
        # config says 24000 but the server says 16000: the server wins
        async with harness(model="halohalo/tts-speecht5", sample_rate=24000) as (srv, ext, rec):
            await _speak(ext, rec)
            assert {e["rate"] for e in rec.of("audio")} == {16000}
            # 16 kHz bookkeeping: durations are derived from the announced rate
            assert rec.of("audio_end")[0]["duration_ms"] == 780

    run(go())


def test_frame_timestamps_are_cumulative_audio_offsets():
    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await _speak(ext, rec)
            frames = rec.of("audio")
            ts = [f["ts"] for f in frames]
            assert ts[0] == 0
            expected = 0
            for f in frames:
                assert f["ts"] == expected
                expected += f["n"] * 1000 // (24000 * 2)

    run(go())


def test_two_sequential_requests_reuse_one_socket():
    async def go():
        async with harness() as (srv, ext, rec):
            await _speak(ext, rec, "a", ("one ",))
            rec.events.clear()
            await _speak(ext, rec, "b", ("two ",))
            assert len(srv.connections) == 1
            assert rec.of("audio_start")[0]["request_id"] == "b"

    run(go())


def test_empty_request_gets_start_and_end_with_zero_duration():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext.request_tts(text_input("e", "", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert rec.names()[-4:] == ["audio_start", "audio_end", "usage", "finish"]
            assert rec.of("audio_end")[0]["duration_ms"] == 0

    run(go())


def test_empty_non_final_chunk_is_not_sent():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext.request_tts(text_input("x", ""))
            assert srv.received("text") == []

    run(go())


def test_text_for_finished_request_is_rejected_without_events():
    async def go():
        async with harness() as (srv, ext, rec):
            await _speak(ext, rec, "r", ("hi ",))
            n = len(rec.events)
            await ext.request_tts(text_input("r", "late"))
            assert len(rec.events) == n
            assert [t["text"] for t in srv.received("text")][-1] == ""

    run(go())


def test_enable_words_maps_alignment_to_text_results():
    async def go():
        async with harness(enable_words=True) as (srv, ext, rec):
            await _speak(ext, rec, "w", ("hello big world",))
            results = [r["result"] for r in rec.of("text_result")]
            assert [w.word for w in results[0].words] == ["hello", "big", "world"]
            assert results[-1].text_result_end is True and results[-1].words == []
            # terminal marker lands after the last word
            last_word = results[0].words[-1]
            assert results[-1].start_ms >= last_word.start_ms

    run(go())


def test_words_not_emitted_by_default():
    async def go():
        async with harness() as (srv, ext, rec):
            await _speak(ext, rec, "w", ("hello big world",))
            assert rec.of("text_result") == []

    run(go())


def test_connection_status_sequence():
    async def go():
        async with harness() as (srv, ext, rec):
            await ext._ensure_connected()
            assert rec.names() == ["connecting", "connected"]
            ext.metrics_connect_delay.assert_awaited_once()

    run(go())


def test_request_text_is_never_logged():
    """Review item 2: log request id, length and status, never the text."""
    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await ext._ensure_connected()
            await _speak(ext, rec, chunks=("Ang PIN ko ", "ay 4821."))
            logged = " ".join(str(c) for c in ext.ten_env.mock_calls)
            assert "chars: 11" in logged and "request ID: r1" in logged
            assert "4821" not in logged and "PIN" not in logged

    run(go())


def test_request_error_log_has_no_text():
    from unittest.mock import AsyncMock

    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await ext._ensure_connected()
            ext.client.send_text = AsyncMock(side_effect=RuntimeError("boom"))
            await ext.request_tts(text_input("r2", "Ang PIN ko ay 4821.", end=True))
            errors = " ".join(str(c) for c in ext.ten_env.log_error.call_args_list)
            assert "request ID r2 (19 chars)" in errors and "boom" in errors
            assert "4821" not in errors

    run(go())

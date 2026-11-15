"""Wire-level edge cases of the NRSP TTS client."""
import json
import struct
from unittest.mock import AsyncMock, MagicMock

from nyo_tts_python.tts_client import NYOTTSClient, _BoundedSet

from .helpers import make_config, run


def _client():
    c = NYOTTSClient(make_config(), MagicMock())
    got = []
    c.on_audio = AsyncMock(side_effect=lambda rid, off, pcm: got.append((rid, off, len(pcm))))
    c.on_audio_start = AsyncMock()
    c.on_audio_end = AsyncMock()
    return c, got


def frame(seq, off, n=4):
    return struct.pack(">II", seq, off) + b"\x00" * n


def test_header_is_big_endian_u32_pair():
    c, got = _client()
    c._rid_by_seq[0x01020304] = "r"
    run(c._on_binary(struct.pack(">II", 0x01020304, 0x0A0B0C0D) + b"\x00\x00"))
    assert got == [("r", 0x0A0B0C0D, 2)]


def test_short_frame_ignored():
    c, got = _client()
    run(c._on_binary(b"\x00\x01"))
    assert got == []


def test_unknown_seq_with_no_live_request_is_dropped_and_counted():
    c, got = _client()
    run(c._on_binary(frame(9, 0)))
    assert got == [] and c.unmapped_frames_dropped == 1


def test_missing_request_seq_in_audio_start_binds_first_frame():
    c, got = _client()
    c._live.add("r")
    run(c._on_text(json.dumps({"type": "audio_start", "request_id": "r", "sample_rate": 24000})))
    run(c._on_binary(frame(7, 0)))
    run(c._on_binary(frame(7, 85)))
    assert got == [("r", 0, 4), ("r", 85, 4)]


def test_two_unbound_live_requests_are_ambiguous_and_dropped():
    c, got = _client()
    c._live.update({"a", "b"})
    run(c._on_binary(frame(1, 0)))
    assert got == [] and c.unmapped_frames_dropped == 1


def test_flush_marks_seq_stale_and_drops_later_frames():
    async def go():
        c, got = _client()
        c._seq_by_rid["r"] = 5
        c._rid_by_seq[5] = "r"
        await c._on_binary(frame(5, 0))
        await c.flush("r")  # no socket: local part only
        await c._on_binary(frame(5, 85))
        assert got == [("r", 0, 4)] and c.stale_frames_dropped == 1

    run(go())


def test_audio_start_of_cancelled_request_makes_its_seq_stale():
    async def go():
        c, got = _client()
        await c.flush("r")  # cancelled before the server even announced it
        await c._on_text(json.dumps({"type": "audio_start", "request_id": "r", "request_seq": 3, "sample_rate": 24000}))
        await c._on_binary(frame(3, 0))
        c.on_audio_start.assert_not_awaited()
        assert got == [] and c.stale_frames_dropped == 1

    run(go())


def test_audio_end_ack_of_our_flush_is_swallowed():
    async def go():
        c, _ = _client()
        await c.flush("r")
        await c._on_text(json.dumps({"type": "audio_end", "request_id": "r", "reason": "flushed", "audio_ms": 10}))
        c.on_audio_end.assert_not_awaited()

    run(go())


def test_non_json_text_frame_does_not_crash():
    c, _ = _client()
    run(c._on_text("{not json"))


def test_unknown_message_type_ignored():
    c, _ = _client()
    run(c._on_text(json.dumps({"type": "future_thing"})))


def test_forget_marks_finished_seq_stale():
    async def go():
        c, got = _client()
        c._seq_by_rid["r"] = 2
        c._rid_by_seq[2] = "r"
        c.forget("r")
        await c._on_binary(frame(2, 0))
        assert got == [] and c.stale_frames_dropped == 1

    run(go())


def test_bounded_set_forgets_oldest():
    s = _BoundedSet(3)
    for i in range(5):
        s.add(i)
    assert 0 not in s and 1 not in s and all(i in s for i in (2, 3, 4))
    s.add(2)
    s.add(5)
    assert 2 in s and 3 not in s


def test_build_url_carries_model_voice_language():
    c = NYOTTSClient(make_config("ws://h:9100/", model="halohalo/tts-ceb", voice="maria", allow_insecure=True),
                     MagicMock())
    assert c.build_url() == "ws://h:9100/v1/tts?model=halohalo%2Ftts-ceb&voice=maria&language=ceb"


def test_text_for_a_cancelled_request_is_never_sent():
    async def go():
        c, _ = _client()
        c.websocket = MagicMock()
        c.websocket.send = AsyncMock()
        await c.flush("r")
        await c.send_text("r", "late text", True)
        c.websocket.send.assert_not_awaited()

    run(go())


def test_odd_trailing_byte_is_trimmed_from_pcm():
    from nyo_tts_python.extension import _Request
    from .helpers import make_extension

    async def go():
        ext, rec = make_extension()
        ext._req = _Request("r")
        await ext._on_audio("r", 0, b"\x00" * 11)
        assert rec.of("audio")[0]["n"] == 10

    run(go())

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nyo_asr_python.recognition import NYOASRRecognition

from .helpers import run


def _rec(sample_rate=16000):
    tl = MagicMock()
    ws = MagicMock()
    ws.send = AsyncMock()
    rec = NYOASRRecognition("k", tl, MagicMock(), {"sample_rate": sample_rate}, MagicMock())
    rec.websocket = ws
    rec.is_started = True
    return rec, tl, ws


def test_duration_is_computed_from_pcm16_bytes(monkeypatch):
    from websockets.protocol import State
    rec, tl, ws = _rec()
    ws.state = State.OPEN
    run(rec.send_audio_frame(b"\x00" * 640))  # 20 ms @ 16 kHz
    tl.add_user_audio.assert_called_once_with(20)
    ws.send.assert_awaited_once_with(b"\x00" * 640)
    assert rec.frames_sent == 1 and rec.send_ns_total > 0


def test_duration_for_8k():
    from websockets.protocol import State
    rec, tl, ws = _rec(8000)
    ws.state = State.OPEN
    run(rec.send_audio_frame(b"\x00" * 320))  # 20 ms @ 8 kHz
    tl.add_user_audio.assert_called_once_with(20)


def test_send_when_closed_raises():
    rec, _, _ = _rec()
    rec.is_started = False
    try:
        run(rec.send_audio_frame(b"\x00\x00"))
    except RuntimeError as e:
        assert "not connected" in str(e)
    else:
        raise AssertionError("expected RuntimeError")

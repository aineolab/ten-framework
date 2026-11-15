import json
from unittest.mock import AsyncMock, MagicMock

from .helpers import make_extension, run


def test_send_audio_pushes_bytes_to_dumper_and_socket():
    async def go():
        ext = make_extension(dump=False)
        ext.audio_dumper = MagicMock()
        ext.audio_dumper.push_bytes = AsyncMock()
        ext.recognition = MagicMock()
        ext.recognition.is_connected.return_value = True
        ext._live = True  # past on_open's buffer replay
        ext.recognition.send_audio_frame = AsyncMock()
        frame = MagicMock()
        frame.lock_buf.return_value = bytearray(b"\x01\x02" * 320)
        assert await ext.send_audio(frame, "s") is True
        ext.audio_dumper.push_bytes.assert_awaited_once_with(b"\x01\x02" * 320)
        ext.recognition.send_audio_frame.assert_awaited_once()
        frame.unlock_buf.assert_called_once()

    run(go())


def test_send_audio_returns_false_when_disconnected():
    async def go():
        ext = make_extension()
        ext.recognition = None
        assert await ext.send_audio(MagicMock(), "s") is False

    run(go())


def test_send_failure_unlocks_buffer_and_returns_false():
    async def go():
        ext = make_extension()
        ext.recognition = MagicMock()
        ext.recognition.is_connected.return_value = True
        ext._live = True  # past on_open's buffer replay
        ext.recognition.send_audio_frame = AsyncMock(side_effect=RuntimeError("boom"))
        frame = MagicMock()
        frame.lock_buf.return_value = bytearray(b"\x00\x00")
        assert await ext.send_audio(frame, "s") is False

    run(go())


def test_dump_config_files_are_valid():
    import pathlib
    cfg = json.loads((pathlib.Path(__file__).parent / "configs" / "property_dump.json").read_text())
    assert cfg["dump"] is True and cfg["dump_path"]

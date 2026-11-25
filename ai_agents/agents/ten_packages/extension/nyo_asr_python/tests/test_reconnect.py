import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nyo_asr_python.reconnect_manager import ReconnectManager

from .helpers import make_extension, run


def test_backoff_doubles_and_caps(monkeypatch):
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def go():
        mgr = ReconnectManager(max_attempts=6)
        fail = AsyncMock(side_effect=RuntimeError("down"))
        for _ in range(6):
            assert await mgr.handle_reconnect(fail) is False
        assert not mgr.can_retry()

    run(go())
    expected = [0.5, 1.0, 2.0, 4.0, 5.0, 5.0]
    assert all(e / 2 <= got <= e for got, e in zip(sleeps, expected)) and len(sleeps) == 6


def test_last_attempt_reports_fatal_earlier_nonfatal(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    async def go():
        mgr = ReconnectManager(max_attempts=2)
        handler = AsyncMock()
        fail = AsyncMock(side_effect=RuntimeError("down"))
        await mgr.handle_reconnect(fail, handler, "nyo")
        await mgr.handle_reconnect(fail, handler, "nyo")
        codes = [c.args[0].code for c in handler.await_args_list]
        assert codes == [1000, -1000]
        assert "Reconnection attempt #1" in handler.await_args_list[0].args[0].message
        assert handler.await_args_list[0].args[1].vendor == "nyo"

    run(go())


def test_success_resets_counter(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    async def go():
        mgr = ReconnectManager(max_attempts=3)
        await mgr.handle_reconnect(AsyncMock(side_effect=RuntimeError("x")))
        assert mgr.attempts == 1
        mgr.mark_connection_successful()
        assert mgr.attempts == 0 and mgr.can_retry()

    run(go())


def test_unexpected_close_schedules_reconnect():
    async def go():
        ext = make_extension()
        ext._schedule_reconnect = MagicMock()
        await ext.on_close(expected=False, reason="closed: 1011")
        assert ext.recognition is None
        ext.on_disconnected.assert_awaited_once()
        ext._schedule_reconnect.assert_called_once()

    run(go())


@pytest.mark.parametrize("flag", ["expected", "stop", "fatal"])
def test_close_does_not_reconnect_when_expected_stopped_or_fatal(flag):
    async def go():
        ext = make_extension()
        ext._schedule_reconnect = MagicMock()
        ext._stop_requested = flag == "stop"
        ext._fatal = flag == "fatal"
        await ext.on_close(expected=flag == "expected", reason="closed")
        ext._schedule_reconnect.assert_not_called()

    run(go())


def test_on_open_resets_server_timeline_and_replays_buffer():
    async def go():
        ext = make_extension()
        ext.reconnect_manager = ReconnectManager()
        ext.reconnect_manager.attempts = 3
        ext.audio_timeline = MagicMock()
        ext.audio_timeline.get_total_user_audio_duration.return_value = 1200
        ext._flush_buffered_audio_frames = AsyncMock()
        ext.connection_start_timestamp = 0
        await ext.on_open()
        assert ext.sent_user_audio_duration_ms_before_last_reset == 1200
        ext.audio_timeline.reset.assert_called_once()
        # QA r2 rule C: an open alone does not end the outage; a result (or 10 s up) does
        assert ext.reconnect_manager.attempts == 3
        ext.reconnect_manager.mark_stable()
        assert ext.reconnect_manager.attempts == 0
        ext.on_connected.assert_awaited_once()
        ext.send_connect_delay_metrics.assert_awaited_once()
        ext._flush_buffered_audio_frames.assert_awaited_once()

    run(go())


def test_keep_buffer_is_two_seconds_of_audio():
    ext = make_extension()
    strat = ext.buffer_strategy()
    assert type(strat).__name__ == "ASRBufferConfigModeKeep"
    assert strat.byte_limit == 2 * 16000 * 2
    assert make_extension(buffer_ms=500).buffer_strategy().byte_limit == 16000


def test_frames_dropped_by_keep_overflow_are_accounted_in_the_timeline():
    from nyo_asr_python.extension import NYOASRExtension  # noqa: F401
    from ten_ai_base.timeline import AudioTimeline

    async def go():
        ext = make_extension()
        ext.reconnect_manager = ReconnectManager()
        ext.audio_timeline = AudioTimeline()
        ext._flush_buffered_audio_frames = AsyncMock()
        ext.connection_start_timestamp = 0
        ext._last_frame_ms = 20
        ext.dropped_audio_frames = 50  # 1 s of 20 ms frames fell out of the Keep window
        await ext.on_open()
        assert ext.audio_timeline.total_dropped_audio_duration == 1000
        # vendor time 0 now maps after the dropped second
        assert ext.audio_timeline.get_audio_duration_before_time(0) >= 0
        ext.dropped_audio_frames = 50
        await ext.on_open()  # nothing new dropped -> no double counting
        assert ext.audio_timeline.total_dropped_audio_duration == 0

    run(go())

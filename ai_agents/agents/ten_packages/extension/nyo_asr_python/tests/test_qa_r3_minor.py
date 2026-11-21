"""QA r3: reconnect-loop vendor code keeps the HTTP status; init failure settles DISCONNECTED."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from nyo_asr_python.reconnect_manager import ReconnectManager
from nyo_asr_python.recognition import NRSPConnectError

from .helpers import make_extension, run


def test_reconnect_loop_error_carries_the_handshake_status(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    async def go():
        ext = make_extension()
        ext._preflight_ok = True
        ext.reconnect_manager = ReconnectManager(window_ms=60_000)
        ext._connect_recognition = AsyncMock(side_effect=NRSPConnectError("HTTP 503", 503, retry_after=2))
        ok = await ext.reconnect_manager.handle_reconnect(ext._reconnect_once, ext._send_reconnect_error, "nyo")
        assert not ok
        err, vi = ext.send_asr_error.await_args.args
        assert err.code == 1000 and vi.code == "503"

    run(go())


def test_start_after_init_failure_settles_disconnected():
    async def go():
        ext = make_extension()
        ext._emit_connection_transition = AsyncMock()
        ext._init_failed = True
        await ext.start_connection()
        ext.on_disconnected.assert_awaited_once()

    run(go())

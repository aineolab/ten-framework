"""QA r3: single-flight connect; callbacks from a superseded socket are ignored; start() never undoes a close."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from websockets.protocol import State

from nyo_asr_python.extension import _CurrentSocketCallback
from nyo_asr_python.recognition import NYOASRRecognition

from .helpers import run
from .test_protocol import session, wait_for


def _track(monkeypatch):
    made = []
    orig_init = NYOASRRecognition.__init__

    def track(self, *a, **k):
        orig_init(self, *a, **k)
        made.append(self)

    monkeypatch.setattr(NYOASRRecognition, "__init__", track)
    return made


def _open(made):
    return [r for r in made if r.websocket is not None and r.websocket.state == State.OPEN]


def test_trigger_connect_during_reconnect_leaks_no_socket(monkeypatch):
    made = _track(monkeypatch)

    async def go():
        async with session() as (srv, ext):
            ext._preflight_ok = True
            ext._schedule_reconnect()           # an outage: the loop reconnects ...
            await asyncio.sleep(0.015)
            await ext.start_connection()        # ... and trigger_connect arrives
            await wait_for(ext.is_connected, what="connected")
            await asyncio.sleep(0.3)
            assert _open(made) == [ext.recognition] and ext.is_connected()

    run(go())


def test_concurrent_connect_attempts_share_one_socket(monkeypatch):
    made = _track(monkeypatch)

    async def go():
        async with session() as (srv, ext):
            ext._preflight_ok = True
            await asyncio.gather(ext._connect_recognition(), ext._connect_recognition())
            await asyncio.sleep(0.2)
            assert len(made) == 1 and _open(made) == [ext.recognition] and len(srv.connections) == 1

    run(go())


def test_superseded_socket_callbacks_do_not_touch_the_live_one(monkeypatch):
    made = _track(monkeypatch)

    async def go():
        async with session() as (srv, ext):
            ext._preflight_ok = True
            await ext.start_connection()
            live = ext.recognition
            stale = NYOASRRecognition("nyo-test-key", ext.audio_timeline, ext.ten_env, ext.config.params, ext)
            stale.callback = _CurrentSocketCallback(ext, stale)
            await stale.start()                  # opened after it was superseded: closes itself
            assert not stale.is_connected()
            await stale.callback.on_error("no client message for 300 s", "idle_timeout", True, "nyo", "idle_timeout")
            await stale.callback.on_close(False, "closed: 1001")
            await asyncio.sleep(0.1)
            assert ext.recognition is live and ext.is_connected() and not ext._idle_closed
            assert ext._reconnect_task is None and _open(made) == [live]

    run(go())


def test_start_does_not_undo_a_close_during_the_handshake():
    async def go():
        async with session() as (srv, ext):
            cb = AsyncMock()
            rec = NYOASRRecognition("nyo-test-key", MagicMock(), MagicMock(), ext.config.params, cb)
            task = asyncio.create_task(rec.start())
            await asyncio.sleep(0)               # handshake in flight
            await rec.close()
            await task
            await asyncio.sleep(0.1)
            assert not rec.is_connected() and rec.websocket is None
            cb.on_open.assert_not_awaited()

    run(go())


def test_trigger_connect_after_stop_is_ignored():
    async def go():
        async with session() as (srv, ext):
            ext.stopped = True                   # base on_stop
            await ext.start_connection()
            assert ext.recognition is None and not srv.connections
            ext.on_disconnected.assert_awaited()

    run(go())

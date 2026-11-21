"""QA review regressions. The two gateway probes run against the real serve/ gateway (mock engines) on
127.0.0.1:9131 (default idle) and 127.0.0.1:9132 (idle_timeout_s=3); they skip when it is not running."""
import asyncio
import socket
import urllib.request
from unittest.mock import AsyncMock, MagicMock

import pytest

from nyo_asr_python.reconnect_manager import ReconnectManager

from .helpers import make_extension, run

FATAL = -1000
NON_FATAL = 1000


def _up(port):
    s = socket.socket()
    try:
        s.settimeout(0.3)
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


class _Frame:
    def __init__(self, data):
        self.data = data

    def lock_buf(self):
        return bytearray(self.data)

    def unlock_buf(self, _):
        pass

    def get_buf(self):
        return self.data

    def get_property_to_json(self, _):
        return "", None


async def _wait(pred, timeout=10.0):
    end = asyncio.get_event_loop().time() + timeout
    while not pred():
        if asyncio.get_event_loop().time() > end:
            return False
        await asyncio.sleep(0.02)
    return True


def _gw_ext(port, **kw):
    ext = make_extension(base_url=f"ws://127.0.0.1:{port}", allow_insecure=True,
                         model="halohalo/asr-fil", language="fil", **kw)
    ext._emit_connection_transition = AsyncMock()
    ext.reconnect_manager = ReconnectManager(window_ms=60_000)
    return ext


@pytest.mark.skipif(not _up(9131), reason="gateway not running")
def test_gateway_engine_swap_1012_is_nonfatal_and_reconnects():
    """Gateway engine swap: fatal engine_restarting + close 1012 -> NON_FATAL, reconnect, keep serving."""
    async def go():
        ext = _gw_ext(9131)
        await ext.start_connection()
        assert ext.is_connected()
        first = ext.recognition
        for _ in range(10):
            await ext.send_audio(_Frame(bytes(640)), "s")
        req = urllib.request.Request("http://127.0.0.1:9131/v1/admin/swap?engine=asr&spec=mock", method="POST",
                                     headers={"Authorization": "Bearer nyo-admin-key"})
        await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=20).read())
        assert await _wait(lambda: ext.is_connected() and ext.recognition is not first)
        codes = [(c.args[0].code, c.args[1].code) for c in ext.send_asr_error.await_args_list]
        print("errors:", codes)
        assert (NON_FATAL, "engine_restarting") in codes and FATAL not in [c for c, _ in codes]
        assert not ext._fatal
        await ext.stop_connection()

    run(go())


@pytest.mark.skipif(not _up(9132), reason="gateway not running")
def test_gateway_idle_timeout_is_silent_and_audio_reconnects():
    """Gateway idle timer only counts audio -> idle_timeout close: no asr_error; next audio reconnects."""
    async def go():
        ext = _gw_ext(9132, finalize_timeout_ms=500)
        consumer = asyncio.create_task(ext._audio_frame_consumer())
        await ext.start_connection()
        assert ext.is_connected()
        first = ext.recognition
        assert await _wait(lambda: ext._idle_closed, timeout=8)
        await asyncio.sleep(0.3)
        assert ext.send_asr_error.await_count == 0 and not ext._fatal
        assert not ext.is_connected() and ext._reconnect_task is None  # lazy: nothing until audio
        for _ in range(5):
            await ext.on_audio_frame(ext.ten_env, _Frame(bytes(640)))
        assert await _wait(lambda: ext.is_connected() and ext.recognition is not first)
        assert ext.send_asr_error.await_count == 0
        consumer.cancel()
        await ext.stop_connection()

    run(go())


def test_overlapping_finalizes_each_get_their_own_finalize_end():
    """TEN dispatches on_data concurrently; a 2nd asr_finalize while the 1st waits."""
    async def go():
        ext = make_extension(finalize_timeout_ms=2000)
        events = {"A": asyncio.Event(), "B": asyncio.Event()}
        rec = MagicMock()
        rec.is_connected.return_value = True
        rec.send_finalize = AsyncMock()

        async def wait(fid, _t):
            await events[fid].wait()
            return True
        rec.wait_finalize_end = wait
        ext.recognition = rec
        sent = []
        ext.send_asr_finalize_end = AsyncMock(side_effect=lambda: sent.append(ext.finalize_id))
        ext.finalize_id = "A"                       # base on_data sets finalize_id, then awaits finalize()
        t1 = asyncio.create_task(ext.finalize("s"))
        await asyncio.sleep(0)
        ext.finalize_id = "B"
        t2 = asyncio.create_task(ext.finalize("s"))
        await asyncio.sleep(0)
        events["A"].set()                            # server finalize_end for A
        await t1
        assert sent == ["A"]                         # B is still open
        events["B"].set()
        await t2
        assert sent == ["A", "B"] and ext.last_finalize_timestamp == 0

    run(go())


def test_close_during_reconnect_on_open_keeps_reconnecting():
    """Reader task dies (server close) while the reconnect task is still inside on_open()."""
    async def go():
        ext = make_extension()
        ext._emit_connection_transition = AsyncMock()
        ext.reconnect_manager = ReconnectManager(base_delay=0.0, max_delay=0.0)
        ext._preflight_ok = True
        attempts = []

        async def connect():
            attempts.append(1)
            rec = MagicMock()
            rec.is_connected.return_value = True
            ext.recognition = rec
            if len(attempts) == 1:
                async def drop():  # the socket drops while on_open awaits on_connected()/the replay
                    rec.is_connected.return_value = False
                    await ext.on_close(False, "closed: 1006")
                ext.on_connected = AsyncMock(side_effect=drop)
            else:
                ext.on_connected = AsyncMock()
            await ext.on_open()

        ext._connect_recognition = connect
        ext._schedule_reconnect()
        assert await _wait(lambda: ext._reconnect_task is None, timeout=2)
        assert len(attempts) == 2 and ext.is_connected()

    run(go())

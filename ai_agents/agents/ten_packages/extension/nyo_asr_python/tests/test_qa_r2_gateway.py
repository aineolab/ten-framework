"""QA round 2: probes against the real serve/ gateway (mock engines).

127.0.0.1:9140: default idle, SAPIN_SWAP_DRAIN_S=4 (a swap keeps the asr slot "swapping" for ~4 s
while a session is open). 127.0.0.1:9141: SAPIN_IDLE_TIMEOUT_S=3. Skipped when not running.
"""
import asyncio
import socket
import urllib.request
from unittest.mock import AsyncMock

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
                         model=kw.pop("model", "halohalo/asr-fil"), language="fil", **kw)
    ext._emit_connection_transition = AsyncMock()
    ext.reconnect_manager = ReconnectManager(window_ms=60_000)
    return ext


def _codes(ext):
    return [(c.args[0].code, c.args[1].code if c.args[1] else None) for c in ext.send_asr_error.await_args_list]


def _swap(port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/admin/swap?engine=asr&spec=mock", method="POST",
                                 headers={"Authorization": "Bearer x"})
    return urllib.request.urlopen(req, timeout=30).read()


@pytest.mark.skipif(not _up(9140), reason="gateway not running")
def test_r2_gateway_engine_swap_is_nonfatal_and_reconnects():
    async def go():
        ext = _gw_ext(9140)
        await ext.start_connection()
        assert ext.is_connected()
        first = ext.recognition
        for _ in range(10):
            await ext.send_audio(_Frame(bytes(640)), "s")
        await asyncio.to_thread(_swap, 9140)
        assert await _wait(lambda: ext.is_connected() and ext.recognition is not first, 20)
        print("errors:", _codes(ext))
        assert (NON_FATAL, "engine_restarting") in _codes(ext) and not ext._fatal
        await ext.stop_connection()

    run(go())


@pytest.mark.skipif(not _up(9141), reason="gateway not running")
def test_r2_gateway_idle_timeout_is_silent_and_audio_reconnects():
    async def go():
        ext = _gw_ext(9141)
        consumer = asyncio.create_task(ext._audio_frame_consumer())
        await ext.start_connection()
        first = ext.recognition
        assert await _wait(lambda: ext._idle_closed, timeout=8)
        await asyncio.sleep(0.3)
        assert ext.send_asr_error.await_count == 0 and not ext._fatal and not ext.is_connected()
        for _ in range(5):
            await ext.on_audio_frame(ext.ten_env, _Frame(bytes(640)))
        assert await _wait(lambda: ext.is_connected() and ext.recognition is not first)
        assert ext.send_asr_error.await_count == 0
        consumer.cancel()
        await ext.stop_connection()

    run(go())


@pytest.mark.skipif(not _up(9140), reason="gateway not running")
@pytest.mark.parametrize("model", ["halohalo/asr-fil"])
def test_r2_session_started_during_engine_swap_goes_fatal_in_preflight(model):
    """QA r2 N1: a session that starts while the gateway swaps its ASR engine (GET /v1/models lists no ASR
    model) must not go FATAL: the preflight is inconclusive and the handshake queues at the swap gate."""
    async def go():
        holder = _gw_ext(9140, model=model)       # keeps the drain phase of the swap busy (~4 s)
        await holder.start_connection()
        assert holder.is_connected()
        swap = asyncio.create_task(asyncio.to_thread(_swap, 9140))
        await asyncio.sleep(0.5)                   # swap is in its drain phase now
        ext = _gw_ext(9140, model=model)          # a new voice session starts
        await ext.start_connection()
        await swap
        assert await _wait(ext.is_connected, 20)
        print("new session errors:", _codes(ext))
        await holder.stop_connection()
        await ext.stop_connection()
        assert not ext._fatal, f"session start during an engine swap went FATAL: {_codes(ext)}"

    run(go())


@pytest.mark.skipif(not _up(9140), reason="gateway not running")
def test_r2_gateway_overlapping_finalizes_get_their_own_ids():
    async def go():
        ext = _gw_ext(9140, finalize_timeout_ms=3000)
        ends = []
        async def fe():
            ends.append(ext.finalize_id)
        ext.send_asr_finalize_end = AsyncMock(side_effect=fe)
        await ext.start_connection()
        for _ in range(50):
            await ext.send_audio(_Frame(bytes(640)), "s")
        ext.finalize_id = "A"
        ta = asyncio.create_task(ext.finalize(None))
        await asyncio.sleep(0)
        for _ in range(20):
            await ext.send_audio(_Frame(bytes(640)), "s")
        ext.finalize_id = "B"
        tb = asyncio.create_task(ext.finalize(None))
        await asyncio.gather(ta, tb)
        finals = [c.args[0].text for c in ext.send_asr_result.await_args_list if c.args[0].final]
        print("finalize_end ids:", ends, "finals:", finals, "errors:", _codes(ext))
        await ext.stop_connection()
        assert sorted(ends) == ["A", "B"]

    run(go())

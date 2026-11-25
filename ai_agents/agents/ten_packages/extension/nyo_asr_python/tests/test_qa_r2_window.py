"""QA round 2: reconnect window vs. sessions that open fine and are then killed with a reconnectable
error, and Keep-buffer audio lost when the socket dies during the on_open replay."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from websockets.asyncio.server import serve

from nyo_asr_python.reconnect_manager import ReconnectManager

from .helpers import make_extension, run

FATAL = -1000
NON_FATAL = 1000
PORT = 9142


async def _killer(ws):
    """Accept, read config, then end the session as the gateway does when an engine session is lost."""
    await ws.recv()
    await asyncio.sleep(0.05)
    await ws.send(json.dumps({"type": "error", "code": "engine_unavailable", "fatal": True,
                              "message": "engine session could not be re-established",
                              "vendor": {"name": "halohalo", "code": "engine_unavailable", "message": "x"}}))
    await ws.close(1011)


def test_r2_window_never_expires_when_each_socket_opens_then_dies():
    """QA r2 N2 (rule C/D): a gateway that accepts and kills every socket ends in one FATAL after
    reconnect_window_ms (an open alone does not reset the outage), with rate-limited NON_FATAL errors."""
    async def go():
        async with serve(_killer, "127.0.0.1", PORT, subprotocols=["nrsp.v1"]):
            ext = make_extension(base_url=f"ws://127.0.0.1:{PORT}", allow_insecure=True,
                                 model="halohalo/asr-fil")
            ext._emit_connection_transition = AsyncMock()
            ext._preflight_ok = True
            ext.reconnect_manager = ReconnectManager(window_ms=1000)
            await ext.start_connection()
            await asyncio.sleep(5.0)              # 5x the window
            codes = [c.args[0].code for c in ext.send_asr_error.await_args_list]
            await ext.stop_connection()
            print("errors in 5 s with a 1 s window:", len(codes), "fatal:", FATAL in codes)
            assert FATAL in codes, f"no FATAL after 5x reconnect_window_ms; {len(codes)} NON_FATAL errors"
            assert codes.count(FATAL) == 1 and codes[-1] == FATAL
            assert codes.count(NON_FATAL) == 1, codes  # rate-limited: first of the outage, then <= 1 per 10 s

    run(go())


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


def test_r2_replay_into_a_dying_socket_discards_the_keep_buffer():
    """QA r2 N3: the on_open replay stops at the first failed send; the rest stays buffered, in order."""
    async def go():
        ext = make_extension()
        rec = MagicMock()
        rec.bytes_per_ms = 32.0
        state = {"n": 0}
        rec.is_connected = lambda: state["n"] < 1
        async def send(_):
            state["n"] += 1
        rec.send_audio_frame = send
        ext.recognition = rec
        for _ in range(10):
            ext.buffered_frames.put_nowait(_Frame(bytes(640)))
        await ext._flush_buffered_audio_frames()
        print("frames sent:", state["n"], "left queued:", ext.buffered_frames.qsize())
        assert ext.buffered_frames.qsize() == 9, "9 buffered frames were discarded after the socket died"
        assert ext.buffered_frames_size == 9 * 640

    run(go())

"""Real NRSP client + extension against ten/fake_nrsp_server.py (no TEN graph)."""
import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock

from fake_nrsp_server import FakeNRSPServer
from nyo_asr_python.reconnect_manager import ReconnectManager

from .helpers import make_extension, run

PCM_20MS = b"\x10\x20" * 320  # non-zero => "speech" for the fake


class Frame:
    def __init__(self, data=PCM_20MS):
        self._d = bytearray(data)

    def get_buf(self):
        return bytes(self._d)

    def lock_buf(self):
        return self._d

    def unlock_buf(self, _):
        pass

    def get_property_to_json(self, _):
        return json.dumps({"session_id": "sess-1"}), None


async def wait_for(pred, timeout=4.0, what="condition"):
    end = asyncio.get_event_loop().time() + timeout
    while not pred():
        if asyncio.get_event_loop().time() > end:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


@contextlib.asynccontextmanager
async def session(key="nyo-test-key", **params):
    async with FakeNRSPServer() as srv:
        ext = make_extension(base_url=srv.url, api_key=key, **params)
        ext.ten_env.get_property_to_json = AsyncMock()
        ext._emit_connection_transition = AsyncMock()
        ext.reconnect_manager = ReconnectManager(base_delay=0.02, max_delay=0.05, max_attempts=5)
        order: list[tuple[str, bool | None]] = []
        ext.send_asr_result.side_effect = lambda r: order.append(("result", r.final))
        ext.send_asr_finalize_end.side_effect = lambda: order.append(("finalize_end", None))
        ext.order = order
        try:
            yield srv, ext
        finally:
            await ext.stop_connection()


def test_handshake_carries_bearer_subprotocol_model_and_config():
    async def go():
        async with session(model="halohalo/asr-ceb", language="ceb") as (srv, ext):
            await ext.start_connection()
            assert ext.is_connected()
            c = srv.connections[0]
            assert c["key"] == "nyo-test-key" and c["subprotocol"] == "nrsp.v1"
            assert c["path"] == "/v1/asr"
            assert c["query"] == {"model": "halohalo/asr-ceb", "language": "ceb", "sample_rate": "16000"}
            await wait_for(lambda: srv.received("config"), what="config message")
            cfg = srv.received("config")[0]
            assert cfg["language"] == "ceb" and cfg["interim"] is True and cfg["sample_rate"] == 16000
            ext.on_connected.assert_awaited_once()

    run(go())


def test_interims_then_final_then_finalize_end_in_that_order():
    async def go():
        async with session(language="fil") as (srv, ext):
            await ext.start_connection()
            for _ in range(40):  # 800 ms
                assert await ext.send_audio(Frame(), "sess-1")
            await wait_for(lambda: any(not f for k, f in ext.order if k == "result"), what="interim")
            ext.finalize_id = "fin-1"
            await ext.finalize("sess-1")
            kinds = ext.order
            assert kinds[-2:] == [("result", True), ("finalize_end", None)], kinds
            assert all(f is False for k, f in kinds[:-2] if k == "result")
            final = ext.send_asr_result.await_args_list[-1].args[0]
            assert final.text == "kumusta mundo" and final.language == "fil-PH"
            assert final.metadata["asr_info"]["backend"] == "fake-whisper"
            assert srv.received("finalize")[0]["id"] == "fin-1"

    run(go())


def test_finalize_without_speech_returns_finalize_end_only():
    async def go():
        async with session() as (srv, ext):
            await ext.start_connection()
            ext.finalize_id = "f0"
            await ext.finalize("s")
            assert ext.order == [("finalize_end", None)]

    run(go())


def test_finalize_timeout_promotes_last_interim():
    async def go():
        async with session(key="k+nofinal", finalize_timeout_ms=150, language="fil") as (srv, ext):
            await ext.start_connection()
            for _ in range(20):
                await ext.send_audio(Frame(), "s")
            await wait_for(lambda: ext.order, what="interim")
            ext.finalize_id = "fin-x"
            await ext.finalize("s")
            assert ext.order[-2:] == [("result", True), ("finalize_end", None)]
            promoted = ext.send_asr_result.await_args_list[-1].args[0]
            assert promoted.metadata["asr_info"]["promoted_on_timeout"] is True

    run(go())


def test_bad_key_is_fatal_401_and_never_retries():
    async def go():
        async with session(key="invalid") as (srv, ext):
            await ext.start_connection()
            err, vi = ext.send_asr_error.await_args.args
            assert err.code == -1000 and vi.code == "401"
            assert ext._fatal and ext._reconnect_task is None
            await asyncio.sleep(0.2)
            assert len(srv.connections) == 0  # rejected before upgrade

    run(go())


def test_unknown_model_is_retried_then_fatal_model_not_found():
    """Rule A': a model /v1/models does not list may be an engine that is down: one NON_FATAL
    model_not_found, retries inside the reconnect budget, then one FATAL model_not_found."""
    async def go():
        async with session(model="halohalo/asr-xyz", language="xyz") as (srv, ext):
            await ext.start_connection()
            err, vi = ext.send_asr_error.await_args_list[0].args
            assert err.code == 1000 and vi.code == "model_not_found" and vi.vendor == "nyo"
            assert "halohalo/asr-xyz" in vi.message and "/v1/models" in vi.message
            assert not ext._fatal
            await wait_for(lambda: ext._reconnect_task is None, what="reconnect loop end")
            codes = [(c.args[0].code, c.args[1].code) for c in ext.send_asr_error.await_args_list]
            assert codes[-1] == (-1000, "model_not_found") and ext._fatal
            assert [c for c, _ in codes].count(-1000) == 1 and all(v == "model_not_found" for _, v in codes)
            assert srv.models_requests > 1  # checked again on every attempt

    run(go())


def test_fatal_midstream_error_stops_reconnect():
    async def go():
        async with session(key="k+fatal") as (srv, ext):
            await ext.start_connection()
            await ext.send_audio(Frame(), "s")
            await wait_for(lambda: ext.send_asr_error.await_count, what="error")
            err, vi = ext.send_asr_error.await_args.args
            assert err.code == -1000 and vi.vendor == "nyo" and vi.code == "engine_error"
            assert vi.message.startswith("[fake-vendor] ")  # review F5: engine named in the message
            await asyncio.sleep(0.2)
            assert len(srv.connections) == 1

    run(go())


def test_connection_refused_retries_with_backoff_then_gives_up():
    """Time-based budget: retries until the outage outlasts the window, then one FATAL. NON_FATAL
    connect errors are rate-limited: the first failure of the outage is reported, the rest are not."""
    async def go():
        ext = make_extension(base_url="ws://127.0.0.1:1", api_key="k")
        ext._emit_connection_transition = AsyncMock()
        ext.reconnect_manager = ReconnectManager(base_delay=0.01, max_delay=0.02, window_ms=300)
        await ext.start_connection()
        await wait_for(lambda: ext._reconnect_task is None, what="reconnect loop end")
        codes = [c.args[0].code for c in ext.send_asr_error.await_args_list]
        assert codes == [1000, -1000] and ext._fatal
        assert ext.reconnect_manager.attempts >= 3  # it kept trying for the whole window

    run(go())


def test_socket_drop_reconnects_and_replays_the_keep_buffer_in_order():
    async def go():
        async with session(key="k+drop3") as (srv, ext):
            await ext.start_connection()
            # drive the base class' real frame path: buffer while down, replay when up
            for _ in range(40):
                await ext._handle_audio_frame(ext.ten_env, Frame())
                await asyncio.sleep(0.01)
            await wait_for(lambda: len(srv.connections) == 2, what="second connection")
            await wait_for(lambda: srv.connections[1]["frames"] >= 5, what="replayed frames")
            assert ext.is_connected()
            assert ext.reconnect_manager.attempts == 0
            assert ext.sent_user_audio_duration_ms_before_last_reset > 0
            assert ext.on_disconnected.await_count >= 1

    run(go())


def test_keep_buffer_drops_oldest_beyond_two_seconds():
    async def go():
        ext = make_extension(buffer_ms=100)  # 3200 bytes == 5 frames
        ext.recognition = None
        for _ in range(12):
            await ext._handle_audio_frame(ext.ten_env, Frame())
        assert ext.buffered_frames.qsize() <= 5
        assert ext.dropped_audio_frames >= 7

    run(go())


def test_stop_sends_close_and_does_not_reconnect():
    async def go():
        async with session() as (srv, ext):
            await ext.start_connection()
            await ext.stop_connection()
            await wait_for(lambda: srv.received("close"), what="close message")
            await asyncio.sleep(0.15)
            assert len(srv.connections) == 1 and ext._reconnect_task is None

    run(go())


def test_socket_drop_during_finalize_gives_one_final_and_keeps_the_next_turn():
    """ten review F3, real client + fake server: the socket aborts on `finalize`; the last interim is promoted
    once, the extension reconnects, and the next turn on the new socket (ids restart at u0) keeps its final."""
    async def go():
        async with session(key="k+dropfin", language="fil", finalize_timeout_ms=2000) as (srv, ext):
            await ext.start_connection()
            for _ in range(20):                                  # 400 ms of speech -> interim u0
                await ext.send_audio(Frame(), "s")
            await wait_for(lambda: ext.order, what="interim on socket 1")
            ext.finalize_id = "fin-a"
            await ext.finalize("s")                              # socket aborts instead of answering
            finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
            info = finals[0].metadata["asr_info"] if finals else {}
            # whichever of the close and the woken finalize promotes it, a close is a close (promoted_on_close)
            assert len(finals) == 1 and info.get("promoted_on_close") and not info.get("promoted_on_timeout"), finals
            await wait_for(lambda: len(srv.connections) == 2 and ext.is_connected(), what="reconnect")
            for _ in range(20):
                await ext.send_audio(Frame(), "s")
            ext.finalize_id = "fin-b"
            await ext.finalize("s")
            finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
            assert len(finals) == 2 and not {"promoted_on_timeout", "promoted_on_close"} & set(finals[1].metadata["asr_info"])
            assert ext.send_asr_finalize_end.await_count == 2

    run(go())

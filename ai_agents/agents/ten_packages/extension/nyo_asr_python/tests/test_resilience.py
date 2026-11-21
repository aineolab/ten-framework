"""Regression tests for the resilience fixes: reconnectable server errors, idle close, time-based
reconnect budget, Retry-After, overlapping finalizes, finalize_end without a final, Nemotron
finalize timeout, deinit cleanup."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from nyo_asr_python.config import NYOASRConfig
from nyo_asr_python.const import NEMOTRON_FINALIZE_TIMEOUT_MS, NEMOTRON_MODEL
from nyo_asr_python.reconnect_manager import ReconnectManager
from nyo_asr_python.recognition import NRSPConnectError, NYOASRRecognition

from .helpers import make_extension, run
from .test_protocol import Frame, session, wait_for

FATAL = -1000
NON_FATAL = 1000


def _errors(ext):
    return [(c.args[0].code, c.args[1].code if c.args[1] else None) for c in ext.send_asr_error.await_args_list]


# ------------------------------------------------------------ error classes
@pytest.mark.parametrize("code", ["engine_restarting", "engine_unavailable", "slow_consumer"])
def test_reconnectable_fatal_codes_are_nonfatal(code):
    ext = make_extension()
    ext._schedule_reconnect = MagicMock()
    run(ext.on_error("going away", code, True, "nyo", code))
    assert _errors(ext) == [(NON_FATAL, code)] and not ext._fatal
    run(ext.on_close(False, "closed: 1012"))
    ext._schedule_reconnect.assert_called_once()


@pytest.mark.parametrize("code", ["unauthorized", "bad_message", "frame_too_large",
                                  "unsupported_language", "engine_error"])
def test_other_fatal_codes_stay_fatal(code):
    ext = make_extension()
    run(ext.on_error("no", code, True, "nyo", code))
    assert _errors(ext) == [(FATAL, code)] and ext._fatal


def test_unknown_model_server_error_is_nonfatal_model_not_found():
    """Rule A': unknown_model may be an engine that is down: NON_FATAL model_not_found, reconnect."""
    ext = make_extension()
    ext._schedule_reconnect = MagicMock()
    run(ext.on_error("unknown model 'halohalo/asr-hil'", "unknown_model", True, "nyo", "unknown_model"))
    assert _errors(ext) == [(NON_FATAL, "model_not_found")] and not ext._fatal
    run(ext.on_close(False, "closed: 1008"))
    ext._schedule_reconnect.assert_called_once()


def test_idle_timeout_is_not_an_error_and_does_not_reconnect_until_audio():
    ext = make_extension()
    ext._schedule_reconnect = MagicMock()
    run(ext.on_error("no client message for 300 s", "idle_timeout", True, "nyo", "idle_timeout"))
    run(ext.on_close(False, "closed: 1001"))
    assert ext.send_asr_error.await_count == 0 and not ext._fatal and ext._idle_closed
    ext._schedule_reconnect.assert_not_called()
    run(ext.on_audio_frame(ext.ten_env, Frame()))
    ext._schedule_reconnect.assert_called_once()
    assert not ext._idle_closed


def test_fake_restart_1012_reconnects_and_replays_audio():
    """fake +restart: fatal engine_restarting + close 1012 after the first frame (as an engine swap)."""
    async def go():
        async with session(key="k+restart") as (srv, ext):
            await ext.start_connection()
            for _ in range(30):
                await ext._handle_audio_frame(ext.ten_env, Frame())
                await asyncio.sleep(0.01)
            await wait_for(lambda: len(srv.connections) == 2 and ext.is_connected(), what="reconnect")
            await wait_for(lambda: srv.connections[1]["frames"] >= 5, what="audio on the new socket")
            assert (NON_FATAL, "engine_restarting") in _errors(ext)
            assert FATAL not in [c for c, _ in _errors(ext)] and not ext._fatal

    run(go())


def test_fake_idle_close_is_silent_then_next_audio_reconnects():
    """fake +idle<N>: fatal idle_timeout + close 1001 -> no asr_error, lazy reconnect on audio."""
    async def go():
        async with session(key="k+idle300") as (srv, ext):
            consumer = asyncio.create_task(ext._audio_frame_consumer())
            try:
                await ext.start_connection()
                await wait_for(lambda: ext._idle_closed and not ext.is_connected(), what="idle close")
                await asyncio.sleep(0.3)
                assert len(srv.connections) == 1 and ext._reconnect_task is None
                assert ext.send_asr_error.await_count == 0 and not ext._fatal
                for _ in range(10):
                    await ext.on_audio_frame(ext.ten_env, Frame())
                    await asyncio.sleep(0.01)
                await wait_for(lambda: len(srv.connections) == 2 and ext.is_connected(), what="lazy reconnect")
                await wait_for(lambda: srv.connections[1]["frames"] >= 5, what="buffered audio replayed")
                assert ext.send_asr_error.await_count == 0
            finally:
                consumer.cancel()

    run(go())


# ------------------------------------------------------- reconnect budget
class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_reconnect_budget_is_time_based_and_rate_limits_nonfatal(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    clock = _Clock()

    async def go():
        mgr = ReconnectManager(window_ms=120_000, report_interval_s=10, clock=clock)
        handler = AsyncMock()
        fail = AsyncMock(side_effect=RuntimeError("down"))
        for _ in range(30):  # 30 failures, 2 s apart: 58 s, inside the window
            await mgr.handle_reconnect(fail, handler, "nyo")
            clock.t += 2
        codes = [c.args[0].code for c in handler.await_args_list]
        assert set(codes) == {NON_FATAL} and 5 <= len(codes) <= 7  # first + one per 10 s
        assert mgr.can_retry()
        clock.t += 70  # outage now > 120 s
        await mgr.handle_reconnect(fail, handler, "nyo")
        assert handler.await_args_list[-1].args[0].code == FATAL and not mgr.can_retry()
        mgr.mark_connection_successful()
        assert mgr.can_retry() and mgr.outage_start is None

    run(go())


def test_backoff_is_capped_at_five_seconds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(side_effect=lambda d: sleeps.append(d)))

    async def go():
        mgr = ReconnectManager()
        for _ in range(8):
            await mgr.handle_reconnect(AsyncMock(side_effect=RuntimeError("x")))

    run(go())
    assert max(sleeps) <= 5.0 and sleeps[-1] >= 2.5


def test_retry_after_is_honoured_and_capped(monkeypatch):
    sleeps = []
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(side_effect=lambda d: sleeps.append(d)))

    async def go():
        mgr = ReconnectManager()
        mgr.set_retry_after(3)
        await mgr.handle_reconnect(AsyncMock())
        mgr.set_retry_after(120)
        await mgr.handle_reconnect(AsyncMock())
        await mgr.handle_reconnect(AsyncMock())  # hint is used once

    run(go())
    assert sleeps[0] == 3 and sleeps[1] == 15.0 and sleeps[2] < 3


def test_503_retry_after_is_read_from_the_handshake():
    """A test-local server that refuses the upgrade like serve/ does (503 + Retry-After)."""
    async def go():
        def refuse(conn, request):
            return Response(503, "Service Unavailable", Headers([("Retry-After", "5"), ("Content-Length", "0")]))

        async with serve(lambda ws: None, "127.0.0.1", 0, process_request=refuse) as srv:
            port = srv.sockets[0].getsockname()[1]
            rec = NYOASRRecognition("k", MagicMock(), MagicMock(), {"base_url": f"ws://127.0.0.1:{port}"}, AsyncMock())
            with pytest.raises(NRSPConnectError) as ei:
                await rec.start(timeout=5)
            assert ei.value.status_code == 503 and ei.value.retry_after == 5.0

            ext = make_extension(base_url=f"ws://127.0.0.1:{port}")
            ext._emit_connection_transition = AsyncMock()
            ext._preflight_ok = True
            ext.reconnect_manager = ReconnectManager()
            ext._schedule_reconnect = MagicMock()
            await ext.start_connection()
            assert ext.reconnect_manager.retry_after == 5.0 and not ext._fatal
            assert _errors(ext) == [(NON_FATAL, "503")]

    run(go())


def test_reconnect_window_is_a_validated_param():
    from .helpers import make_config
    assert make_config().params["reconnect_window_ms"] == 120_000
    with pytest.raises(ValueError, match="reconnect_window_ms"):
        make_config(reconnect_window_ms=-1).validate_config()


# ------------------------------------------------------------- finalize
def _rec(wait_result=True):
    rec = MagicMock()
    rec.is_connected.return_value = True
    rec.send_finalize = AsyncMock()
    rec.wait_finalize_end = AsyncMock(return_value=wait_result)
    return rec


def test_finalize_end_without_final_promotes_last_interim_once():
    async def go():
        ext = make_extension()
        ext.recognition = _rec(True)
        ext._last_interim = {"id": "utt-1", "text": "kumusta", "final": False, "start_ms": 0, "duration_ms": 400}
        ext.finalize_id = "f1"
        await ext.finalize("s")
        finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
        assert [f.text for f in finals] == ["kumusta"] and finals[0].metadata["asr_info"]["promoted_on_timeout"]
        assert ext._last_interim is None
        ext.finalize_id = "f2"
        await ext.finalize("s")  # nothing left to promote: one empty final answers f2
        finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
        assert [f.text for f in finals] == ["kumusta", ""]
        # the replayed engine's late final for the promoted turn is dropped
        await ext.on_result({"id": "r1-utt-0", "text": "kumusta po", "final": True, "start_ms": 0})
        assert len([c for c in ext.send_asr_result.await_args_list if c.args[0].final]) == 2
        assert ext.send_asr_finalize_end.await_count == 2

    run(go())


def test_finalize_end_after_a_final_does_not_promote():
    async def go():
        ext = make_extension()
        rec = _rec(True)

        async def wait(fid, _t):
            await ext.on_result({"id": "utt-1", "text": "kumusta ka", "final": True})
            return True
        rec.wait_finalize_end = wait
        ext.recognition = rec
        ext._last_interim = {"id": "utt-1", "text": "kumusta", "final": False}
        await ext.finalize("s")
        finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
        assert [f.text for f in finals] == ["kumusta ka"]

    run(go())


def test_finalize_when_disconnected_uses_its_own_id():
    async def go():
        ext = make_extension()
        ext.recognition = None
        ids = []
        ext.send_asr_finalize_end = AsyncMock(side_effect=lambda: ids.append(ext.finalize_id))
        ext.finalize_id = "x1"
        await ext.finalize("s")
        assert ids == ["x1"]

    run(go())


def test_nemotron_default_finalize_timeout_exceeds_the_gateway_30s():
    cfg = NYOASRConfig.model_validate_json(
        '{"params": {"api_key": "k", "base_url": "wss://h", "model": "%s"}}' % NEMOTRON_MODEL)
    cfg.apply_defaults()
    assert cfg.finalize_timeout_ms == NEMOTRON_FINALIZE_TIMEOUT_MS > 30_000
    cfg = NYOASRConfig.model_validate_json(
        '{"finalize_timeout_ms": 2500, "params": {"api_key": "k", "base_url": "wss://h", "model": "%s"}}' % NEMOTRON_MODEL)
    cfg.apply_defaults()
    assert cfg.finalize_timeout_ms == 2500  # the user's value wins
    cfg = NYOASRConfig.model_validate_json('{"params": {"api_key": "k", "base_url": "wss://h"}}')
    cfg.apply_defaults()
    assert cfg.finalize_timeout_ms == 3500  # streaming models: above the gateway's 3000 ms


def test_shipped_property_json_leaves_finalize_timeout_to_the_model_default():
    import json
    import pathlib
    prop = json.loads((pathlib.Path(__file__).resolve().parents[1] / "property.json").read_text())
    assert "finalize_timeout_ms" not in prop and prop["params"]["reconnect_window_ms"] == 120_000
    assert "gc_freeze" not in prop["params"]


# ------------------------------------------------------------- lifecycle
def test_on_deinit_awaits_the_background_close():
    async def go():
        ext = make_extension()
        closed = []

        async def slow_close():
            await asyncio.sleep(0.05)
            closed.append(1)
        rec = MagicMock()
        rec.close = slow_close
        ext.recognition = rec
        await ext._send_error(MagicMock(code=FATAL, message="x"), None)  # spawns rec.close()
        assert ext._bg_tasks and not closed
        await ext.on_deinit(ext.ten_env)
        assert closed == [1] and not ext._bg_tasks

    run(go())


def test_stop_during_reconnect_waits_for_the_task():
    async def go():
        ext = make_extension()
        started = asyncio.Event()

        async def hang():
            started.set()
            await asyncio.sleep(30)
        ext._reconnect_task = asyncio.create_task(hang())
        await started.wait()
        task = ext._reconnect_task
        await ext.stop_connection()
        assert task.done() and ext._reconnect_task is None

    run(go())

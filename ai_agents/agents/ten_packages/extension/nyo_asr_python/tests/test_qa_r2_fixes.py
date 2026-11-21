"""QA round 2 fixes: conclusive model checks (rule A/B), stable-connection outage reset (rule C),
rate-limited reconnectable errors (rule D), finalize attribution and cancellation (N5), defaults (N4)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nyo_asr_python import transport
from nyo_asr_python.config import NYOASRConfig
from nyo_asr_python.reconnect_manager import ReconnectManager

from .helpers import make_extension, run

FATAL = -1000
NON_FATAL = 1000
URL = "ws://127.0.0.1:9200"


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ------------------------------------------------------------------ rule A
def test_listing_without_asr_models_is_not_conclusive():
    tts_only = [{"id": "halohalo/tts-fil", "kind": "tts", "language": "fil"}]
    assert not transport.lists_kind(tts_only, "asr")
    assert transport.check_model(tts_only, "halohalo/asr-fil", "asr", "fil", URL) is None
    assert transport.check_model([], "halohalo/asr-fil", "asr", "fil", URL) is None
    nemo = [{"id": "nvidia/nemotron-3.5-transcribe", "kind": "asr"}]     # what serve/ lists during an ASR swap
    assert transport.check_model(nemo, "halohalo/asr-fil", "asr", "fil", URL) is None
    assert not transport.lists_kind(nemo, "asr", "fw/whisper-base") and transport.lists_kind(nemo, "asr")
    asr = tts_only + [{"id": "halohalo/asr-hil", "kind": "asr", "language": "hil"}]
    code, msg = transport.check_model(asr, "halohalo/asr-fil", "asr", "fil", URL)
    assert code == "model_not_found" and "halohalo/asr-hil" in msg


def test_language_string_field_is_checked():
    entries = [{"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"}]
    assert transport.check_model(entries, "halohalo/asr-fil", "asr", "fil", URL) is None
    code, msg = transport.check_model(entries, "halohalo/asr-fil", "asr", "hil", URL)
    assert code == "language_not_supported" and "fil" in msg
    assert transport.model_languages({"languages": ["fil", "hil"], "language": "eng"}) == ["fil", "hil", "eng"]


@pytest.mark.parametrize("entries,fatal,ok,errors", [
    ([{"id": "halohalo/tts-fil", "kind": "tts", "language": "fil"}], False, False, []),   # swap / boot: inconclusive
    ([{"id": "nvidia/nemotron-3.5-transcribe", "kind": "asr"}], False, False, []),        # ASR swap: Nemotron stays listed
    # rule A': missing while same-namespace models are listed = typo or that engine down: NON_FATAL, retried
    ([{"id": "halohalo/asr-hil", "kind": "asr", "language": "hil"}], False, False, [(NON_FATAL, "model_not_found")]),
    # listed, but not with our language / not as ASR: conclusive, FATAL at once
    ([{"id": "halohalo/asr-fil", "kind": "asr", "language": "hil"}], True, False, [(FATAL, "language_not_supported")]),
    ([{"id": "halohalo/asr-fil", "kind": "tts", "language": "fil"}], True, False, [(FATAL, "model_not_found")]),
    ([{"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"}], False, True, []),
])
def test_preflight_decision(monkeypatch, entries, fatal, ok, errors):
    ext = make_extension(model="halohalo/asr-fil", language="fil")
    ext.on_disconnected = AsyncMock()
    monkeypatch.setattr(transport, "fetch_models", AsyncMock(return_value=entries))
    assert run(ext._preflight()) is (not fatal)
    assert ext._fatal is fatal and ext._preflight_ok is ok
    assert [(c.args[0].code, c.args[1].code) for c in ext.send_asr_error.await_args_list] == errors


def test_preflight_non_200_is_inconclusive(monkeypatch):
    ext = make_extension()
    monkeypatch.setattr(transport, "fetch_models",
                        AsyncMock(side_effect=transport.TransportError("models_http_error", "503", 503)))
    assert run(ext._preflight()) and not ext._fatal and not ext._preflight_ok


# ------------------------------------------------------------------ rule B
def test_handshake_404_retries_inside_the_window_while_models_are_inconclusive(monkeypatch):
    from nyo_asr_python.recognition import NRSPConnectError
    ext = make_extension()
    monkeypatch.setattr(transport, "fetch_models", AsyncMock(return_value=[]))
    for _ in range(6):
        run(ext._report_connect_failure(NRSPConnectError("HTTP 404", 404)))
    assert not ext._fatal and {c.args[0].code for c in ext.send_asr_error.await_args_list} == {NON_FATAL}
    # rule A': listed namespace but not our model -> still NON_FATAL, vendor code model_not_found
    monkeypatch.setattr(transport, "fetch_models",
                        AsyncMock(return_value=[{"id": "halohalo/asr-hil", "kind": "asr", "language": "hil"}]))
    run(ext._report_connect_failure(NRSPConnectError("HTTP 404", 404)))
    err, vi = ext.send_asr_error.await_args.args
    assert not ext._fatal and err.code == NON_FATAL and vi.code == "model_not_found"
    # listed without our language -> conclusive FATAL
    monkeypatch.setattr(transport, "fetch_models",
                        AsyncMock(return_value=[{"id": "halohalo/asr-fil", "kind": "asr", "language": "hil"}]))
    run(ext._report_connect_failure(NRSPConnectError("HTTP 400", 400)))
    err, vi = ext.send_asr_error.await_args.args
    assert ext._fatal and err.code == FATAL and vi.code == "language_not_supported"


def test_missing_model_reported_once_per_window_then_fatal_model_not_found(monkeypatch):
    """Rule A' end to end: the engine of our model stays down for the whole window."""
    from nyo_asr_python.recognition import NRSPConnectError
    async def go():
        ext = make_extension(model="halohalo/asr-hil", language="hil")
        ext._emit_connection_transition = AsyncMock()
        ext.reconnect_manager = ReconnectManager(base_delay=0.01, max_delay=0.02, window_ms=300)
        monkeypatch.setattr(transport, "fetch_models",
                            AsyncMock(return_value=[{"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"}]))
        ext._connect_recognition = AsyncMock(side_effect=NRSPConnectError("HTTP 404", 404))
        await ext.start_connection()
        for _ in range(200):
            if ext._reconnect_task is None:
                break
            await asyncio.sleep(0.01)
        codes = [(c.args[0].code, c.args[1].code) for c in ext.send_asr_error.await_args_list]
        assert codes == [(NON_FATAL, "model_not_found"), (FATAL, "model_not_found")], codes
        assert ext._fatal and ext.reconnect_manager.attempts >= 3

    run(go())


def test_missing_model_engine_back_inside_the_window_connects(monkeypatch):
    """Rule A': the engine comes back (model listed again) before the window ends: no FATAL."""
    async def go():
        ext = make_extension(model="halohalo/asr-hil", language="hil")
        ext._emit_connection_transition = AsyncMock()
        ext.reconnect_manager = ReconnectManager(base_delay=0.01, max_delay=0.02, window_ms=5000)
        listings = [[{"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"}]] * 3
        listings.append([{"id": "halohalo/asr-hil", "kind": "asr", "language": "hil"}])
        monkeypatch.setattr(transport, "fetch_models",
                            AsyncMock(side_effect=lambda *a: listings.pop(0) if len(listings) > 1 else listings[0]))
        async def connect():
            if len(listings) > 1:
                from nyo_asr_python.recognition import NRSPConnectError
                raise NRSPConnectError("HTTP 404", 404)
        ext._connect_recognition = AsyncMock(side_effect=connect)
        await ext.start_connection()
        for _ in range(300):
            if ext._reconnect_task is None:
                break
            await asyncio.sleep(0.01)
        codes = [c.args[0].code for c in ext.send_asr_error.await_args_list]
        assert FATAL not in codes and codes.count(NON_FATAL) == 1 and ext._preflight_ok

    run(go())


# ------------------------------------------------------------------ rule C
def test_outage_resets_only_after_a_stable_connection():
    clock = _Clock()
    mgr = ReconnectManager(window_ms=60_000, clock=clock)
    mgr.start_outage(); mgr.attempts = 4
    mgr.note_open(); clock.t += 3; mgr.note_closed()          # up 3 s, nothing delivered
    assert mgr.outage_start is not None and mgr.attempts == 4
    mgr.note_open(); clock.t += 10; mgr.note_closed()         # up 10 s
    assert mgr.outage_start is None and mgr.attempts == 0
    mgr.start_outage(); mgr.attempts = 2
    mgr.note_open(); mgr.mark_stable()                       # a result / finalize_end
    assert mgr.outage_start is None and mgr.attempts == 0


def test_result_and_finalize_end_mark_the_connection_stable():
    ext = make_extension()
    ext.reconnect_manager = ReconnectManager()
    ext.reconnect_manager.start_outage(); ext.reconnect_manager.attempts = 3
    run(ext.on_finalize_end("auto-1"))
    assert ext.reconnect_manager.attempts == 0
    ext.reconnect_manager.start_outage(); ext.reconnect_manager.attempts = 3
    run(ext.on_result({"type": "result", "id": "utt-1", "text": "a", "final": False}))
    assert ext.reconnect_manager.attempts == 0


# ------------------------------------------------------------------ rule D
def test_reconnectable_server_errors_are_rate_limited_per_outage():
    clock = _Clock()
    ext = make_extension()
    ext._schedule_reconnect = MagicMock()
    ext.reconnect_manager = ReconnectManager(clock=clock)
    for _ in range(5):
        run(ext.on_error("x", "engine_unavailable", True, "nyo", "engine_unavailable"))
        clock.t += 1
    assert [c.args[0].code for c in ext.send_asr_error.await_args_list] == [NON_FATAL]
    clock.t += 10
    run(ext.on_error("x", "engine_restarting", True, "nyo", "engine_restarting"))
    assert ext.send_asr_error.await_count == 2 and not ext._fatal


# ------------------------------------------------------------------ N4
def test_default_finalize_timeout_is_above_the_gateway_3000ms():
    cfg = NYOASRConfig.model_validate_json('{"params": {"api_key": "k", "base_url": "wss://h"}}')
    cfg.apply_defaults()
    assert cfg.finalize_timeout_ms == 3500
    cfg = NYOASRConfig.model_validate_json('{"finalize_timeout_ms": 1200, "params": {"api_key": "k", "base_url": "wss://h"}}')
    cfg.apply_defaults()
    assert cfg.finalize_timeout_ms == 1200


# ------------------------------------------------------------------ N5
class _Rec:
    """Recognition stub: finalize_end events driven by the test."""
    def __init__(self):
        self.events = {}

    def is_connected(self):
        return True

    async def send_finalize(self, fid):
        self.events[fid] = asyncio.Event()

    async def wait_finalize_end(self, fid, timeout_ms):
        await self.events[fid].wait()
        return True


def test_final_after_a_second_finalize_started_counts_for_the_oldest():
    """A's final arrives after B started; B then ends without its own final: B must promote its interim."""
    async def go():
        ext = make_extension(finalize_timeout_ms=2000)
        rec = _Rec()
        ext.recognition = rec
        ext.audio_timeline.add_user_audio(1000)   # turn one sent
        ext.finalize_id = "A"
        ta = asyncio.create_task(ext.finalize(None))
        await asyncio.sleep(0)
        ext.audio_timeline.add_user_audio(1000)   # turn two sent
        ext.finalize_id = "B"
        tb = asyncio.create_task(ext.finalize(None))
        await asyncio.sleep(0)
        await ext.on_result({"type": "result", "id": "utt-1", "text": "turn one", "final": True,
                             "start_ms": 0, "duration_ms": 1000})
        await ext.on_finalize_end("A"); rec.events["A"].set()
        await ta
        # B's turn: its audio was sent before B (start_ms 1000 < 2000 sent), its interim arrives after B was sent
        await ext.on_result({"type": "result", "id": "utt-2", "text": "turn two so far", "final": False,
                             "start_ms": 1000, "duration_ms": 800})
        await ext.on_finalize_end("B"); rec.events["B"].set()       # engine lost: no final for B
        await tb
        finals = [c.args[0].text for c in ext.send_asr_result.await_args_list if c.args[0].final]
        assert finals == ["turn one", "turn two so far"]

    run(go())


def test_final_naming_its_finalize_id_is_attributed_to_it():
    ext = make_extension()
    ext._fin_finals = {"A": 0, "B": 0}
    ext._attribute_final({"final": True, "metadata": {"finalize_id": "B"}})
    assert ext._fin_finals == {"A": 0, "B": 1}
    ext._fin_ended.add("A")
    ext._attribute_final({"final": True})
    assert ext._fin_finals == {"A": 0, "B": 2}

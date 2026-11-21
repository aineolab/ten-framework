import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from .helpers import make_extension, run


def _connected(ext, wait_result=True):
    rec = MagicMock()
    rec.is_connected.return_value = True
    rec.send_finalize = AsyncMock()
    rec.wait_finalize_end = AsyncMock(return_value=wait_result)
    ext.recognition = rec
    return rec


def test_finalize_sends_nrsp_finalize_with_finalize_id_and_emits_end():
    async def go():
        ext = make_extension()
        rec = _connected(ext)
        ext.finalize_id = "fin-7"
        await ext.finalize("s")
        rec.send_finalize.assert_awaited_once_with("fin-7")
        rec.wait_finalize_end.assert_awaited_once_with("fin-7", 50)
        ext.send_asr_finalize_end.assert_awaited_once()
        assert ext.last_finalize_timestamp == 0

    run(go())


def test_finalize_timeout_promotes_last_interim_then_emits_end():
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._last_interim = {"text": "partial words", "backend": "ctc"}
        ext._emit_asr_result = AsyncMock()
        await ext.finalize("s")
        args, kw = ext._emit_asr_result.await_args
        assert args[0]["text"] == "partial words" and kw["final"] is True and kw["promoted"] == "timeout"
        assert ext._last_interim is None
        ext.send_asr_finalize_end.assert_awaited_once()

    run(go())


def test_finalize_timeout_without_interim_still_emits_end():
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._emit_asr_result = AsyncMock()
        await ext.finalize("s")
        ext._emit_asr_result.assert_not_awaited()
        ext.send_asr_finalize_end.assert_awaited_once()

    run(go())


def test_finalize_send_error_still_emits_end():
    async def go():
        ext = make_extension()
        rec = _connected(ext)
        rec.send_finalize.side_effect = RuntimeError("socket gone")
        ext._emit_asr_result = AsyncMock()
        await ext.finalize("s")
        ext.send_asr_finalize_end.assert_awaited_once()

    run(go())


def test_finalize_cancelled_still_emits_end():
    async def go():
        ext = make_extension()
        rec = _connected(ext)
        rec.wait_finalize_end.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):  # cancellation propagates (QA r2 N5)
            await ext.finalize("s")
        ext.send_asr_finalize_end.assert_awaited_once()

    run(go())


def test_finalize_when_disconnected_emits_end_without_waiting():
    async def go():
        ext = make_extension()
        ext.recognition = MagicMock()
        ext.recognition.is_connected.return_value = False
        await ext.finalize("s")
        ext.send_asr_finalize_end.assert_awaited_once()
        ext.recognition.send_finalize.assert_not_called()

    run(go())


def test_finalize_with_no_recognition_emits_end():
    async def go():
        ext = make_extension()
        ext.recognition = None
        await ext.finalize("s")
        ext.send_asr_finalize_end.assert_awaited_once()

    run(go())


def test_late_real_final_after_timeout_promotion_is_dropped():
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._last_interim = {"text": "partial", "backend": "ctc"}
        await ext.finalize("s")                      # promotes "partial"
        n = ext.send_asr_result.await_count
        await ext.on_result({"text": "partial words", "final": True})   # late real final
        assert ext.send_asr_result.await_count == n  # no second final for the turn
        await ext.on_result({"text": "next turn", "final": False})
        await ext.on_result({"text": "next turn", "final": True})
        assert ext.send_asr_result.await_count == n + 2

    run(go())


def _finals(ext):
    return [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]


def test_replayed_turn_after_engine_reopen_is_not_a_second_final():
    """ten review F3 (D28): promoted on timeout, then the gateway re-opens the engine on the same socket and
    replays the turn's audio; the re-decode has r1- ids and a server finalize id, but the same session clock."""
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._last_interim = {"id": "utt-3", "text": "kumusta", "final": False, "start_ms": 1000, "duration_ms": 2000}
        ext.finalize_id = "fin-1"
        await ext.finalize("s")                                       # promotes "kumusta"
        await ext.on_result({"id": "r1-utt-0", "text": "kumusta ka", "final": False,
                             "start_ms": 1000, "duration_ms": 900})
        await ext.on_result({"id": "r1-utt-0", "text": "kumusta ka na", "final": True, "start_ms": 1000,
                             "duration_ms": 2400, "metadata": {"finalize_id": "r1-auto-0", "endpoint": "auto"}})
        await ext.on_result({"id": "r1-utt-1", "text": "salamat", "final": True, "start_ms": 3600, "duration_ms": 800})
        assert [f.text for f in _finals(ext)] == ["kumusta", "salamat"]
        assert [c.args[0].text for c in ext.send_asr_result.await_args_list if not c.args[0].final] == []
        assert not ext._suppress_late_final

    run(go())


def test_late_final_matched_by_its_finalize_id_is_dropped():
    """A late final whose utterance id changed but which carries the promoted finalize id is the same turn."""
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._last_interim = {"id": "utt-3", "text": "kumusta", "final": False}
        ext.finalize_id = "fin-2"
        await ext.finalize("s")
        await ext.on_result({"id": "r1-utt-0", "text": "kumusta ka", "final": True,
                             "metadata": {"finalize_id": "fin-2"}})
        await ext.on_result({"id": "r1-utt-1", "text": "salamat", "final": True,
                             "metadata": {"finalize_id": "r1-auto-1"}})
        assert [f.text for f in _finals(ext)] == ["kumusta", "salamat"]

    run(go())


def test_socket_lost_during_finalize_promotes_without_suppressing_the_next_socket():
    """ten review F3: the socket dropped while waiting (close() sets every finalize event). The real final can
    never arrive, so nothing is suppressed: the next socket restarts its ids and its utt-3 is a new turn."""
    async def go():
        ext = make_extension()
        rec = _connected(ext)

        async def dropped(fid, timeout_ms):
            rec.is_connected.return_value = False
            return False

        rec.wait_finalize_end.side_effect = dropped
        ext._last_interim = {"id": "utt-3", "text": "kumusta", "final": False, "start_ms": 0, "duration_ms": 900}
        ext.finalize_id = "fin-3"
        await ext.finalize("s")
        assert not ext._suppress_late_final
        await ext.on_result({"id": "utt-3", "text": "salamat po", "final": False, "start_ms": 0, "duration_ms": 400})
        await ext.on_result({"id": "utt-3", "text": "salamat po", "final": True, "start_ms": 0, "duration_ms": 700})
        assert [f.text for f in _finals(ext)] == ["kumusta", "salamat po"]
        ext.send_asr_finalize_end.assert_awaited_once()

    run(go())


def test_on_open_disarms_a_pending_late_final_suppression():
    """A new socket is a new server session (new ids, new timeline): an armed suppression must not survive it."""
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._last_interim = {"id": "utt-0", "text": "kumusta", "final": False, "start_ms": 0, "duration_ms": 900}
        await ext.finalize("s")
        assert ext._suppress_late_final
        await ext.on_open()
        assert not ext._suppress_late_final and ext._promoted_id is None
        await ext.on_result({"id": "utt-0", "text": "magandang umaga", "final": True, "start_ms": 0, "duration_ms": 900})
        assert [f.text for f in _finals(ext)] == ["kumusta", "magandang umaga"]

    run(go())


def test_new_turn_starting_inside_the_promoted_span_keeps_its_final():
    """architect r3 F1: the span rule is only for replayed results (gateway `r<N>-` ids from a newer engine
    session); a plain-id final of another utterance that starts before the promoted end is a new turn."""
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._last_interim = {"id": "utt-3", "text": "kumusta", "final": False, "start_ms": 1000, "duration_ms": 2000}
        await ext.finalize("s")
        await ext.on_result({"id": "utt-4", "text": "oo nga", "final": True, "start_ms": 2500, "duration_ms": 600})
        assert [f.text for f in _finals(ext)] == ["kumusta", "oo nga"]

    run(go())


def test_span_rule_needs_a_newer_reopen_generation():
    """Promoted on an already re-opened session (r1-): only r2- (a later re-open) replays it."""
    async def go():
        ext = make_extension()
        _connected(ext, wait_result=False)
        ext._last_interim = {"id": "r1-utt-3", "text": "kumusta", "final": False, "start_ms": 1000,
                             "duration_ms": 2000}
        await ext.finalize("s")
        await ext.on_result({"id": "r1-utt-4", "text": "oo nga", "final": True, "start_ms": 2500, "duration_ms": 600})
        assert [f.text for f in _finals(ext)] == ["kumusta", "oo nga"]
        ext._last_interim = {"id": "r1-utt-5", "text": "salamat", "final": False, "start_ms": 4000,
                             "duration_ms": 900}
        await ext.finalize("s")
        await ext.on_result({"id": "r2-utt-0", "text": "salamat po", "final": True, "start_ms": 4000,
                             "duration_ms": 1200})
        assert [f.text for f in _finals(ext)] == ["kumusta", "oo nga", "salamat"]

    run(go())

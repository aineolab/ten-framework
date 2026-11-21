"""QA r5 fix 2: an unexpected socket close promotes the turn in progress (its last interim) once, as a final marked
asr_info.promoted_on_close. No late-final suppression is armed: the next socket is a new gateway session that never
re-decodes audio sent on the dead one, and its ids restart (utt-0 again)."""
from unittest.mock import MagicMock

from .helpers import make_extension, run
from .test_protocol import Frame, session, wait_for


def _finals(ext):
    return [r for r in (c.args[0] for c in ext.send_asr_result.await_args_list) if r.final]


def test_r5_unexpected_close_promotes_the_last_interim_once():
    async def go():
        ext = make_extension()
        ext._schedule_reconnect = MagicMock()
        ext.recognition = MagicMock()
        await ext.on_result({"type": "result", "id": "utt-0", "text": "magandang", "final": False,
                             "start_ms": 0, "duration_ms": 600})
        await ext.on_result({"type": "result", "id": "utt-0", "text": "magandang umaga", "final": False,
                             "start_ms": 0, "duration_ms": 1200})
        await ext.on_close(False, "closed: 1012 engine_restarting")
        finals = _finals(ext)
        assert [f.text for f in finals] == ["magandang umaga"]
        assert finals[0].metadata["asr_info"].get("promoted_on_close") is True
        assert not ext._suppress_late_final
        ext._schedule_reconnect.assert_called_once()
        # finalize while still disconnected: nothing left to promote (once)
        ext.finalize_id = "f1"
        await ext.finalize(None)
        assert len(_finals(ext)) == 1 and ext.send_asr_finalize_end.await_count == 1
        # the new socket's first turn reuses id utt-0: it is not suppressed
        await ext.on_result({"type": "result", "id": "utt-0", "text": "pupunta ako", "final": True,
                             "start_ms": 0, "duration_ms": 900, "metadata": {"finalize_id": "auto-1"}})
        assert [f.text for f in _finals(ext)] == ["magandang umaga", "pupunta ako"]

    run(go())


def test_r5_expected_close_stop_or_fatal_does_not_promote():
    async def go():
        for setup in ("expected", "stop", "fatal"):
            ext = make_extension()
            ext._schedule_reconnect = MagicMock()
            await ext.on_result({"type": "result", "id": "utt-0", "text": "kumusta", "final": False,
                                 "start_ms": 0, "duration_ms": 600})
            if setup == "stop":
                ext._stop_requested = True
            if setup == "fatal":
                ext._fatal = True
            await ext.on_close(setup == "expected", "closed")
            assert not _finals(ext), setup

    run(go())


def test_r5_idle_close_promotes_only_when_an_interim_is_pending():
    async def go():
        ext = make_extension()
        ext._schedule_reconnect = MagicMock()
        await ext.on_error("no client message for 300 s", "idle_timeout", True, "halohalo", "idle_timeout")
        await ext.on_close(False, "closed: 1001")
        assert not _finals(ext) and not ext.send_asr_error.await_args_list
        ext._schedule_reconnect.assert_not_called()  # the next audio frame reconnects

    run(go())


def test_r5_only_the_next_turn_interim_is_promoted_on_close_after_its_final():
    """A final ends its own utterance's interim only: the next turn's interim survives it and is what a close
    promotes."""
    async def go():
        ext = make_extension()
        ext._schedule_reconnect = MagicMock()
        await ext.on_result({"type": "result", "id": "utt-0", "text": "magandang umaga", "final": False,
                             "start_ms": 0, "duration_ms": 1500})
        await ext.on_result({"type": "result", "id": "utt-1", "text": "pupunta", "final": False,
                             "start_ms": 1600, "duration_ms": 500})
        await ext.on_result({"type": "result", "id": "utt-0", "text": "magandang umaga po", "final": True,
                             "start_ms": 0, "duration_ms": 1500, "metadata": {"finalize_id": "auto-1"}})
        await ext.on_close(False, "closed: abrupt")
        assert [f.text for f in _finals(ext)] == ["magandang umaga po", "pupunta"]

    run(go())


def test_r5_fake_server_drop_mid_utterance_keeps_its_words():
    """Real client + fake server: the socket drops mid-utterance (no finalize); its interim is promoted once and
    the reconnected socket's turn is kept."""
    async def go():
        async with session(key="k+drop25", language="fil") as (srv, ext):
            await ext.start_connection()
            for _ in range(20):  # 400 ms of speech -> interim at 320 ms
                await ext.send_audio(Frame(), "s")
            await wait_for(lambda: any(not f for k, f in ext.order if k == "result"), what="interim")
            for _ in range(5):  # the 25th frame drops the socket
                await ext.send_audio(Frame(), "s")
            await wait_for(lambda: len(_finals(ext)) == 1, what="promoted final")
            assert _finals(ext)[0].metadata["asr_info"].get("promoted_on_close") is True
            await wait_for(lambda: len(srv.connections) == 2 and ext.is_connected(), what="reconnect")
            for _ in range(20):
                await ext.send_audio(Frame(), "s")
            ext.finalize_id = "fin-b"
            await ext.finalize("s")
            finals = _finals(ext)
            assert len(finals) == 2 and not finals[1].metadata["asr_info"].get("promoted_on_close"), finals

    run(go())

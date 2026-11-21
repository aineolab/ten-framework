"""QA r6: edges of interim promotion (per-result-id tracking, close promotion, re-open suppression).

asr_fw decodes finals in a worker separate from interims (modules/asr_fw/server.py _final_worker vs _interim), so the
next utterance's interim (utt-N+1) can arrive before the previous endpointed utterance's final (utt-N). The addon then
holds two pending interims of two different turns. A promotion removes only interims inside the promoted span; only a
final of another, later turn ends late-final suppression.
"""
import asyncio
from unittest.mock import MagicMock

from nyo_asr_python import transport

from .helpers import make_config, make_extension, run


def _finals(ext):
    return [r for r in (c.args[0] for c in ext.send_asr_result.await_args_list) if r.final]


def _rec():
    rec = MagicMock()
    rec.is_connected = lambda: True
    return rec


def test_r6_close_with_two_pending_turns_keeps_both():
    """Close while turn A's endpointed final is still decoding and turn B already has an interim: both turns die
    with the socket, both are promoted (start order, promoted_on_close)."""
    async def go():
        ext = make_extension(endpointing_ms=1000)
        ext._schedule_reconnect = MagicMock()
        ext.recognition = MagicMock()
        await ext.on_result({"type": "result", "id": "utt-3", "text": "magandang umaga po", "final": False,
                             "start_ms": 0, "duration_ms": 2000})
        await ext.on_result({"type": "result", "id": "utt-4", "text": "pupunta ako", "final": False,
                             "start_ms": 2300, "duration_ms": 1600})
        await ext.on_close(False, "closed: 1012 engine_restarting")
        finals = _finals(ext)
        assert [f.text for f in finals] == ["magandang umaga po", "pupunta ako"]
        assert all(f.metadata["asr_info"].get("promoted_on_close") for f in finals)
        assert not ext._interims and not ext._suppress_late_final

    run(go())


def test_r6_close_promotes_a_reopen_duplicate_once():
    """The original session's interim and the re-opened engine's interim of the same audio are one turn: a close
    promotes it once."""
    async def go():
        ext = make_extension(endpointing_ms=1000)
        ext._schedule_reconnect = MagicMock()
        ext.recognition = MagicMock()
        await ext.on_result({"type": "result", "id": "utt-2", "text": "salamat", "final": False,
                             "start_ms": 1000, "duration_ms": 900})
        await ext.on_result({"type": "result", "id": "r1-utt-0", "text": "salamat po", "final": False,
                             "start_ms": 1000, "duration_ms": 1500})
        await ext.on_close(False, "closed")
        assert [f.text for f in _finals(ext)] == ["salamat po"]

    run(go())


def test_r6_reopen_next_turn_interim_keeps_suppression():
    """Engine lost during finalize f1: T's interim promoted, suppression armed. In the replay, the next turn U's
    interim arrives before T's re-decoded auto final: U's interim is emitted, T's re-decode is still dropped."""
    async def go():
        ext = make_extension(endpointing_ms=1000)
        rec = _rec()
        ext.recognition = rec
        await ext.on_result({"type": "result", "id": "utt-0", "text": "pupunta ako sa palengke", "final": False,
                             "start_ms": 0, "duration_ms": 1500})
        ext._open_finalizes.append("f1")
        ext._fin_finals["f1"] = 0
        ext._client_fids.append("f1")
        await ext._promote_last_interim("f1", rec, "finalize_end without a final")
        ext._open_finalizes.remove("f1")
        ext._fin_finals.pop("f1")
        await ext.on_result({"type": "result", "id": "r1-utt-1", "text": "bibili ako", "final": False,
                             "start_ms": 2400, "duration_ms": 1600})
        await ext.on_result({"type": "result", "id": "r1-utt-0", "text": "pupunta ako sa palengke bukas",
                             "final": True, "start_ms": 0, "duration_ms": 2100,
                             "metadata": {"finalize_id": "r1-auto-1", "endpoint": "auto"}})
        out = [(r.final, r.text) for r in (c.args[0] for c in ext.send_asr_result.await_args_list)]
        assert sum("palengke" in t for f, t in out if f) == 1, out
        assert (False, "bibili ako") in out, out
        # U's own final is a later turn: emitted, and it ends the suppression
        await ext.on_result({"type": "result", "id": "r1-utt-1", "text": "bibili ako ng isda", "final": True,
                             "start_ms": 2400, "duration_ms": 2500,
                             "metadata": {"finalize_id": "r1-auto-2", "endpoint": "auto"}})
        assert [f.text for f in _finals(ext)][-1] == "bibili ako ng isda" and not ext._suppress_late_final

    run(go())


def test_r6_reopen_pending_auto_turn_redecode_is_kept():
    """Turn A endpointed (final still decoding), turn T finalized by f1, the engine is lost: finalize_end without a
    final promotes T only; A's interim stays pending. The replay re-decodes A (before T's span): emitted, and T's
    re-decode is still dropped."""
    async def go():
        ext = make_extension(endpointing_ms=1000)
        rec = _rec()
        ext.recognition = rec
        ext.audio_timeline.add_user_audio(6000)
        await ext.on_result({"type": "result", "id": "utt-3", "text": "magandang umaga po", "final": False,
                             "start_ms": 0, "duration_ms": 2000})
        await ext.on_result({"type": "result", "id": "utt-4", "text": "pupunta ako", "final": False,
                             "start_ms": 2300, "duration_ms": 1600})
        ext._fin_snap["f1"] = (ext._latest_interim_key(), 6000)
        ext._open_finalizes.append("f1")
        ext._fin_finals["f1"] = 0
        ext._client_fids.append("f1")
        await ext._promote_last_interim("f1", rec, "finalize_end without a final")
        ext._open_finalizes.remove("f1")
        ext._fin_finals.pop("f1")
        ext._fin_snap.pop("f1")
        assert list(ext._interims) == ["utt-3"]
        await ext.on_result({"type": "result", "id": "r1-utt-0", "text": "magandang umaga po", "final": True,
                             "start_ms": 0, "duration_ms": 2200,
                             "metadata": {"finalize_id": "r1-auto-1", "endpoint": "auto"}})
        await ext.on_result({"type": "result", "id": "r1-utt-1", "text": "pupunta ako sa palengke", "final": True,
                             "start_ms": 2250, "duration_ms": 1900,
                             "metadata": {"finalize_id": "r1-auto-2", "endpoint": "auto"}})
        assert [f.text for f in _finals(ext)] == ["pupunta ako", "magandang umaga po"]
        assert not ext._interims

    run(go())


def test_r6_close_during_finalize_wait_is_promoted_on_close():
    """The socket closes while a finalize waits: the woken finalize labels its promotion promoted_on_close."""
    async def go():
        ext = make_extension(finalize_timeout_ms=2000)
        rec = MagicMock()
        state = {"up": True}
        rec.is_connected = lambda: state["up"]
        rec.send_finalize = MagicMock(side_effect=lambda fid: asyncio.sleep(0))

        async def wait(fid, ms):
            state["up"] = False
            return False
        rec.wait_finalize_end = wait
        ext.recognition = rec
        await ext.on_result({"type": "result", "id": "utt-0", "text": "kumusta ka", "final": False,
                             "start_ms": 0, "duration_ms": 900})
        ext.finalize_id = "f1"
        await ext.finalize(None)
        finals = _finals(ext)
        assert [f.text for f in finals] == ["kumusta ka"]
        info = finals[0].metadata["asr_info"]
        assert info.get("promoted_on_close") and not info.get("promoted_on_timeout"), info
        assert not ext._suppress_late_final and ext.send_asr_finalize_end.await_count == 1

    run(go())


def test_r6_language_only_nemotron_pick_uses_the_nemotron_finalize_default(monkeypatch):
    entries = [{"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"},
               {"id": "nvidia/nemotron-3.5-transcribe", "kind": "asr", "languages": ["en"]}]

    async def fake(*a, **k):
        return entries
    monkeypatch.setattr(transport, "fetch_models", fake)
    for explicit in (False, True):
        ext = make_extension(language="en")
        if not explicit:  # helpers always set finalize_timeout_ms: model an unset one
            cfg = make_config(language="en")
            cfg.model_fields_set.discard("finalize_timeout_ms")
            ext.config = cfg
        v = run(ext._models_verdict())
        assert v[0] == "ok" and ext.config.params["model"] == "nvidia/nemotron-3.5-transcribe", v
        assert ext.config.finalize_timeout_ms == (50 if explicit else 32_000), (explicit, ext.config.finalize_timeout_ms)

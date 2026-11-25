"""QA r4: late-final suppression after a gateway engine re-open swallows the NEXT turn.

Gateway (serve/gateway.py on_engine_lost): pending finalizes get finalize_end without a final, then the re-opened
engine session is fed the audio since the last final (ring replay), ids get an r<N>- prefix, start_ms stays on the
session clock (base_ms = end of the last final). With endpointing_ms=0 (TEN drives asr_finalize) the replayed turn-1
audio is not cut on its own: it becomes the head of turn 2's utterance, whose final starts inside the promoted span.
The addon promoted turn 1's interim (finalize_end without a final) and then drops every result whose start_ms is in
that span and whose id is from a newer re-open generation -- including turn 2, which carries its own finalize id.
"""
from unittest.mock import MagicMock

from .helpers import make_extension, run


def test_r4_turn_after_engine_reopen_is_not_swallowed():
    async def go():
        ext = make_extension(endpointing_ms=0)
        rec = MagicMock()
        rec.is_connected = lambda: True
        ext.recognition = rec
        # turn 1: interim from the original engine session, then the engine dies during finalize f1
        await ext.on_result({"type": "result", "id": "utt-0", "text": "pupunta ako", "final": False,
                             "start_ms": 0, "duration_ms": 1500, "language": "fil"})
        ext._open_finalizes.append("f1")
        ext._fin_finals["f1"] = 0
        await ext._promote_last_interim("f1", rec, "finalize_end without a final")
        ext._open_finalizes.remove("f1")
        ext._fin_finals.pop("f1")
        # turn 2 on the re-opened engine session: replayed turn-1 audio + new speech, finalized by f2
        ext._open_finalizes.append("f2")
        ext._fin_finals["f2"] = 0
        for txt, dur in (("pupunta ako sa", 2500), ("pupunta ako sa palengke bukas", 4000)):
            await ext.on_result({"type": "result", "id": "r1-utt-0", "text": txt, "final": False,
                                 "start_ms": 0, "duration_ms": dur, "language": "fil"})
        await ext.on_result({"type": "result", "id": "r1-utt-0", "text": "pupunta ako sa palengke bukas",
                             "final": True, "start_ms": 0, "duration_ms": 4000, "language": "fil",
                             "metadata": {"finalize_id": "f2", "endpoint": "finalize"}})
        out = [(r.final, r.text) for r in (c.args[0] for c in ext.send_asr_result.await_args_list)]
        print(out)
        assert any(f and "palengke" in t for f, t in out), f"turn 2 lost: {out}"

    run(go())

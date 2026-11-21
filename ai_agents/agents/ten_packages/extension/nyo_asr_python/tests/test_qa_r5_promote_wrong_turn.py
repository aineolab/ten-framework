"""QA r5 (fixed): a finalize_end without a final must not promote an interim of the NEXT utterance.

Real gateway flow (serve/gateway.py): finalize f1 cuts turn 1 in asr_fw; the user keeps talking, so asr_fw emits
interims for utt-1 (turn 2) while turn 1's final decode is still running. If the gateway then answers f1 with a
finalize_end and no final (its 3000 ms finalize timer, or complete_pending on engine loss), the addon used to
promote turn 2's partial as f1's final and arm late-final suppression on turn 2's id/span. Now f1 promotes only
the interim that was latest when f1 was sent (or one starting before the audio sent by then).
"""
import asyncio
from unittest.mock import MagicMock

from .helpers import make_extension, run


class _Rec:
    def __init__(self):
        self.ev = asyncio.Event()

    def is_connected(self):
        return True

    async def send_finalize(self, fid):
        pass

    async def wait_finalize_end(self, fid, timeout_ms):
        await self.ev.wait()
        return True  # finalize_end arrived (gateway finalize timeout): no final for f1


def _texts(ext):
    return [(r.final, r.text) for r in (c.args[0] for c in ext.send_asr_result.await_args_list)]


def test_r5_finalize_end_without_final_does_not_promote_the_next_turn_interim():
    async def go():
        ext = make_extension(endpointing_ms=1000, finalize_timeout_ms=3500)
        rec = _Rec()
        ext.recognition = rec
        # turn 1
        await ext.on_result({"type": "result", "id": "utt-0", "text": "magandang umaga", "final": False,
                             "start_ms": 0, "duration_ms": 1500, "language": "fil"})
        ext.finalize_id = "f1"
        fin = asyncio.create_task(ext.finalize(None))
        await asyncio.sleep(0.01)
        # the user keeps talking: asr_fw interims of utt-1 (turn 2) while turn 1's final decode runs
        await ext.on_result({"type": "result", "id": "utt-1", "text": "pupunta ako", "final": False,
                             "start_ms": 1600, "duration_ms": 1200, "language": "fil"})
        # gateway finalize timer: error finalize_timeout + finalize_end f1, no final
        rec.ev.set()
        await fin
        # turn 2 is endpointed by the server
        await ext.on_result({"type": "result", "id": "utt-1", "text": "pupunta ako sa palengke", "final": True,
                             "start_ms": 1600, "duration_ms": 2500, "language": "fil",
                             "metadata": {"finalize_id": "auto-1", "endpoint": "auto"}})
        out = _texts(ext)
        print(out)
        finals = [t for f, t in out if f]
        assert "pupunta ako sa palengke" in finals, f"turn 2's real final was dropped: {out}"
        assert "pupunta ako" not in finals, f"turn 2's partial was emitted as turn 1's final: {out}"

    run(go())


def test_r5_engine_loss_replay_keeps_turn_1_after_wrong_turn_promotion():
    """Engine-loss variant: complete_pending answers f1 without a final, the re-opened engine re-decodes the ring
    (turn 1 + turn 2, r1- ids). Turn 1's re-decode starts before the (wrong, turn-2) promoted span ends, so the
    span rule used to drop it as 'the promoted turn': turn 1 was never emitted at all."""
    async def go():
        ext = make_extension(endpointing_ms=1000, finalize_timeout_ms=3500)
        rec = _Rec()
        ext.recognition = rec
        await ext.on_result({"type": "result", "id": "utt-0", "text": "magandang umaga", "final": False,
                             "start_ms": 0, "duration_ms": 1500, "language": "fil"})
        ext.finalize_id = "f1"
        fin = asyncio.create_task(ext.finalize(None))
        await asyncio.sleep(0.01)
        await ext.on_result({"type": "result", "id": "utt-1", "text": "pupunta ako", "final": False,
                             "start_ms": 1600, "duration_ms": 1200, "language": "fil"})
        rec.ev.set()  # engine lost: complete_pending -> finalize_end f1 without a final
        await fin
        # re-opened engine session replays from the last final (0 ms): turn 1 re-decoded, endpointed
        await ext.on_result({"type": "result", "id": "r1-utt-0", "text": "magandang umaga", "final": True,
                             "start_ms": 0, "duration_ms": 1500, "language": "fil",
                             "metadata": {"finalize_id": "r1-auto-1", "endpoint": "auto"}})
        out = _texts(ext)
        print(out)
        assert (True, "magandang umaga") in out, f"turn 1 lost: {out}"

    run(go())

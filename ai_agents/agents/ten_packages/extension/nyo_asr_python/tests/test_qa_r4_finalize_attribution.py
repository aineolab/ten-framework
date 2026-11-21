"""QA r4: an endpointed auto-N final answers no asr_finalize; a finalize_end followed by a close is a normal end."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from nyo_asr_python.recognition import NYOASRRecognition

from .helpers import make_extension, run


def test_r4_auto_final_is_not_credited_to_a_client_finalize():
    async def go():
        ext = make_extension()
        rec = MagicMock()
        rec.is_connected.return_value = True
        rec.send_finalize = AsyncMock()

        async def wait(fid, timeout_ms):
            # utterance A was cut by server endpointing while f1 was in flight; then B's interim; then f1 ends
            # without a final of its own (e.g. the gateway's finalize_timeout)
            await ext.on_result({"id": "utt-4", "text": "kumusta", "final": True, "start_ms": 0, "duration_ms": 900,
                                 "metadata": {"finalize_id": "auto-3", "endpoint": "auto"}})
            await ext.on_result({"id": "utt-5", "text": "salamat po", "final": False, "start_ms": 1500,
                                 "duration_ms": 700})
            return True

        rec.wait_finalize_end = wait
        ext.recognition = rec
        ext.audio_timeline.add_user_audio(2200)  # utterances A and B were sent before f1
        ext.finalize_id = "f1"
        await ext.finalize(None)
        finals = [c.args[0].text for c in ext.send_asr_result.await_args_list if c.args[0].final]
        assert finals == ["kumusta", "salamat po"], finals
        ext.send_asr_finalize_end.assert_awaited_once()

    run(go())


def test_r4_finalize_end_then_close_is_a_normal_end():
    async def go():
        rec = NYOASRRecognition("k", MagicMock(), MagicMock(), {}, AsyncMock())
        rec.websocket = MagicMock()
        rec.websocket.send = AsyncMock()
        connected = [True]
        rec.is_connected = lambda: connected[0]
        await rec.send_finalize("f1")
        w = asyncio.create_task(rec.wait_finalize_end("f1", 3000))
        await asyncio.sleep(0)
        await rec._dispatch({"type": "finalize_end", "id": "f1"})
        connected[0] = False                     # the server closed right after finalize_end
        assert await w is True
        # a close without finalize_end still reads as "no finalize_end"
        connected[0] = True
        await rec.send_finalize("f2")
        w2 = asyncio.create_task(rec.wait_finalize_end("f2", 3000))
        await asyncio.sleep(0)
        connected[0] = False
        await rec.close()
        assert await w2 is False

    run(go())


def test_r4_handshake_404_maps_to_model_not_found():
    async def go():
        from nyo_asr_python.recognition import NRSPConnectError
        ext = make_extension(model="nyo/custom")

        async def inconclusive():
            return "inconclusive", "no_asr_models", "GET /v1/models lists no ASR model of this engine"
        ext._models_verdict = inconclusive
        fatal, code, _ = await ext._classify_connect_failure(NRSPConnectError("HTTP 404", 404))
        assert (fatal, code) == (False, "model_not_found")
        await ext._report_connect_failure(NRSPConnectError("HTTP 404", 404))
        assert ext.send_asr_error.await_args.args[1].code == "model_not_found"

    run(go())

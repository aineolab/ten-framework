"""QA round 6 (fixed): a request the extension already ended (vendor error mid-stream or completed), then a tts_flush (barge-in), then a
late LLM chunk of that same request id, then the next turn. The flush clears `_finished`; the late chunk must
still release the base input loop (finish_request), or the next request stays buffered until another flush. The late
chunk itself is not synthesised. Uses the real base on_data / _process_input_queue."""
import asyncio
import json

from fake_nrsp_server import FakeNRSPServer
from ten_runtime import Data

from .helpers import run, text_input, wait_for
from .test_qa_r3_late_chunks import _real_base_ext, _sent


def _flush_data():
    d = Data.create("tts_flush")
    d.set_property_from_json(None, json.dumps({"flush_id": "f1"}))
    return d


def _scenario(first_text, wait_first_end):
    async def go():
        async with FakeNRSPServer() as srv:
            ext, env = _real_base_ext(srv.url, "nyo-test-key+ttserr")
            loop = asyncio.create_task(ext._process_input_queue(env))
            try:
                from ten_ai_base.tts2 import RequestState

                async def put(t):
                    if t.request_id not in ext.request_states:
                        ext._cleanup_completed_states()
                        ext._transition_state(t.request_id, RequestState.QUEUED, "request received")
                    await ext.input_queue.put(t)

                await put(text_input("a", first_text, end=not wait_first_end))
                if wait_first_end:  # round 8: a failed segment does not end "a"; the barge-in below does
                    await wait_for(lambda: len(_sent(env, "error")) == 1, timeout=8, what="a's segment failed")
                else:
                    await wait_for(lambda: len(_sent(env, "tts_audio_end")) == 1, timeout=8, what="a ended")
                await ext.on_data(env, _flush_data())        # barge-in after "a" already ended (or failed)
                await put(text_input("a", "mundo ", end=True))  # aborted LLM's in-flight chunk of "a"
                await asyncio.sleep(0.2)
                await put(text_input("b", "kumusta mundo", end=True))  # next turn, no further flush
                try:
                    await wait_for(lambda: len(_sent(env, "tts_audio_end")) == 2, timeout=5, what="b ended")
                except AssertionError:
                    raise AssertionError(
                        f"request b never processed: base _processing_request_id={ext._processing_request_id!r}, "
                        f"buffered={list(ext._pending_messages)}")
                ends = [json.loads(j) for j in _sent(env, "tts_audio_end")]
                assert [e["request_id"] for e in ends] == ["a", "b"], ends
                assert ends[1]["reason"] == 1, ends
                # the late chunk of "a" never reached the gateway
                assert [t["request_id"] for t in srv.received("text")].count("a") == 1
            finally:
                ext._stopped = True
                loop.cancel()
                if ext.client:
                    await ext.client.stop()

    run(go())


def test_qa_r6_late_chunk_after_flush_of_errored_request_releases_base():
    _scenario("ERROR_TRIGGER kumusta ", wait_first_end=True)


def test_qa_r6_late_chunk_after_flush_of_completed_request_releases_base():
    _scenario("kumusta po", wait_first_end=False)

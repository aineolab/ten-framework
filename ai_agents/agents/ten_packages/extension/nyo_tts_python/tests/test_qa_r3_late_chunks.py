"""QA round 3: a request the extension ends early (vendor error, admission_rejected, connection loss) while the
LLM is still streaming its text. The base input loop then sees more chunks of that request: it re-selects the
finished request as `_processing_request_id` and calls request_tts, which ignores the chunk without calling
finish_request. `_processing_request_id` stays pinned, so the NEXT request is buffered by the base loop forever
(until some flush). Fixed: a chunk of a request the extension already ended releases the base loop
(finish_request). Uses the REAL base-class _process_input_queue / finish_request (no stubs)."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from fake_nrsp_server import FakeNRSPServer

from .helpers import make_config, run, text_input, wait_for
from nyo_tts_python.extension import NYOTTSExtension


def _real_base_ext(url, key):
    ext = NYOTTSExtension("nyo_tts_python")
    env = MagicMock()
    env.send_data = AsyncMock()
    env.send_audio_frame = AsyncMock()
    ext.ten_env = env
    ext.config = make_config(url, api_key=key)
    ext.metrics_connect_delay = AsyncMock()
    return ext, env


def _sent(env, name):
    out = []
    for c in env.send_data.call_args_list:
        d = c.args[0]
        if d.get_name() == name:
            js, _ = d.get_property_to_json("")
            out.append(js)
    return out


def test_qa_r3_next_request_starves_after_early_end_and_late_chunks():
    async def go():
        async with FakeNRSPServer() as srv:
            ext, env = _real_base_ext(srv.url, "nyo-test-key+ttserr")
            loop = asyncio.create_task(ext._process_input_queue(env))
            try:
                async def put(t):  # what on_data does, minus the Data parsing
                    if t.request_id not in ext.request_states:
                        ext._cleanup_completed_states()
                        from ten_ai_base.tts2 import RequestState
                        ext._transition_state(t.request_id, RequestState.QUEUED, "request received")
                    await ext.input_queue.put(t)

                await put(text_input("a", "ERROR_TRIGGER kumusta "))  # gateway fails a segment of "a" mid-stream
                await wait_for(lambda: len(_sent(env, "error")) == 1, timeout=5, what="a's segment failed")
                await put(text_input("a", "mundo ", end=True))  # round 8: "a" goes on with a new segment
                await asyncio.sleep(0.2)
                await put(text_input("b", "kumusta mundo", end=True))   # next turn
                try:
                    await wait_for(lambda: len(_sent(env, "tts_audio_end")) == 2, timeout=5, what="b ended")
                except AssertionError:
                    raise AssertionError(
                        f"request b never processed: base _processing_request_id={ext._processing_request_id!r}, "
                        f"buffered={list(ext._pending_messages)}")
                ends = [json.loads(j) for j in _sent(env, "tts_audio_end")]
                assert [(e["request_id"], e["reason"]) for e in ends] == [("a", 1), ("b", 1)], ends
            finally:
                ext._stopped = True
                loop.cancel()
                if ext.client:
                    await ext.client.stop()

    run(go())


def test_qa_r3_duplicate_chunk_after_end_does_not_release_a_playing_request():
    """A chunk of a request that is still playing (text_input_end sent) is ignored, but must NOT finish it in
    the base early: the next request would then interrupt it."""
    async def go():
        async with FakeNRSPServer() as srv:
            ext, env = _real_base_ext(srv.url, "nyo-test-key+rt")
            loop = asyncio.create_task(ext._process_input_queue(env))
            try:
                from ten_ai_base.tts2 import RequestState

                async def put(t):
                    if t.request_id not in ext.request_states:
                        ext._cleanup_completed_states()
                        ext._transition_state(t.request_id, RequestState.QUEUED, "request received")
                    await ext.input_queue.put(t)

                await put(text_input("a", "kumusta mundo ito ay isang pagsubok", end=True))
                await wait_for(lambda: _sent(env, "tts_audio_start"), timeout=5, what="a playing")
                await put(text_input("a", "duplicate"))
                await put(text_input("b", "kumusta", end=True))
                await wait_for(lambda: len(_sent(env, "tts_audio_end")) == 2, timeout=15, what="a and b ended")
                ends = [json.loads(j) for j in _sent(env, "tts_audio_end")]
                assert [(e["request_id"], e["reason"]) for e in ends] == [("a", 1), ("b", 1)], ends
            finally:
                ext._stopped = True
                loop.cancel()
                if ext.client:
                    await ext.client.stop()

    run(go())

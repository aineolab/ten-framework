"""QA round 4 (fixed): a non-reconnectable fatal server error handled on the socket's reader task.

_error_once() (FATAL) spawned client.stop(), and stop() cancelled the reader task. If the reader yielded anywhere
later in _finalize_request() (always with `dump` on: the PCMWriter flush awaits a task), it was cancelled there:
finish_request() was never called (the base loop stayed pinned to the request) and the one FATAL tts_error was
never sent. Now the socket is closed without cancelling the reader that reports the FATAL.
Test 1: dump on, runtime-like fire-and-forget send_data. Test 2: a send_data that yields. Uses the REAL base.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from fake_nrsp_server import FakeNRSPServer

from .helpers import make_config, run, text_input
from nyo_tts_python.extension import NYOTTSExtension


def _ext(url, key, yield_send=True, dump_dir=None):
    ext = NYOTTSExtension("nyo_tts_python")
    env = MagicMock()
    sent = []

    async def send_data(d):
        if yield_send:
            await asyncio.sleep(0.002)
        js, _ = d.get_property_to_json("")
        sent.append((d.get_name(), json.loads(js) if js else {}))

    env.send_data = send_data
    env.send_audio_frame = AsyncMock()
    ext.ten_env = env
    ext.config = make_config(url, api_key=key)
    ext.metrics_connect_delay = AsyncMock()
    if dump_dir:
        ext.config.dump, ext.config.dump_path = True, dump_dir
    return ext, env, sent


def _case(**kw):
    async def go():
        async with FakeNRSPServer() as srv:
            ext, env, sent = _ext(srv.url, "nyo-test-key+fatal", **kw)   # fatal engine_error + close 1011 on first text
            loop = asyncio.create_task(ext._process_input_queue(env))
            try:
                from ten_ai_base.tts2 import RequestState
                ext._transition_state("a", RequestState.QUEUED, "request received")
                await ext.input_queue.put(text_input("a", "kumusta mundo", end=True))
                await asyncio.sleep(2.0)
                names = [n for n, _ in sent]
                errors = [p for n, p in sent if n == "error"]
                ends = [p for n, p in sent if n == "tts_audio_end"]
                assert ext._fatal_sent, names
                assert ends and ends[0]["request_id"] == "a", f"no tts_audio_end for 'a'; sent: {names}"
                assert errors and errors[0]["code"] == -1000, f"FATAL error never sent; sent: {names}"
                assert ext._processing_request_id is None, "base loop still pinned to 'a'"
            finally:
                ext._stopped = True
                loop.cancel()
                if ext.client:
                    await ext.client.stop()

    run(go())


def test_qa_r4_fatal_with_dump_reports_fatal_and_releases_loop(tmp_path):
    _case(yield_send=False, dump_dir=str(tmp_path))


def test_qa_r4_fatal_with_yielding_send_still_ends_request():
    _case(yield_send=True)

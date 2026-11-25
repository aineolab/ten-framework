"""QA r3: real serve/ gateway + real engines on 127.0.0.1:9150 (read-only use). Skipped when not running."""
import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from nyo_asr_python.reconnect_manager import ReconnectManager

from .helpers import make_extension, run
from .test_qa_r2_gateway import _Frame, _up, _wait

KEY = os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key")
AUDIO = os.environ.get("NYO_QA_REAL_AUDIO_DIR", "/nonexistent/nyo-qa-audio")
pytestmark = pytest.mark.skipif(not (_up(9150) and os.path.exists(KEY)), reason="real stack not running")


def _ext(model="halohalo/asr-fil"):
    ext = make_extension(base_url="ws://127.0.0.1:9150", allow_insecure=True, model=model, language="fil",
                         api_key=open(KEY).read().strip(), finalize_timeout_ms=8000)
    ext._emit_connection_transition = AsyncMock()
    ext.reconnect_manager = ReconnectManager(window_ms=60_000)
    return ext


async def _feed(ext, pcm, pace=0.005):
    for i in range(0, len(pcm), 640):
        await ext._handle_audio_frame(ext.ten_env, _Frame(pcm[i:i + 640]))
        await asyncio.sleep(pace)


@pytest.mark.parametrize("model", ["halohalo/asr-fil", "fw/whisper-base"])
def test_r3_real_reconnect_keeps_text_and_timeline(model):
    s1 = open(f"{AUDIO}/s1_16k.pcm", "rb").read()
    s2 = open(f"{AUDIO}/s2_16k.pcm", "rb").read()
    sil = b"\x00\x00" * 16000  # 1 s

    async def go():
        ext = _ext(model)
        try:
            await ext.start_connection()
            assert ext.is_connected()
            await _feed(ext, s1 + sil)
            ext.finalize_id = "f1"
            await ext.finalize(None)
            first = ext.recognition
            # the socket dies (not our close): reconnect + Keep replay
            await first.websocket.close()
            await _feed(ext, sil[:3200])
            await _feed(ext, s2 + sil)
            assert await _wait(ext.is_connected, 15)
            ext.finalize_id = "f2"
            await ext.finalize(None)
            finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
            for r in [c.args[0] for c in ext.send_asr_result.await_args_list]:
                print(model, r.final, r.start_ms, r.duration_ms, repr(r.text), r.metadata.get("asr_info"))
            print("errors", [(c.args[0].code, c.args[1].code if c.args[1] else None) for c in ext.send_asr_error.await_args_list])
            texts = [r for r in finals if r.text.strip()]
            assert len(texts) >= 2
            assert texts[-1].start_ms > len(s1) // 32, "turn 2 timestamp is after turn 1 audio"
        finally:
            await ext.stop_connection()

    run(go())

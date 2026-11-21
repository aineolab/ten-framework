"""QA round 6, real serve/ gateway with SAPIN_IDLE_TIMEOUT_S=2 on 127.0.0.1:9145 (started by the reviewer; skipped
otherwise): requests that start around the gateway's idle close must all complete (reconnect or resend), with no
error reported."""
import asyncio
import os
import socket

import pytest
from ten_ai_base.message import TTSAudioEndReason

from .helpers import make_extension, run, text_input, wait_for

GW = "ws://127.0.0.1:9145"
KEY_FILE = os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key")


def _up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", 9145), 0.3).close()
        return os.path.exists(KEY_FILE)
    except OSError:
        return False


@pytest.mark.skipif(not _up(), reason="reviewer gateway on 9145 not running")
def test_qa_r6_real_requests_around_idle_close():
    async def go():
        key = open(KEY_FILE).read().strip()
        ext, rec = make_extension(GW, api_key=key, model="halohalo/tts-fil")
        try:
            await ext._ensure_connected()
            for i, delay in enumerate([2.0, 2.2, 2.4, 2.6, 3.5]):
                await asyncio.sleep(delay)
                await ext.request_tts(text_input(f"r{i}", "Salamat po.", end=True))
                await wait_for(lambda: len(rec.of("audio_end")) == i + 1, timeout=40, what=f"r{i}")
                end = rec.of("audio_end")[-1]
                assert end["reason"] == TTSAudioEndReason.REQUEST_END and end["duration_ms"] > 0, (
                    end, rec.of("error"), rec.of("finish_error"))
            assert not rec.of("error") and not rec.of("finish_error")
            print("\nconnects:", len(rec.of("connected")), "disconnects:",
                  [d.get("message") for d in rec.of("disconnected")])
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

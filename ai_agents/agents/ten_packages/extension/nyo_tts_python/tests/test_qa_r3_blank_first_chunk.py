"""QA round 3: a whitespace-only first chunk (e.g. an LLM delta "\\n\\n") is sent to the gateway, which opens request a
on it, but sent_text stays False, so the closing empty text_input_end is answered locally (REQUEST_END) and the
gateway never gets end=true. The gateway's per-socket worker then sits on a until tts_stall_s, so the next request b on
that socket is delayed by the stall, and a's stall error surfaces as a NON_FATAL tts_error for a finished request.
Fixed: blank text before the first real text is not sent.
Needs the real serve/ gateway with mock engines at NYO_QA_GW3 (default ws://127.0.0.1:9146, SAPIN_TTS_STALL_S=4)."""
import os
import socket
import time
from urllib.parse import urlparse

import pytest

from .helpers import make_extension, run, text_input, wait_for

GW = os.environ.get("NYO_QA_GW3", "ws://127.0.0.1:9146")


def _gw_up() -> bool:
    u = urlparse(GW)
    try:
        socket.create_connection((u.hostname, u.port), 0.3).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _gw_up(), reason="real gateway not running")
def test_qa_r3_blank_first_chunk_leaves_request_open_on_gateway():
    async def go():
        ext, rec = make_extension(GW, model="halohalo/tts-fil")
        try:
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "\n\n"))
            await ext.request_tts(text_input("a", "", end=True))
            assert rec.of("audio_end"), "a ended locally"
            t0 = time.monotonic()
            await ext.request_tts(text_input("b", "Mabuhay.", end=True))
            await wait_for(lambda: len(rec.of("audio_end")) == 2, timeout=15, what="b ended")
            b_ms = (time.monotonic() - t0) * 1000
            errs = [e["error"].message for e in rec.of("error")]
            assert b_ms < 2000 and not errs, (
                f"b took {b_ms:.0f} ms (queued behind a, which the gateway never saw end); errors: {errs}")
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

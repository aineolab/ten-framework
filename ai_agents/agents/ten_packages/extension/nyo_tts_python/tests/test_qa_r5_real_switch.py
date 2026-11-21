"""QA round 5, real gateway (NYO_QA_REAL, default ws://127.0.0.1:9150, key in NYO_QA_REAL_KEY_FILE): a deferred
update_configs (fil -> hil) during a request, then a flush, then the next request: it must run on hil at 16 kHz,
its first audio must not wait for a long close handshake, and nothing of the flushed request may leak."""
import asyncio
import os
import socket
import time
from urllib.parse import urlparse

import pytest
from ten_ai_base.message import TTSAudioEndReason

from .helpers import make_extension, run, text_input, wait_for

GW = os.environ.get("NYO_QA_REAL", "ws://127.0.0.1:9150")
KEY_FILE = os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key")


def _up() -> bool:
    u = urlparse(GW)
    try:
        socket.create_connection((u.hostname, u.port), 0.3).close()
        return os.path.exists(KEY_FILE)
    except OSError:
        return False


@pytest.mark.skipif(not _up(), reason="real gateway not running")
def test_qa_r5_real_deferred_switch_then_flush_then_next():
    async def go():
        key = open(KEY_FILE).read().strip()
        ext, rec = make_extension(GW, api_key=key, model="halohalo/tts-fil")
        try:
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "Magandang umaga sa inyong lahat, kumusta kayo ngayong araw?", end=True))
            await wait_for(lambda: rec.of("audio"), timeout=30, what="audio of a")
            await ext.update_configs({"params": {"model": "halohalo/tts-hil"}})
            assert ext._pending_config is not None
            await ext.cancel_tts()
            n_a = len([e for e in rec.events if e[0] == "audio"])
            ends = rec.of("audio_end")
            assert ends and ends[-1]["reason"] == TTSAudioEndReason.INTERRUPTED
            t0 = time.monotonic()
            await ext.request_tts(text_input("b", "Maayong aga.", end=True))
            sent_after = time.monotonic() - t0
            await wait_for(lambda: len(rec.of("audio_end")) >= 2, timeout=40, what="audio_end of b")
            idx = rec.names().index("audio_end")
            leaked = [kw for n, kw in rec.events[idx + 1:] if n == "audio" and kw["rate"] != 16000]
            b_end = rec.of("audio_end")[-1]
            assert b_end["request_id"] == "b" and b_end["reason"] == TTSAudioEndReason.REQUEST_END, b_end
            assert not leaked, f"audio at a non-16k rate after a's end: {leaked[:3]} (a had {n_a} frames)"
            assert ext._sample_rate == 16000
            assert sent_after < 3, f"b's text took {sent_after:.2f} s to reach the gateway"
            print(f"\n[r5] b sent after {sent_after*1000:.0f} ms; b audio {b_end['duration_ms']} ms")
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

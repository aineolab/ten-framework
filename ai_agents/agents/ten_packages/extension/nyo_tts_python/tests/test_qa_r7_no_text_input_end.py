"""QA round 7 (fixed), real serve/ gateway + real engine adapter on 127.0.0.1:9150 (skipped otherwise).

voice-assistant-with-turn-detection sends every LLM sentence as tts_text_input with text_input_end=false and never
sends text_input_end=true for an LLM reply (main_python _on_llm_response; only the greeting ends). parse_sentences()
cuts right after the punctuation, so a sentence carries no trailing whitespace; the next one starts with " ".

The NYO engines split sentences on punctuation FOLLOWED BY whitespace (or at `end`), and SpeechT5 waits for `end`.
Before the fix the last sentence of every reply (and a one-sentence reply entirely) was never synthesised. Now the
extension sends a sentence-boundary hint and ends the NRSP request after segment_idle_ms without a chunk; the next
chunk opens a new NRSP request, and all audio stays in the one TEN request (one tts_audio_start, no tts_audio_end
until text_input_end or a flush).
"""
import asyncio
import os
import socket

import pytest
from ten_ai_base.message import TTSAudioEndReason

from .helpers import make_extension, run, text_input, wait_for

GW = "ws://127.0.0.1:9150"
KEY_FILE = os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key")


def _up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", 9150), 0.3).close()
        return os.path.exists(KEY_FILE)
    except OSError:
        return False


def _audio_ms(ext, rec) -> int:
    return sum(a["n"] for a in rec.of("audio")) * 1000 // (2 * ext.synthesize_audio_sample_rate())


def _scenario(model):
    async def go():
        key = open(KEY_FILE).read().strip()
        ext, rec = make_extension(GW, api_key=key, model=model)
        try:
            await ext._ensure_connected()
            # what parse_sentences() yields for "Salamat po. Paalam na." with text_input_end=false
            await ext.request_tts(text_input("tts-request-1", "Salamat po.", end=False))
            await wait_for(lambda: _audio_ms(ext, rec) > 0, timeout=20, what="sentence 1 spoken without an end")
            await wait_for(lambda: "tts-request-1" in ext._req.seg_done, timeout=20, what="segment 1 done")
            first = _audio_ms(ext, rec)
            await ext.request_tts(text_input("tts-request-1", " Paalam na.", end=False))
            await wait_for(lambda: "tts-request-1~2" in ext._req.seg_done, timeout=20, what="segment 2 done")
            total = _audio_ms(ext, rec)
            print(f"\n{model}: audio after sentence 1: {first} ms; after sentence 2: {total} ms")
            assert total > first > 0
            names = rec.names()
            assert names.count("audio_start") == 1 and "audio_end" not in names, names
            assert not rec.of("error") and not rec.of("finish_error")
            ts = [a["ts"] for a in rec.of("audio")]
            assert ts == sorted(ts) and ts[0] == 0, ts  # one timeline across both segments
            # the LLM is slow: no watchdog / timeout while the TEN request waits for its next chunk
            assert ext._req.open and not ext._outstanding(ext._req)
            await ext.request_tts(text_input("tts-request-1", "", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            end = rec.of("audio_end")[0]
            assert end["reason"] == TTSAudioEndReason.REQUEST_END and abs(end["duration_ms"] - total) <= 1, end
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())


@pytest.mark.skipif(not _up(), reason="real stack on 9150 not running")
def test_qa_r7_turn_detection_reply_without_text_input_end_is_spoken():
    _scenario("halohalo/tts-fil")


@pytest.mark.skipif(not _up(), reason="real stack on 9150 not running")
def test_qa_r7_turn_detection_reply_without_text_input_end_is_spoken_hil():
    _scenario("halohalo/tts-hil")

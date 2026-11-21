"""QA round 8 (fixed): the sentence-boundary hint must not split decimals, prices or domains.

main_python parse_sentences() (voice-assistant and voice-assistant-with-turn-detection) cuts after EVERY '.', ',', '?',
'!' -- also the '.' inside "3.50" or "nyolab.ai" -- and leaves the whitespace at the start of the NEXT chunk. Round 7
appended a space to every sentence-final chunk at once, so the engine received "₱3. 50" and the Orpheus front end
(modules/tts/frontend.py split_sentences, regex copied below) cut a sentence there. Now chunks go out verbatim; a
sentence-final chunk (not after a digit) that no next chunk follows within boundary_hint_ms gets one " " increment.
"""
import asyncio
import os
import re
import socket

import pytest

from .helpers import harness, make_extension, run, text_input, wait_for

# modules/tts/frontend.py split_sentences (nyo-serve), the halohalo/tts-fil (Orpheus) sentence splitter
_ORPHEUS_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def _parse_sentences(fragment, content):
    """main_python helper.parse_sentences (verbatim logic)."""
    out, cur = [], fragment
    for ch in content:
        cur += ch
        if ch in [",", "，", ".", "。", "?", "？", "!", "！"]:
            if any(c.isalnum() for c in cur):
                out.append(cur)
            cur = ""
    return out, cur


def _orpheus_sentences(text):
    return [p.strip() for p in _ORPHEUS_SPLIT.split(text) if p.strip()]


def _wire(srv):
    return [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]


def test_qa_r8_decimal_price_and_domain_stay_whole_on_the_wire():
    reply = "Ang presyo ay ₱3.50 lang po. Bisitahin ang nyolab.ai para sa iba."
    chunks, rest = _parse_sentences("", reply)
    assert chunks[0] == "Ang presyo ay ₱3." and chunks[1] == "50 lang po."
    original = "".join(chunks) + rest
    assert _orpheus_sentences(original) == ["Ang presyo ay ₱3.50 lang po.", "Bisitahin ang nyolab.ai para sa iba."]

    async def go():
        async with harness() as (srv, ext, rec):  # defaults: boundary_hint_ms 250, segment_idle_ms 1000
            await ext._ensure_connected()
            for c in chunks:  # the LLM streams: the next chunk follows within a few ms
                await ext.request_tts(text_input("r", c, end=False))
                await asyncio.sleep(0.01)
            await ext.request_tts(text_input("r", rest, end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            wire = "".join(t["text"] for t in srv.received("text"))
            assert wire == original and _orpheus_sentences(wire) == _orpheus_sentences(original), wire

    run(go())


def test_qa_r8_last_sentence_gets_its_hint_after_boundary_hint_ms():
    async def go():
        async with harness(segment_idle_ms=0, boundary_hint_ms=100) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Salamat po.", end=False))
            await asyncio.sleep(0.05)
            assert _wire(srv) == [("r", "Salamat po.", False)]  # not yet
            await wait_for(lambda: len(_wire(srv)) == 2, timeout=2, what="hint")
            assert _wire(srv)[1] == ("r", " ", False)
            await asyncio.sleep(0.3)
            assert len(_wire(srv)) == 2  # once
            assert ext._req.open and not rec.of("audio_end")

    run(go())


def test_qa_r8_no_hint_after_a_digit_and_a_next_chunk_cancels_it():
    async def go():
        async with harness(segment_idle_ms=0, boundary_hint_ms=100) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Ang presyo ay ₱3.", end=False))
            await asyncio.sleep(0.3)  # a slow LLM after "3.": still no hint (the next chunk is "50 ...")
            await ext.request_tts(text_input("r", "50 lang po.", end=False))
            await asyncio.sleep(0.05)
            await ext.request_tts(text_input("r", " Salamat.", end=False))  # in time: cancels the hint of "po."
            await asyncio.sleep(0.05)
            await ext.request_tts(text_input("r", "", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            assert [w[1] for w in _wire(srv)] == ["Ang presyo ay ₱3.", "50 lang po.", " Salamat.", ""]

    run(go())


# ------------------------------------------------------------------- real engine (Orpheus, halohalo/tts-fil)
GW = "ws://127.0.0.1:9150"
KEY_FILE = os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key")


def _up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", 9150), 0.3).close()
        return os.path.exists(KEY_FILE)
    except OSError:
        return False


@pytest.mark.skipif(not _up(), reason="real stack on 9150 not running")
def test_qa_r8_real_last_sentence_is_spoken_by_the_hint_alone():
    """No text_input_end and no idle end (segment_idle_ms 0): only the delayed hint makes Orpheus speak it."""
    async def go():
        key = open(KEY_FILE).read().strip()
        ext, rec = make_extension(GW, api_key=key, model="halohalo/tts-fil", segment_idle_ms=0)
        try:
            await ext._ensure_connected()
            await ext.request_tts(text_input("tts-request-1", "Salamat po sa inyong tulong.", end=False))
            await wait_for(lambda: rec.of("audio"), timeout=20, what="speech without an end")
            assert rec.names().count("audio_start") == 1 and not rec.of("audio_end") and not rec.of("error")
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

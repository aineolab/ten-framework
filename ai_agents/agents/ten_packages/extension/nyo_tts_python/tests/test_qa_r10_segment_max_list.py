"""QA round 10 (fixed): segment_max_ms hard cut.

A numbered list reaches TTS (parse_sentences) as "1.", " Bigas,\n2.", " Itlog,\n3.", ...: no chunk ends a sentence
"not after a digit", so the sentence rule of segment_max_ms never ends the segment, and on SpeechT5 (hil), which
speaks only at the end of its request, a list streamed for longer than the gateway's tts_stall_s (10 s) was aborted
and lost. Now a segment older than segment_max_ms + HARD_CUT_EXTRA_MS ends after the next chunk that ends with a
clause mark, a list number at a line start or a newline; never inside "₱3.50".
"""
import asyncio

from nyo_tts_python import extension
from nyo_tts_python.extension import hard_cut

from .helpers import harness, run, text_input


def test_qa_r10_hard_cut_marks():
    for yes in ("Bigas,", "Una;", "Ito ang listahan:", " Itlog,\n2.", "1.", "Ito ang listahan:\n1.", "linya\n",
                'sabi niya, "oo,"', "好，"):
        assert hard_cut(yes), yes
    for no in ("₱3.", "ay 3.", "Salamat po", "nyolab.", "Salamat po.", ""):
        assert not hard_cut(no), no


def test_qa_r10_segment_max_hard_cut_ends_a_numbered_list(monkeypatch):
    monkeypatch.setattr(extension, "HARD_CUT_EXTRA_MS", 100)

    async def go():
        async with harness(segment_idle_ms=1000, segment_max_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            chunks = ["Ito ang listahan:\n1."] + [f" Item numero {i} para sa iyo,\n{i + 1}." for i in range(1, 16)]
            for c in chunks:
                await ext.request_tts(text_input("r", c))
                await asyncio.sleep(0.05)
            wire = [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]
            ends = [i for i, w in enumerate(wire) if w[2]]
            assert ends, "segment_max_ms never ended the segment"
            cut = wire[ends[0] - 1]
            assert cut[0] == "r" and hard_cut(cut[1]), cut  # cut right after a list-number / clause chunk
            assert any(w[0] == "r~2" for w in wire), wire  # the next chunk opened a new segment
            # every TEN character reached the wire once, in order (no hint: boundary_hint_ms=0)
            assert "".join(w[1] for w in wire) == "".join(chunks)

    run(go())


def test_qa_r10_hard_cut_waits_for_the_extra_time(monkeypatch):
    async def go():
        async with harness(segment_idle_ms=1000, segment_max_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            for c in ["Una,", " dalawa,", " tatlo,"]:
                await ext.request_tts(text_input("r", c))
                await asyncio.sleep(0.1)
            # past segment_max_ms but not past + HARD_CUT_EXTRA_MS (2000 ms): commas do not end the segment yet
            assert not [t for t in srv.received("text") if t["end"]]

    run(go())

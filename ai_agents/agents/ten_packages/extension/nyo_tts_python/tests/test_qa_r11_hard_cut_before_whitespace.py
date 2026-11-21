"""QA round 11: segment_max_ms hard cut BEFORE a chunk that begins with whitespace.

The hard cut never cuts after a digit ("₱1," + "250" stays whole), so a long price-only comma chain
("Bigas ₱52," " itlog ₱8," " gatas ₱95," ...) had no cut point and SpeechT5 (hil), which speaks only at the end of
its request, could hit the gateway's 10 s stall. Now, once the hard-cut age is reached, a next chunk that begins with
whitespace (main_python parse_sentences keeps it at the start of the NEXT chunk) ends the current segment before it
is sent and opens the next segment. "₱1," + "250." (no space) is never cut. The wire text is the TEN text."""
import asyncio

from nyo_tts_python import extension

from .helpers import harness, run, text_input, wait_for

ITEMS = ["Bigas ₱52,", " itlog ₱8,", " gatas ₱95,", " asukal ₱70,", " kape ₱120,", " tinapay ₱45,",
         " sardinas ₱25,", " noodles ₱15,", " mantika ₱90,", " asin ₱10,", " suka ₱30,", " toyo ₱28,"]


def _wire(srv):
    return [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]


def test_qa_r11_price_only_comma_chain_is_cut_before_a_whitespace_chunk(monkeypatch):
    monkeypatch.setattr(extension, "HARD_CUT_EXTRA_MS", 100)

    async def go():
        async with harness(segment_idle_ms=1000, segment_max_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            for c in ITEMS:
                await ext.request_tts(text_input("r", c))
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.05)
            wire = _wire(srv)
            ends = [i for i, w in enumerate(wire) if w[2]]
            assert ends, f"the price chain was never cut: {wire}"
            first = ends[0]
            assert wire[first] == ("r", "", True), wire
            # cut after a "₱N," chunk, before the chunk that begins with whitespace, which opens segment ~2
            assert wire[first - 1][1].endswith(",") and wire[first - 1][1][-2].isdigit(), wire
            assert wire[first + 1][0] == "r~2" and wire[first + 1][1][:1].isspace(), wire
            # every TEN character reached the wire once, in order, nothing added (no hint: boundary_hint_ms=0)
            assert "".join(w[1] for w in wire) == "".join(ITEMS)
            assert all(w[1] in ITEMS or w[1] == "" for w in wire), wire

    run(go())


def test_qa_r11_no_cut_before_a_chunk_without_leading_whitespace(monkeypatch):
    monkeypatch.setattr(extension, "HARD_CUT_EXTRA_MS", 100)

    async def go():
        async with harness(segment_idle_ms=1000, segment_max_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            # "₱1," + "250," + ... + "500." (one long number, parse_sentences cut it at every mark): never cut
            chunks = ["Ang kabuuan ay ₱1,"] + ["250,"] * 8 + ["500"]
            for c in chunks:
                await ext.request_tts(text_input("r", c))
                await asyncio.sleep(0.05)
            assert not [w for w in _wire(srv) if w[2]], _wire(srv)
            assert "".join(w[1] for w in _wire(srv)) == "".join(chunks)

    run(go())


def test_qa_r11_no_cut_before_the_hard_cut_age():
    async def go():
        # default HARD_CUT_EXTRA_MS (2000 ms): past segment_max_ms only, a whitespace chunk does not cut yet
        async with harness(segment_idle_ms=1000, segment_max_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            for c in ITEMS[:5]:
                await ext.request_tts(text_input("r", c))
                await asyncio.sleep(0.05)
            assert not [w for w in _wire(srv) if w[2]], _wire(srv)

    run(go())


def test_qa_r11_blank_chunk_is_no_cut_point(monkeypatch):
    monkeypatch.setattr(extension, "HARD_CUT_EXTRA_MS", 50)

    async def go():
        async with harness(segment_idle_ms=1000, segment_max_ms=50, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Bigas ₱52,"))
            await asyncio.sleep(0.15)
            await ext.request_tts(text_input("r", " "))  # whitespace only: goes to the open segment, no cut
            await wait_for(lambda: len(_wire(srv)) >= 2, timeout=1, what="the blank chunk on the wire")
            await asyncio.sleep(0.05)
            assert _wire(srv) == [("r", "Bigas ₱52,", False), ("r", " ", False)], _wire(srv)

    run(go())

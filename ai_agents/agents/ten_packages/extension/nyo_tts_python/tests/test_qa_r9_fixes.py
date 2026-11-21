"""QA round 9 (fixed): the real cause in vendor_info.code, no useless hint before a closing quote, and a segment
timer that fires while an idle-close resend is pending acts on the fresh socket instead of being lost."""
import asyncio

from ten_ai_base.message import TTSAudioEndReason

from nyo_tts_python.extension import ends_sentence, hint_splits
from nyo_tts_python.tts_client import NYOTTSConnectError

from .helpers import harness, make_extension, run, text_input, wait_for


def _wire(srv):
    return [(t["request_id"], t["text"], t["end"]) for t in srv.received("text")]


def test_qa_r9_fail_request_carries_the_cause_and_keeps_the_status_in_the_message():
    async def go():
        ext, rec = make_extension()
        req = await ext._begin_request(text_input("r", "Una."))
        e = NYOTTSConnectError("not connected within request_timeout_ms (30000 ms): no such model", 404,
                               code="model_not_found")
        await ext._fail_request(req, e, status=404, fatal=False)
        err = rec.of("finish_error")[0]["error"]
        assert err.vendor_info.code == "model_not_found" and "HTTP 404" in err.message, err
        assert int(err.code) == 1000
        req = await ext._begin_request(text_input("r2", "Una."))
        await ext._fail_request(req, NYOTTSConnectError("refused", None), status=None, fatal=False)
        assert rec.of("finish_error")[1]["error"].vendor_info.code == "connect_failed"

    run(go())


def test_qa_r9_hint_only_where_the_engine_splitter_cuts():
    assert hint_splits("Salamat po.") and hint_splits(" Ano?") and hint_splits("Ay!") and hint_splits("Hmm…")
    for no in ('Sabi niya, "salamat po."', "(salamat po.)", "₱3.", "好。", "Salamat po,"):
        assert not hint_splits(no), no
    assert ends_sentence('Sabi niya, "salamat po."')  # segment_max_ms still ends a segment there (end:true splits)


def test_qa_r9_no_hint_after_a_closing_quote_the_idle_end_speaks_it():
    async def go():
        async with harness(segment_idle_ms=400, boundary_hint_ms=50) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", 'Sabi niya, "salamat po."'))
            await asyncio.sleep(0.2)
            assert _wire(srv) == [("r", 'Sabi niya, "salamat po."', False)], _wire(srv)  # no " " that cannot split
            await wait_for(lambda: ("r", "", True) in _wire(srv), timeout=2, what="idle end")
            await wait_for(lambda: "r" in ext._req.seg_done, timeout=3, what="spoken")

    run(go())


def test_qa_r9_timer_firing_while_a_resend_is_pending_acts_after_the_resend():
    async def go():
        async with harness(segment_idle_ms=100, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            live = ext.client
            await ext.request_tts(text_input("r", "Una po"))  # no sentence end: only the idle end can speak it
            req = ext._req
            # the socket the text went to is closing for idle_timeout; the resend to the fresh socket is pending
            dead = type(live)(ext.config, ext.ten_env)
            dead.mark_unusable()
            req.sent_on = dead
            go_on = asyncio.Event()

            async def resend():
                await go_on.wait()
                req.sent_on = live
            req.resend_task = asyncio.ensure_future(resend())
            await asyncio.sleep(0.3)  # the idle timer fired meanwhile, on the closing socket
            assert ("r", "", True) not in _wire(srv)
            go_on.set()
            await wait_for(lambda: ("r", "", True) in _wire(srv), timeout=2, what="idle end after the resend")
            await wait_for(lambda: "r" in req.seg_done, timeout=3, what="segment spoken")
            await ext.request_tts(text_input("r", "", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=3, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END and not rec.of("error")

    run(go())

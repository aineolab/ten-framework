"""QA round 10 (fixed): a socket that dies during an idle-close resend.

_fail_request() read ``e.code`` since round 9, and a websockets ConnectionClosed has a deprecated ``code`` property
(its IntEnum close code): vendor_info.code became "CloseCode.ABNORMAL_CLOSURE" and the whole TEN request ended. Now
a lost socket is ``connection_lost`` and, in the resend as on every other lost-socket path, costs those segments only.
"""
import asyncio
import warnings

from ten_ai_base.message import TTSAudioEndReason
from websockets.exceptions import ConnectionClosedError

from .helpers import harness, make_extension, run, text_input, wait_for


def test_qa_r10_fail_request_connection_closed_is_connection_lost():
    async def go():
        ext, rec = make_extension()
        req = await ext._begin_request(text_input("r", "Una po."))
        req.wire_ids.append("r")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)  # the deprecated .code is never read
            await ext._fail_request(req, ConnectionClosedError(None, None), status=None, fatal=False)
        err = rec.of("finish_error")[0]["error"]
        assert err.vendor_info.code == "connection_lost" and int(err.code) == 1000, err

    run(go())


def test_qa_r10_resend_on_a_dying_socket_costs_one_segment():
    async def go():
        async with harness(segment_idle_ms=0, boundary_hint_ms=0) as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.request_tts(text_input("r", "Una po"))
            req = ext._req
            live = ext.client
            real_connect, real_send = ext._connect_for_request, live.send_text

            async def no_connect(r):
                return None

            async def dying_send(*a, **k):
                raise ConnectionClosedError(None, None)

            ext._connect_for_request = no_connect
            live.send_text = dying_send
            req.resending = True
            await ext._resend(req, ["r"])
            ext._connect_for_request, live.send_text = real_connect, real_send
            assert req.open and "r" in req.seg_failed and req.seg_ended, req
            errs = rec.of("error")
            assert len(errs) == 1 and errs[0]["error"].vendor_info.code == "connection_lost", errs
            assert int(errs[0]["error"].code) == 1000 and not rec.of("audio_end")
            # the TEN request goes on: its next chunk opens a new segment and it ends normally
            await ext.request_tts(text_input("r", " Dalawa po.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5, what="tts_audio_end")
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END
            assert ("r~2", " Dalawa po.", True) in [(t["request_id"], t["text"], t["end"])
                                                    for t in srv.received("text")]

    run(go())

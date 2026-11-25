"""QA round 2: once the gateway has announced idle_timeout, the socket is closing. A request that starts
before the close frame is processed is still written to that socket and fails with connection_lost,
although "idle_timeout reconnects lazily with no error" is the contract."""
import json

from ten_ai_base.message import TTSAudioEndReason

from .helpers import harness, run, text_input, wait_for


def test_qa_r2_request_right_after_idle_notice_uses_a_new_socket():
    async def go():
        async with harness() as (srv, ext, rec):
            ext._reconnect_delay = lambda attempt: 0.01
            await ext._ensure_connected()
            client = ext.client
            ws = client.websocket
            # the gateway's idle notice arrives; its 1001 close frame is still in flight
            await client._on_text(json.dumps({"type": "error", "code": "idle_timeout", "fatal": True,
                                              "vendor": {"name": "nyo", "code": "idle_timeout",
                                                         "message": "no client message for 300 s"}}))
            await ext.request_tts(text_input("r", "kumusta ", end=True))
            await ws.close(1001)  # the close frame lands (on the old socket)
            await wait_for(lambda: rec.of("audio_end"), timeout=5)
            errs = [f["error"] for f in rec.of("finish") if f["error"]]
            assert len(srv.connections) == 2 and ext.client is not client
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END and not errs, (
                f"request after idle notice: {rec.of('audio_end')[0]['reason']}, "
                f"{[(e.vendor_info.code, e.message) for e in errs]}")

    run(go())

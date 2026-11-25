"""QA round 2: update_configs with a voice the gateway does not serve used to be accepted (cmd OK), then the 400 on
the upgrade made the extension FATAL for good. Fixed: the command is rejected and the old config stays.
Needs serve/ with mock engines at NYO_QA_GW (default ws://127.0.0.1:9145)."""
import asyncio
import os
import socket
from urllib.parse import urlparse

import pytest
from ten_ai_base.message import ModuleErrorCode, TTSAudioEndReason

from .helpers import make_extension, run, text_input, wait_for

GW = os.environ.get("NYO_QA_GW", "ws://127.0.0.1:9145")
FATAL = int(ModuleErrorCode.FATAL_ERROR.value)


def _gw_up() -> bool:
    u = urlparse(GW)
    try:
        socket.create_connection((u.hostname, u.port), 0.3).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _gw_up(), reason="real gateway not running")
def test_qa_r2_bad_voice_update_must_not_kill_the_extension():
    async def go():
        ext, rec = make_extension(GW, model="halohalo/tts-fil")
        ext._reconnect_delay = lambda attempt: 0.05
        try:
            await ext._ensure_connected()
            try:
                await ext.update_configs({"params": {"voice": "no-such-voice"}})
                # the mock gateway lists no catalogue: accepted; the 400 must then not be FATAL by count
                await asyncio.sleep(2)
                assert not ext._fatal
                await ext.update_configs({"params": {"voice": ""}})            # operator corrects it
            except ValueError as e:
                assert "rejected" in str(e) and ext.config.voice == ""         # rejected, old config kept
            assert not ext._fatal
            rec.events.clear()
            await ext.request_tts(text_input("r", "Mabuhay.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=5)
            assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END, (
                "a rejected voice in update_configs (cmd returned OK) made TTS permanently FATAL")
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())


@pytest.mark.skipif(not _gw_up(), reason="real gateway not running")
def test_qa_r2_real_swap_then_immediate_next_request_succeeds():
    """Real gateway: engine swap ends a, the base releases b at once; b must complete on a new socket."""
    import asyncio
    import urllib.request

    async def go():
        ext, rec = make_extension(GW, model="halohalo/tts-fil")
        ext._reconnect_delay = lambda attempt: 0.05
        orig_finish, tasks = ext.finish_request, []

        async def finish(request_id, reason=None, error=None):
            await orig_finish(request_id, reason=reason, error=error)
            if request_id == "a":
                tasks.append(asyncio.create_task(ext.request_tts(text_input("b", "Mabuhay.", end=True))))
            await asyncio.sleep(0)

        ext.finish_request = finish
        try:
            await ext._ensure_connected()
            await ext.request_tts(text_input("a", "Ito ay isang mahabang pangungusap. " * 8, end=True))
            await wait_for(lambda: rec.of("audio"), timeout=5)
            u = urlparse(GW)
            req = urllib.request.Request(f"http://{u.hostname}:{u.port}/v1/admin/swap?engine=tts&spec=mock", method="POST")
            await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=20).read())
            await wait_for(lambda: len(rec.of("audio_end")) == 2, timeout=25, what="a and b end")
            ends = {e["request_id"]: e["reason"] for e in rec.of("audio_end")}
            b_err = [f["error"].message for f in rec.of("finish") if f["request_id"] == "b" and f["error"]]
            assert ends["b"] == TTSAudioEndReason.REQUEST_END, (ends, b_err)
            assert not ext._fatal
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

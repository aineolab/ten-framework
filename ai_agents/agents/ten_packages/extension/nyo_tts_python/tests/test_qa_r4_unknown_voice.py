"""QA round 4 (fixed): a voice the gateway refuses on a route without a voice catalogue (SpeechT5 / hil today; the
mock engines here) was treated as a transient outage: the handshake's HTTP 400 {"code":"unknown_voice"} body was
never read and served_verdict() ignored the route's `voices` allow-list, so requests failed NON_FATAL "400" for
reconnect_window_ms and then one FATAL connect_failed; update_configs accepted the voice. Now: FATAL unknown_voice
at once, and update_configs rejects it (old config kept).

Runs the REAL serve/ gateway (mock engines, auth off) on 127.0.0.1:9146; skipped when serve/ is not available.
"""
import sys
import asyncio
import os
import subprocess
import time
import urllib.request

import pytest

from ten_ai_base.message import ModuleErrorCode

from .helpers import make_extension, run, text_input, wait_for

PY = os.environ.get("NYO_QA_SERVE_PYTHON", sys.executable)
SERVE = os.environ.get("NYO_QA_SERVE_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[3])
PORT = 9146
pytestmark = pytest.mark.skipif(not (os.path.exists(PY) and os.path.isdir(f"{SERVE}/serve")),
                                reason="serve/ gateway not available")


@pytest.fixture(scope="module")
def gateway():
    env = dict(os.environ, SAPIN_HOST="127.0.0.1", SAPIN_PORT=str(PORT), SAPIN_AUTH="off",
               SAPIN_TTS_LANGUAGES="fil,hil", SAPIN_ASR_LANGUAGES="fil", SAPIN_MIN_AVAILABLE_GIB="0",
               SAPIN_USAGE_SINK="none")
    p = subprocess.Popen([PY, "-m", "serve"], cwd=SERVE, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=0.5).read()
                break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.skip("serve/ gateway did not start (set NYO_QA_SERVE_PYTHON to a venv with its deps)")
        yield f"ws://127.0.0.1:{PORT}"
    finally:
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(5)


def test_qa_r4_refused_voice_is_fatal_unknown_voice_at_once(gateway):
    async def go():
        ext, rec = make_extension(gateway, api_key="k", model="halohalo/tts-hil", voice="alloy",
                                  request_timeout_ms=1500, reconnect_window_ms=4000)
        try:
            t0 = time.monotonic()
            await ext.request_tts(text_input("r", "kumusta", end=True))
            await wait_for(lambda: rec.of("error") or rec.of("finish_error"), timeout=10, what="an error")
            dt = time.monotonic() - t0
            err = (rec.of("error") or rec.of("finish_error"))[0]["error"]
            got = (int(err.code), err.vendor_info.code, round(dt, 1))
            assert got[:2] == (int(ModuleErrorCode.FATAL_ERROR.value), "unknown_voice"), \
                f"first error (code, vendor code, s): {got}"
            assert dt < 1.0 and rec.of("audio_end")[-1]["request_id"] == "r"
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())


def test_qa_r4_update_configs_rejects_refused_voice(gateway):
    async def go():
        ext, rec = make_extension(gateway, api_key="k", model="halohalo/tts-fil")
        try:
            await ext._ensure_connected()
            with pytest.raises(ValueError, match="unknown_voice"):
                await ext.update_configs({"params": {"voice": "alloy"}})
            assert ext.config.voice == "" and not ext._fatal
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

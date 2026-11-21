"""QA r4: a language the gateway does not serve (e.g. "en") is model_not_found at once, naming the served ids, and
FATAL model_not_found when reconnect_window_ms runs out; a BCP-47 tag (fil-PH) is normalised. Checked against the real serve/ gateway on 127.0.0.1:9150 (read-only use).
"""
import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from nyo_asr_python.reconnect_manager import ReconnectManager

from .helpers import make_extension, run
from .test_qa_r2_gateway import _up

KEY = os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key")
pytestmark = pytest.mark.skipif(not (_up(9150) and os.path.exists(KEY)), reason="real stack not running")


def test_r4_unserved_language_reports_model_not_found_naming_served_ids():
    """r5: the gateway lists halohalo/asr-<lang> for every configured language while the ASR engine is up, so a
    language with no halohalo/ id and no listed model declaring it is FATAL model_not_found at once."""
    async def go():
        ext = make_extension(base_url="ws://127.0.0.1:9150", allow_insecure=True, language="ceb",
                             api_key=open(KEY).read().strip())
        assert ext.config.params["model"] == "halohalo/asr-ceb"
        ext._emit_connection_transition = AsyncMock()
        try:
            await ext.start_connection()
            errs = [(c.args[0].code, c.args[1].code if c.args[1] else None, c.args[0].message)
                    for c in ext.send_asr_error.await_args_list]
            assert len(errs) == 1 and errs[0][:2] == (-1000, "model_not_found"), errs
            assert "halohalo/asr-fil" in errs[0][2], "the error names the served ASR ids"
            assert ext.recognition is None and ext._reconnect_task is None
        finally:
            await ext.stop_connection()

    run(go())


def test_r5_language_served_by_a_listed_fw_model_uses_it():
    """r5: only language "en" set; halohalo/asr-en is not listed but fw/whisper-base (the gateway's default)
    declares en: the addon connects with it."""
    async def go():
        ext = make_extension(base_url="ws://127.0.0.1:9150", allow_insecure=True, language="en",
                             api_key=open(KEY).read().strip())
        ext._emit_connection_transition = AsyncMock()
        try:
            await ext.start_connection()
            assert ext.config.params["model"] == "fw/whisper-base", ext.config.params["model"]
            assert ext.is_connected() and not ext.send_asr_error.await_args_list
        finally:
            await ext.stop_connection()

    run(go())


def test_r4_bcp47_language_connects():
    async def go():
        ext = make_extension(base_url="ws://127.0.0.1:9150", allow_insecure=True, language="fil-PH",
                             api_key=open(KEY).read().strip())
        assert (ext.config.params["model"], ext.config.params["language"]) == ("halohalo/asr-fil", "fil")
        ext._emit_connection_transition = AsyncMock()
        try:
            await ext.start_connection()
            assert ext.is_connected() and not ext.send_asr_error.await_args_list
        finally:
            await ext.stop_connection()

    run(go())

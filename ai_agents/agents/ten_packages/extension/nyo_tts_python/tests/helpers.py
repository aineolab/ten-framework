"""Test harness: a real NYOTTSExtension + client against the in-process fake NRSP server.

Only the base-class *senders* (the things that need a live TEN graph) are replaced with
recorders; the extension logic, the NRSP client and the websocket are real.
"""
import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

from ten_ai_base.struct import TTSTextInput

from fake_nrsp_server import FakeNRSPServer
from nyo_tts_python.config import NYOTTSConfig
from nyo_tts_python.extension import NYOTTSExtension


def make_config(base_url="ws://127.0.0.1:1", **params) -> NYOTTSConfig:
    cfg = NYOTTSConfig(params={"api_key": "nyo-test-key", "base_url": base_url, **params})
    cfg.update_params()
    cfg.validate_config()
    return cfg


class Recorder:
    """Ordered log of everything the extension emits to the TEN graph."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def add(self, name, **kw):
        self.events.append((name, kw))

    def names(self):
        return [n for n, _ in self.events]

    def of(self, name):
        return [kw for n, kw in self.events if n == name]


def make_extension(base_url="ws://127.0.0.1:1", **params):
    ext = NYOTTSExtension("nyo_tts_python")
    ext.ten_env = MagicMock()
    ext.config = make_config(base_url, **params)
    rec = Recorder()

    async def audio_start(request_id, turn_id=-1, extra_metadata=None):
        rec.add("audio_start", request_id=request_id)
        ext.current_audio_request_id = request_id

    async def audio_data(data, timestamp=0):
        rec.add("audio", n=len(data), ts=timestamp, rate=ext.synthesize_audio_sample_rate())

    async def audio_end(request_id, request_event_interval_ms, request_total_audio_duration_ms,
                        turn_id=-1, reason=None, extra_metadata=None):
        rec.add("audio_end", request_id=request_id, reason=reason, duration_ms=request_total_audio_duration_ms,
                interval_ms=request_event_interval_ms)

    async def ttfb(request_id, ttfb_ms, turn_id=-1, extra_metadata=None):
        rec.add("ttfb", request_id=request_id, ms=ttfb_ms, extra=extra_metadata)

    async def error(request_id, error, turn_id=-1, extra_metadata=None):
        rec.add("error", request_id=request_id, error=error)

    async def finish(request_id, reason=None, error=None):
        rec.add("finish", request_id=request_id, reason=reason, error=error)
        if error is not None:
            rec.add("finish_error", request_id=request_id, error=error)

    async def usage(request_id="", extra_metadata=None):
        rec.add("usage", request_id=request_id)

    async def text_result(t):
        rec.add("text_result", result=t)

    async def conn(name, **kw):
        rec.add(name, **kw)

    ext.send_tts_audio_start = audio_start
    ext.send_tts_audio_data = audio_data
    ext.send_tts_audio_end = audio_end
    ext.send_tts_ttfb_metrics = ttfb
    ext.send_tts_error = error
    ext.finish_request = finish
    ext.send_usage_metrics = usage
    ext.send_tts_text_result = text_result
    async def metrics(m, request_id=""):
        rec.add("metrics", metrics=m.metrics)

    ext.send_metrics = metrics
    ext.metrics_connect_delay = AsyncMock()
    ext.on_connecting = lambda: conn("connecting")
    ext.on_connected = lambda: conn("connected")
    ext.on_disconnected = lambda **kw: conn("disconnected", **kw)
    return ext, rec


def text_input(rid, text, end=False, **meta):
    return TTSTextInput(request_id=rid, text=text, text_input_end=end, metadata=meta or {"session_id": "s1"})


async def wait_for(pred, timeout=3.0, what="condition"):
    end = asyncio.get_event_loop().time() + timeout
    while not pred():
        if asyncio.get_event_loop().time() > end:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


@contextlib.asynccontextmanager
async def harness(key="nyo-test-key", server_kw=None, **params):
    async with FakeNRSPServer(**(server_kw or {})) as srv:
        ext, rec = make_extension(srv.url, api_key=key, **params)
        try:
            yield srv, ext, rec
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()


def run(coro):
    return asyncio.run(coro)

"""QA round 5: update_configs that lands while a connect (prewarm, reconnect, background connect) is mid-handshake.

_apply_config mutates self.config and recycles self.client, but the in-flight connect built its URL from the old
values and installs its socket afterwards; _recycle_socket's own _ensure_connected then finds that socket "connected"
and keeps it. Every following request is synthesised with the OLD voice/model, and the cmd returned OK.
"""
import asyncio

from nyo_tts_python import tts_client

from .helpers import harness, run, text_input, wait_for


def test_qa_r5_update_configs_during_prewarm_handshake_is_lost():
    async def go():
        real_connect = tts_client.websockets.connect

        async def slow_connect(url, **kw):  # a WAN handshake (TLS + upgrade) that takes a few hundred ms
            await asyncio.sleep(0.4)
            return await real_connect(url, **kw)

        tts_client.websockets.connect = slow_connect
        try:
            async with harness(voice="") as (srv, ext, rec):
                prewarm = asyncio.create_task(ext._prewarm())
                await asyncio.sleep(0.15)  # preflight done, handshake in flight with voice=""
                await ext.update_configs({"params": {"voice": "nova"}})  # idle: applied at once, cmd OK
                await prewarm
                assert ext.config.voice == "nova"
                await ext.request_tts(text_input("r1", "Mabuhay.", end=True))
                await wait_for(lambda: rec.of("audio_end"), timeout=5)
                texts = srv.received("text")
                assert texts, "nothing reached the server"
                # which socket carried r1?
                used = [c for c in srv.connections if c["path"] == "/v1/tts"]
                voices = [c["query"].get("voice", "") for c in used]
                assert voices[-1] == "nova" and len(used) >= 1, (
                    f"update_configs(voice=nova) returned OK but the request ran on a socket opened with "
                    f"voice={voices[-1]!r} (sockets opened: {voices})")
        finally:
            tts_client.websockets.connect = real_connect

    run(go())

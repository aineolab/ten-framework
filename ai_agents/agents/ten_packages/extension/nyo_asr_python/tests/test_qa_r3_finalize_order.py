"""QA r3: asr_finalize around a reconnect."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from nyo_asr_python.recognition import NYOASRRecognition

from .helpers import make_extension, run
from .test_protocol import Frame, session, wait_for


def test_finalize_during_on_open_waits_for_the_keep_buffer_replay(monkeypatch):
    """on_open awaits send_connect_delay_metrics/on_connected before replaying the Keep buffer; a finalize
    arriving then waits for the replay, so it never overtakes audio captured before it."""
    wire = []
    orig_audio, orig_fin = NYOASRRecognition.send_audio_frame, NYOASRRecognition.send_finalize

    async def audio(self, data):
        wire.append("audio")
        await orig_audio(self, data)

    async def fin(self, fid):
        wire.append("finalize")
        await orig_fin(self, fid)

    monkeypatch.setattr(NYOASRRecognition, "send_audio_frame", audio)
    monkeypatch.setattr(NYOASRRecognition, "send_finalize", fin)

    async def go():
        async with session() as (srv, ext):
            ext._preflight_ok = True
            for _ in range(20):  # 400 ms captured while disconnected -> Keep buffer
                await ext._handle_audio_frame(ext.ten_env, Frame())

            async def slow_connected():
                await asyncio.sleep(0.05)  # send_data of connection_status_changed
            ext.on_connected.side_effect = slow_connected
            conn = asyncio.create_task(ext.start_connection())
            await wait_for(lambda: ext.recognition is not None and ext.recognition.is_connected(), what="socket up")
            ext.finalize_id = "f1"
            await ext.finalize(None)  # the user stopped talking: VAD sends asr_finalize
            await conn
            assert wire.index("finalize") == len([w for w in wire if w == "audio"]), \
                "finalize sent before buffered audio captured before it"

    run(go())


def test_finalize_while_disconnected_promotes_the_last_interim():
    """finalize() with no socket promotes the dead socket's last interim once, then answers finalize_end."""
    async def go():
        ext = make_extension()
        ext._last_interim = {"type": "result", "id": "utt-3", "text": "pupunta ako", "final": False,
                             "start_ms": 0, "duration_ms": 900}
        ext.recognition = None
        ext.finalize_id = "f1"
        await ext.finalize(None)
        finals = [c.args[0] for c in ext.send_asr_result.await_args_list if c.args[0].final]
        assert [r.text for r in finals] == ["pupunta ako"] and ext.send_asr_finalize_end.await_count == 1
        assert finals[0].metadata["asr_info"].get("promoted_on_timeout") is True
        ext.finalize_id = "f2"
        await ext.finalize(None)  # once: nothing left to promote
        assert len([c for c in ext.send_asr_result.await_args_list if c.args[0].final]) == 1

    run(go())


def test_duplicate_finalize_id_ends_both_waiters_on_the_first_finalize_end():
    async def go():
        rec = NYOASRRecognition("k", MagicMock(), MagicMock(), {}, AsyncMock())
        rec.websocket = MagicMock()
        rec.websocket.send = AsyncMock()
        rec.is_connected = lambda: True
        await rec.send_finalize("f1")
        a = asyncio.create_task(rec.wait_finalize_end("f1", 3000))
        await asyncio.sleep(0)
        await rec.send_finalize("f1")
        b = asyncio.create_task(rec.wait_finalize_end("f1", 3000))
        await asyncio.sleep(0)
        t0 = asyncio.get_event_loop().time()
        await rec._dispatch({"type": "finalize_end", "id": "f1"})
        assert await a and await b and asyncio.get_event_loop().time() - t0 < 0.5

    run(go())


def test_replay_keeps_the_keep_buffer_size_true():
    async def go():
        ext = make_extension()
        for _ in range(10):
            ext.buffered_frames.put_nowait(Frame())
        ext.buffered_frames_size = 10 * len(Frame().get_buf())
        seen = []

        async def send(frame):
            seen.append(ext.buffered_frames_size)
            return True
        ext._send_frame = send
        await ext._flush_buffered_audio_frames()
        step = len(Frame().get_buf())
        assert seen == [step * (9 - i) for i in range(10)] and ext.buffered_frames_size == 0

    run(go())

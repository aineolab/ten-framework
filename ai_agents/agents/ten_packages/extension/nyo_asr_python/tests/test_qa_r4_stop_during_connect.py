"""QA r4: on_stop while start_connection (auto-connect / trigger_connect) is still in its GET /v1/models preflight."""
import asyncio

from websockets.protocol import State

from nyo_asr_python.recognition import NYOASRRecognition

from .helpers import run
from .test_protocol import session


def test_r4_stop_during_preflight_opens_no_socket(monkeypatch):
    made = []
    orig = NYOASRRecognition.__init__

    def track(self, *a, **k):
        orig(self, *a, **k)
        made.append(self)

    monkeypatch.setattr(NYOASRRecognition, "__init__", track)

    async def go():
        async with session() as (srv, ext):
            async def slow_verdict():
                await asyncio.sleep(0.3)          # /v1/models slow (cold gateway, far region)
                return "ok", "", ""
            ext._models_verdict = slow_verdict
            conn = asyncio.create_task(ext.start_connection())
            await asyncio.sleep(0.05)
            ext.stopped = True                    # base on_stop
            await ext.stop_connection()
            await conn
            await asyncio.sleep(0.2)
            await ext.on_deinit(ext.ten_env)       # TEN on_deinit: last chance to close it
            live = [r for r in made if r.websocket is not None and r.websocket.state == State.OPEN]
            print("open sockets after stop:", len(live), "server connections:", len(srv.connections))
            assert not live, "a socket was opened after on_stop and nothing will close it (on_deinit does not)"

    run(go())


def test_r4_stop_during_handshake_keeps_no_socket(monkeypatch):
    made = []
    orig_start = NYOASRRecognition.start

    async def start(self, *a, **k):
        made.append(self)
        ext_ref[0].stopped = True                 # on_stop lands while the upgrade is in flight
        ext_ref[0]._stop_requested = True
        await orig_start(self, *a, **k)

    monkeypatch.setattr(NYOASRRecognition, "start", start)
    ext_ref = [None]

    async def go():
        async with session() as (srv, ext):
            ext_ref[0] = ext
            ext._preflight_ok = True
            ext._stop_requested = False
            await ext._connect_recognition()
            await asyncio.sleep(0.2)
            assert made and ext.recognition is None
            assert not [r for r in made if r.websocket is not None and r.websocket.state == State.OPEN]

    run(go())

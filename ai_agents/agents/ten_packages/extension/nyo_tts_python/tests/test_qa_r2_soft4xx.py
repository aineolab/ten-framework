"""QA round 2 (rule B): a 400/404 on the upgrade (seen live while the gateway restarts) is FATAL only when
GET /v1/models conclusively says the route is not served; otherwise the extension keeps retrying inside
reconnect_window_ms (no fixed 3-strikes count)."""
import json
import time

from ten_ai_base.message import ModuleErrorCode
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from .helpers import make_extension, run, text_input, wait_for

FATAL = int(ModuleErrorCode.FATAL_ERROR.value)


def _server(models):
    """Every upgrade -> 404; GET /v1/models -> 200 with ``models`` (or 404 when None)."""
    async def reject(conn, request):
        if request.path.startswith("/v1/models") and models is not None:
            body = json.dumps({"object": "list", "data": models}).encode()
            return Response(200, "OK", Headers({"Content-Type": "application/json"}), body)
        return Response(404, "Not Found", Headers({}), b"{}")

    return serve(lambda ws: None, "127.0.0.1", 0, process_request=reject)


def _fatal_errors(rec):
    return [e["error"] for e in rec.of("error") if int(e["error"].code) == FATAL]


def test_qa_r2_restart_404s_shorter_than_the_window_must_not_be_fatal():
    async def go():
        for models in (None, [], [{"id": "halohalo/asr-fil", "kind": "asr"}]):  # all inconclusive
            async with _server(models) as server:
                url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                ext, rec = make_extension(url, reconnect_window_ms=120000)
                ext._preflight_ok = True
                ext._reconnect_delay = lambda attempt: 0.05
                ext._schedule_reconnect()
                t0 = time.monotonic()
                try:
                    await wait_for(lambda: ext._fatal, timeout=1.5, what="fatal")
                except AssertionError:
                    continue  # correct: still retrying inside the 120 s window
                finally:
                    ext._stopped = True
                    ext._cancel_reconnect_task()
                raise AssertionError(f"models={models}: FATAL after {time.monotonic() - t0:.1f} s of 404s "
                                     f"(window 120 s): {[(e.vendor_info.code, e.message) for e in _fatal_errors(rec)]}")

    run(go())


def test_qa_r2_404_with_a_listing_that_lacks_the_model_retries_then_fatal_model_not_found():
    """Rule A': the model may be an engine that is down: one NON_FATAL model_not_found, retries inside the window,
    and the FATAL at the end of the window carries model_not_found."""
    async def go():
        async with _server([{"id": "halohalo/tts-hil", "kind": "tts"}]) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            ext, rec = make_extension(url, model="halohalo/tts-fil", reconnect_window_ms=700)
            ext._preflight_ok = True                    # the model disappeared after start
            ext._reconnect_delay = lambda attempt: 0.05
            await ext._begin_request(text_input("r", "x"))  # a request waits (idle, the window end is not FATAL)
            t0 = time.monotonic()
            ext._schedule_reconnect()
            try:
                await wait_for(lambda: ext._fatal, timeout=4, what="fatal")
            finally:
                ext._stopped = True
                ext._cancel_reconnect_task()
            assert time.monotonic() - t0 >= 0.6         # not before the window
            codes = [(int(e["error"].code), e["error"].vendor_info.code) for e in rec.of("error")]
            assert codes == [(int(ModuleErrorCode.NON_FATAL_ERROR.value), "model_not_found"),
                             (FATAL, "model_not_found")], codes
            assert "halohalo/tts-hil" in rec.of("error")[0]["error"].message

    run(go())

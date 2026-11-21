from nyo_asr_python.recognition import extract_status_code


class _Resp:
    status_code = 401


class _WithResponse(Exception):
    response = _Resp()


class _WithStatus(Exception):
    status_code = 503


def test_status_from_response():
    assert extract_status_code(_WithResponse()) == 401


def test_status_direct():
    assert extract_status_code(_WithStatus()) == 503


def test_status_absent():
    assert extract_status_code(RuntimeError("x")) is None


def test_fatal_close_task_is_kept_alive_until_done():
    """ten r3 F3: the background close started on a FATAL is strongly referenced until it finishes."""
    import asyncio
    from unittest.mock import MagicMock
    from ten_ai_base.message import ModuleError, ModuleErrorCode
    from .helpers import make_extension

    async def go():
        ext = make_extension()
        gate = asyncio.Event()
        rec = MagicMock()

        async def slow_close():
            await gate.wait()

        rec.close = slow_close
        ext.recognition = rec
        await ext._send_error(ModuleError(module="asr", code=ModuleErrorCode.FATAL_ERROR.value, message="x"))
        assert len(ext._bg_tasks) == 1
        gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(ext._bg_tasks) == 0

    asyncio.run(go())

from .helpers import make_extension, run


def test_usage_becomes_vendor_metrics():
    ext = make_extension()
    run(ext.on_usage({"type": "usage", "audio_ms": 1500, "credits": 0.25, "extra": 1}))
    ext.send_vendor_metrics.assert_awaited_once_with({"audio_ms": 1500, "credits": 0.25})


def test_empty_usage_is_ignored():
    ext = make_extension()
    run(ext.on_usage({"type": "usage"}))
    ext.send_vendor_metrics.assert_not_awaited()


def test_vendor_metadata_carries_model_and_masked_key_field():
    ext = make_extension(model="halohalo/asr-ceb")
    md = ext.vendor_metadata()
    assert md["model"] == "halohalo/asr-ceb" and md["mode"] == "nrsp.v1"
    # the base class redacts "key" before it leaves the process
    from ten_ai_base.utils import redact_json
    assert redact_json(md)["key"] != "nyo-test-key"


def test_server_side_ttfw_ttlw_are_surfaced_as_vendor_metrics():
    ext = make_extension()
    run(ext.on_usage({"server_ttfw_ms": 329.8}))
    run(ext.on_usage({"server_ttlw_ms": 60.6}))
    got = [c.args[0] for c in ext.send_vendor_metrics.await_args_list]
    assert got == [{"server_ttfw_ms": 329.8}, {"server_ttlw_ms": 60.6}]


def test_vendor_metadata_masks_the_key_at_source():
    """ten r3 F4: the raw key never leaves vendor_metadata(), whatever the base class does later."""
    md = make_extension(api_key="nyo-super-secret").vendor_metadata()
    assert "nyo-super-secret" not in md["key"] and md["key"]


# ---------------------------------------------------- client-side TTFW / TTLW (ten r3 F1)
from unittest.mock import AsyncMock, MagicMock  # noqa: E402


class _Frame:
    def __init__(self, n=640):
        self._d = bytearray(b"\x10\x20" * (n // 2))

    def lock_buf(self):
        return self._d

    def unlock_buf(self, _):
        pass


def _live(ext):
    rec = MagicMock()
    rec.is_connected.return_value = True
    rec.bytes_per_ms = 32
    rec.send_audio_frame = AsyncMock()
    rec.send_finalize = AsyncMock()
    ext.recognition = rec
    return rec


def _metrics(ext, key):
    return [c.args[0][key] for c in ext.send_vendor_metrics.await_args_list if key in c.args[0]]


def test_client_ttfw_once_per_turn_when_the_server_sends_none():
    async def go():
        ext = make_extension()
        _live(ext)
        await ext._send_frame(_Frame())
        await ext.on_result({"id": "utt-0", "text": "kumusta", "final": False})
        await ext.on_result({"id": "utt-0", "text": "kumusta ka", "final": False})
        await ext.on_result({"id": "utt-0", "text": "kumusta ka", "final": True})
        assert len(_metrics(ext, "client_ttfw_ms")) == 1
        await ext._send_frame(_Frame())                                  # next turn
        await ext.on_result({"id": "utt-1", "text": "salamat", "final": True})
        ttfw = _metrics(ext, "client_ttfw_ms")
        assert len(ttfw) == 2 and all(v >= 0 for v in ttfw)

    run(go())


def test_server_ttfw_wins_over_the_client_value():
    async def go():
        ext = make_extension()
        _live(ext)
        await ext._send_frame(_Frame())
        await ext.on_result({"id": "utt-0", "text": "kumusta", "final": False, "ttfw_ms": 312.0})
        assert _metrics(ext, "client_ttfw_ms") == []

    run(go())


def test_client_ttlw_from_finalize_to_its_final():
    async def go():
        ext = make_extension()
        rec = _live(ext)

        async def answer(fid, timeout_ms):
            await ext.on_result({"id": "utt-0", "text": "kumusta", "final": True,
                                 "metadata": {"finalize_id": fid}})
            return True

        rec.wait_finalize_end = AsyncMock(side_effect=answer)
        ext.finalize_id = "fin-9"
        await ext.finalize("s")
        ttlw = _metrics(ext, "client_ttlw_ms")
        assert len(ttlw) == 1 and ttlw[0] >= 0

    run(go())


def test_server_ttlw_wins_over_the_client_value():
    async def go():
        ext = make_extension()
        rec = _live(ext)

        async def answer(fid, timeout_ms):
            await ext.on_result({"id": "utt-0", "text": "kumusta", "final": True,
                                 "metadata": {"finalize_id": fid}})
            await ext.on_usage({"server_ttlw_ms": 88.0})
            return True

        rec.wait_finalize_end = AsyncMock(side_effect=answer)
        await ext.finalize("s")
        assert _metrics(ext, "client_ttlw_ms") == [] and _metrics(ext, "server_ttlw_ms") == [88.0]

    run(go())


def test_auto_final_of_another_cut_is_not_the_finalize_ttlw():
    async def go():
        ext = make_extension()
        rec = _live(ext)

        async def answer(fid, timeout_ms):
            await ext.on_result({"id": "utt-0", "text": "una", "final": True,
                                 "metadata": {"finalize_id": "auto-1"}})       # VAD cut, not ours
            return True

        rec.wait_finalize_end = AsyncMock(side_effect=answer)
        await ext.finalize("s")
        assert _metrics(ext, "client_ttlw_ms") == []

    run(go())

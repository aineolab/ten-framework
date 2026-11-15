"""Exactly-once finalisation and reason mapping."""
from ten_ai_base.message import TTSAudioEndReason

from nyo_tts_python.extension import _END_REASONS, _Request

from .helpers import make_extension, run


def test_end_reason_mapping():
    assert _END_REASONS == {
        "end": TTSAudioEndReason.REQUEST_END,
        "flushed": TTSAudioEndReason.INTERRUPTED,
        "interrupted": TTSAudioEndReason.INTERRUPTED,
        "error": TTSAudioEndReason.ERROR,
    }


def test_finalize_is_idempotent():
    async def go():
        ext, rec = make_extension()
        ext._req = _Request("r")
        await ext._finalize_request(TTSAudioEndReason.REQUEST_END)
        await ext._finalize_request(TTSAudioEndReason.INTERRUPTED)
        assert rec.names().count("audio_end") == 1 and rec.names().count("finish") == 1

    run(go())


def test_callbacks_for_other_requests_are_ignored():
    async def go():
        ext, rec = make_extension()
        ext._req = _Request("current")
        await ext._on_audio("old", 0, b"\x00\x00")
        await ext._on_audio_end("old", "end", 0, {})
        await ext._on_audio_start("old", 16000, {})
        assert rec.events == [] and ext._sample_rate is None

    run(go())


def test_unknown_end_reason_defaults_to_request_end():
    async def go():
        ext, rec = make_extension()
        ext._req = _Request("r")
        await ext._on_audio_end("r", "something-new", 0, {})
        assert rec.of("audio_end")[0]["reason"] == TTSAudioEndReason.REQUEST_END

    run(go())


def test_unsupported_sample_rate_is_ignored():
    async def go():
        ext, _ = make_extension()
        ext._req = _Request("r")
        await ext._on_audio_start("r", 12345, {})
        assert ext._sample_rate is None and ext.synthesize_audio_sample_rate() == 24000

    run(go())


def test_audio_gap_is_counted():
    async def go():
        ext, rec = make_extension()
        ext._req = _Request("r")
        await ext._on_audio("r", 0, b"\x00" * 4800)     # 100 ms @ 24k
        await ext._on_audio("r", 500, b"\x00" * 4800)   # server skipped 400 ms
        assert ext._req.gaps == 1

    run(go())


def test_vendor_metadata_redacts_key():
    ext, _ = make_extension(api_key="super-secret-key")
    from ten_ai_base.utils import redact_json
    md = ext.vendor_metadata()
    assert md["model"] == "halohalo/tts-fil" and "super-secret-key" not in json_dumps(redact_json(md))


def json_dumps(o):
    import json
    return json.dumps(o)


def test_text_after_text_input_end_is_not_forwarded():
    from .helpers import harness, text_input, wait_for

    async def go():
        async with harness() as (srv, ext, rec):
            await ext.request_tts(text_input("999", "", end=True))
            await ext.request_tts(text_input("999", " ", end=True))
            await ext.request_tts(text_input("999", "", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert len(srv.received("text")) == 0  # an empty request ends locally, no round trip
            assert len(rec.of("audio_end")) == 1 and len(rec.of("finish")) == 1
            assert rec.of("error") == []

    run(go())


def test_empty_end_after_real_text_still_goes_to_the_server():
    from .helpers import harness, text_input, wait_for

    async def go():
        async with harness() as (srv, ext, rec):
            await ext.request_tts(text_input("r", "kumusta", end=False))
            await ext.request_tts(text_input("r", "", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert [m["end"] for m in srv.received("text")] == [False, True]

    run(go())


def test_stale_request_server_error_is_not_surfaced():
    async def go():
        ext, rec = make_extension()
        await ext._on_error({"type": "error", "code": "stale_request", "request_id": "old", "fatal": False,
                             "vendor": {"name": "halohalo", "code": "stale_request", "message": "x"}})
        assert rec.events == []

    run(go())


def test_update_configs_switches_route_on_a_fresh_socket_between_requests():
    from .helpers import harness, text_input, wait_for

    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await ext.request_tts(text_input("a", "one ", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert ext.synthesize_audio_sample_rate() == 24000
            await ext.update_configs({"params": {"model": "halohalo/tts-speecht5", "voice": "ana"}})
            assert len(srv.connections) == 2
            q = srv.connections[1]["query"]
            assert q["model"] == "halohalo/tts-speecht5" and q["voice"] == "ana"
            rec.events.clear()
            await ext.request_tts(text_input("b", "two ", end=True))
            await wait_for(lambda: "finish" in rec.names())
            assert {e["rate"] for e in rec.of("audio")} == {16000}

    run(go())


def test_update_configs_without_change_keeps_the_socket():
    from .helpers import harness

    async def go():
        async with harness(model="halohalo/tts-ceb") as (srv, ext, rec):
            await ext._ensure_connected()
            await ext.update_configs({"params": {"model": "halohalo/tts-ceb"}})
            assert len(srv.connections) == 1

    run(go())


def test_vendor_metadata_masks_the_key_at_source():
    """ten r3 F4: masked in vendor_metadata() itself; base-class redaction is a second layer."""
    ext, _ = make_extension(api_key="super-secret-key")
    assert "super-secret-key" not in ext.vendor_metadata()["key"]


def test_gc_freeze_param_defaults_on_and_can_be_disabled():
    from .helpers import make_config
    assert make_config().gc_freeze is True
    assert make_config(gc_freeze=False).gc_freeze is False

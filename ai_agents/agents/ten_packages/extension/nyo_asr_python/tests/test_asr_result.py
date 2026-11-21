from unittest.mock import MagicMock

from .helpers import make_extension, run


def _ext(**kw):
    ext = make_extension(**kw)
    ext.audio_timeline = MagicMock()
    ext.audio_timeline.get_audio_duration_before_time.side_effect = lambda t: t
    ext.metadata = {"session_id": "s-1", "turn": 3}
    return ext


def test_final_result_maps_fields_and_backend():
    ext = _ext(language="fil")

    async def go():
        await ext._emit_asr_result(
            {"text": "kumusta mundo", "start_ms": 100, "duration_ms": 800, "backend": "whisper-ct2",
             "words": [{"word": "kumusta", "start_ms": 100, "duration_ms": 400, "stable": True},
                       {"word": "mundo", "start_ms": 500, "duration_ms": 400}]},
            final=True,
        )

    run(go())
    r = ext.send_asr_result.await_args.args[0]
    assert (r.text, r.final, r.start_ms, r.duration_ms) == ("kumusta mundo", True, 100, 800)
    assert r.language == "fil-PH"
    assert r.metadata["session_id"] == "s-1"
    info = r.metadata["asr_info"]
    assert info["backend"] == "whisper-ct2" and info["vendor"] == "nyo"
    assert info["model"] == "halohalo/asr-fil" and info["turn"] == 3
    assert [w.word for w in r.words] == ["kumusta", "mundo"]
    assert all(w.stable for w in r.words[1:])  # unspecified -> inherits final


def test_interim_words_keep_server_stability_flag():
    ext = _ext()
    run(ext._emit_asr_result(
        {"text": "a b", "words": [{"word": "a", "stable": True}, {"word": "b", "stable": False}]},
        final=False))
    r = ext.send_asr_result.await_args.args[0]
    assert [w.stable for w in r.words] == [True, False]
    assert r.metadata["asr_info"]["locked"] is False


def test_empty_text_is_dropped():
    ext = _ext()
    run(ext._emit_asr_result({"text": ""}, final=True))
    ext.send_asr_result.assert_not_awaited()


def test_empty_interim_is_dropped_even_with_an_open_finalize():
    ext = _ext()
    ext._open_finalizes = ["f1"]
    run(ext._emit_asr_result({"text": ""}, final=False))
    ext.send_asr_result.assert_not_awaited()


def test_empty_final_answering_an_open_finalize_is_forwarded():
    # Silence before asr_finalize: the engine's final is empty. The caller still gets a final for its
    # finalize (closes the turn), as Deepgram does; downstream agents ignore empty text.
    ext = _ext()
    ext._open_finalizes = ["f1"]
    run(ext._emit_asr_result({"text": "", "metadata": {"finalize_id": "f1"}}, final=True))
    r = ext.send_asr_result.await_args.args[0]
    assert (r.text, r.final) == ("", True)


def test_empty_final_without_finalize_id_answers_the_open_finalize():
    ext = _ext()
    ext._open_finalizes = ["f1"]
    run(ext._emit_asr_result({"text": ""}, final=True))
    assert ext.send_asr_result.await_args.args[0].final is True


def test_empty_endpoint_final_for_another_finalize_is_dropped():
    ext = _ext()
    ext._open_finalizes = ["f1"]
    run(ext._emit_asr_result({"text": "", "metadata": {"finalize_id": "auto-3"}}, final=True))
    ext.send_asr_result.assert_not_awaited()


def test_backend_sticky_when_server_omits_it():
    ext = _ext()

    async def go():
        await ext._emit_asr_result({"text": "x", "backend": "ctc"}, final=False)
        await ext._emit_asr_result({"text": "xy"}, final=False)

    run(go())
    assert ext.send_asr_result.await_args.args[0].metadata["asr_info"]["backend"] == "ctc"


def test_unknown_backend_is_labelled():
    ext = _ext()
    run(ext._emit_asr_result({"text": "x"}, final=False))
    assert ext.send_asr_result.await_args.args[0].metadata["asr_info"]["backend"] == "unknown"


def test_offset_after_reconnect_is_added_to_start_ms():
    ext = _ext()
    ext.sent_user_audio_duration_ms_before_last_reset = 5000
    run(ext._emit_asr_result({"text": "x", "start_ms": 40}, final=False))
    assert ext.send_asr_result.await_args.args[0].start_ms == 5040


def test_auto_language_uses_server_language():
    ext = _ext(language="")
    run(ext._emit_asr_result({"text": "x", "language": "ceb"}, final=False))
    assert ext.send_asr_result.await_args.args[0].language == "ceb-PH"


def test_on_result_tracks_last_interim_until_final():
    ext = _ext()

    async def go():
        await ext.on_result({"type": "result", "text": "hel", "final": False})
        assert ext._last_interim["text"] == "hel"
        await ext.on_result({"type": "result", "text": "hello", "final": True})
        assert ext._last_interim is None

    run(go())

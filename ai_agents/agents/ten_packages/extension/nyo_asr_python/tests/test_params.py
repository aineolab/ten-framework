import json

import pytest

from nyo_asr_python.config import NYOASRConfig, normalize_language

from .helpers import make_config


def test_defaults_applied():
    cfg = NYOASRConfig(params={"api_key": "k", "base_url": "wss://nyo.example.test"})
    cfg.apply_defaults()
    p = cfg.params
    assert p["base_url"] == "wss://nyo.example.test"
    assert p["endpointing_ms"] == 1000  # C1: server-side endpointing on by default
    assert p["tls_verify"] is True and p["allow_insecure"] is False
    assert p["sample_rate"] == 16000
    assert p["language"] == "fil"
    assert p["model"] == "halohalo/asr-fil"  # derived from the language
    assert cfg.finalize_timeout_ms == 2000
    cfg.validate_config()


@pytest.mark.parametrize(
    "lang,model",
    [("ceb", "halohalo/asr-ceb"), ("pag", "halohalo/asr-pag"), ("en", "nyo/auto")],
)
def test_model_derived_from_language(lang, model):
    assert make_config(language=lang).params["model"] == model


def test_explicit_model_kept():
    cfg = make_config(model="nvidia/nemotron-3.5-transcribe", language="en")
    assert cfg.params["model"] == "nvidia/nemotron-3.5-transcribe"


@pytest.mark.parametrize("url", [None, "", "   "])
def test_missing_base_url_rejected(url):
    """No default host: without base_url nothing may connect anywhere."""
    params = {"api_key": "k"} if url is None else {"api_key": "k", "base_url": url}
    cfg = NYOASRConfig(params=params)
    cfg.apply_defaults()
    with pytest.raises(ValueError, match="base_url is required"):
        cfg.validate_config()


def test_missing_api_key_rejected():
    cfg = NYOASRConfig(params={"api_key": "  ", "base_url": "wss://nyo.example.test"})
    cfg.apply_defaults()
    with pytest.raises(ValueError, match="API key"):
        cfg.validate_config()


@pytest.mark.parametrize("url", ["http://x", "speech-host.local:9100", "ws://"])
def test_bad_base_url_rejected(url):
    with pytest.raises(ValueError, match="base_url"):
        make_config(base_url=url).validate_config()


def test_bad_sample_rate_rejected():
    with pytest.raises(ValueError, match="sample_rate"):
        make_config(sample_rate=44100).validate_config()


@pytest.mark.parametrize(
    "code,tag",
    [("en", "en-US"), ("zh", "zh-CN"), ("es", "es-ES"), ("fil", "fil-PH"),
     ("ceb", "ceb-PH"), ("pag", "pag-PH"), ("eng", "en-PH"), ("xx-YY", "xx-YY"), ("", "")],
)
def test_language_normalisation(code, tag):
    assert normalize_language(code) == tag


def test_api_key_is_masked_in_logs():
    out = json.loads(make_config(api_key="super-secret-key").to_json(sensitive_handling=True))
    assert "super-secret-key" not in json.dumps(out)
    assert "super-secret-key" in make_config(api_key="super-secret-key").to_json()


def test_property_json_matches_manifest():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    prop = json.loads((root / "property.json").read_text())
    man = json.loads((root / "manifest.json").read_text())
    declared = man["api"]["property"]["properties"]["params"]["properties"]
    assert set(prop["params"]) <= set(declared)
    assert prop["params"]["api_key"] == "${env:NYO_API_KEY|}"
    assert man["dependencies"][0] == {"type": "system", "name": "ten_runtime_python", "version": "0.11"}
    assert man["dependencies"][1] == {"type": "system", "name": "ten_ai_base", "version": "0.7"}


def test_endpointing_ms_bool_is_rejected():
    """ten r3 F8: bool is an int subclass; `true` must not pass as 1 ms."""
    with pytest.raises(ValueError):
        make_config(endpointing_ms=True).validate_config()
    make_config(endpointing_ms=0).validate_config()


def test_gc_freeze_defaults_on():
    assert make_config().params["gc_freeze"] is True
    assert make_config(gc_freeze=False).params["gc_freeze"] is False


def test_transport_copies_are_byte_identical():
    """kernel r3 F6: transport.py is copied verbatim into both addons; drift fails here."""
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[1]
    other = here.parent / "nyo_tts_python" / "transport.py"
    if not other.exists():
        pytest.skip("nyo_tts_python not next to nyo_asr_python")
    assert (here / "transport.py").read_bytes() == other.read_bytes()


def _init(ext, params, monkeypatch):
    """Run NYOASRExtension.on_init with the base class' on_init stubbed and gc.freeze recorded."""
    import asyncio
    import gc
    from unittest.mock import AsyncMock, MagicMock
    from ten_ai_base.asr import AsyncASRBaseExtension
    calls = []
    monkeypatch.setattr(gc, "freeze", lambda: calls.append("freeze"))
    monkeypatch.setattr(AsyncASRBaseExtension, "on_init", AsyncMock())
    env = MagicMock()
    env.get_property_to_json = AsyncMock(return_value=(json.dumps({"params": params}), None))
    ext.ten_env = env
    asyncio.run(ext.on_init(env))
    return env, calls


def _warns(env):
    return " | ".join(str(c.args[0]) for c in env.log_warn.call_args_list)


def test_gc_freeze_is_applied_and_logged_by_default(monkeypatch):
    from .helpers import make_extension
    env, calls = _init(make_extension(), {"api_key": "k", "base_url": "ws://127.0.0.1:9"}, monkeypatch)
    assert calls == ["freeze"]
    assert any("gc.freeze" in str(c.args[0]) for c in env.log_info.call_args_list)


def test_gc_freeze_false_leaves_the_host_heap_alone(monkeypatch):
    from .helpers import make_extension
    env, calls = _init(make_extension(), {"api_key": "k", "base_url": "ws://127.0.0.1:9", "gc_freeze": False},
                       monkeypatch)
    assert calls == []


def test_hotwords_set_warns_that_they_are_ignored(monkeypatch):
    from .helpers import make_extension
    env, _ = _init(make_extension(), {"api_key": "k", "base_url": "ws://127.0.0.1:9", "hotwords": ["Marikina"]},
                   monkeypatch)
    assert "hotwords" in _warns(env)
    env, _ = _init(make_extension(), {"api_key": "k", "base_url": "ws://127.0.0.1:9"}, monkeypatch)
    assert "hotwords" not in _warns(env)


def test_missing_private_connecting_hooks_warn_once_and_skip(monkeypatch):
    """ten r3 F5: a ten_ai_base without the private hooks is reported at init, not silently skipped."""
    import asyncio
    from unittest.mock import AsyncMock
    from .helpers import make_extension
    ext = make_extension()
    ext._connection_machine = object()                           # no try_connecting
    env, _ = _init(ext, {"api_key": "k", "base_url": "ws://127.0.0.1:9"}, monkeypatch)
    assert "_connection_machine" in _warns(env) and ext._can_announce is False
    ext._emit_connection_transition = AsyncMock()
    asyncio.run(ext._announce_connecting())
    ext._emit_connection_transition.assert_not_awaited()
    ok = make_extension()
    env, _ = _init(ok, {"api_key": "k", "base_url": "ws://127.0.0.1:9"}, monkeypatch)
    assert ok._can_announce is True and "_connection_machine" not in _warns(env)

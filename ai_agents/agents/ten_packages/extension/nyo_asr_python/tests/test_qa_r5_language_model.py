"""QA r5 fix 3: only params.language set. serve/gateway.py models() lists halohalo/asr-<lang> for every configured
language whenever the ASR engine is up, so with a halohalo/ ASR id listed the absence of ours is conclusive:
use a listed ASR model that declares the language, else FATAL model_not_found at once. No ASR model listed at all
stays inconclusive."""
from nyo_asr_python import transport

from .helpers import make_extension, run

API = [
    {"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"},
    {"id": "fw/flipvox-whisper-base-fil", "kind": "asr", "languages": ["fil"], "default": True},
    {"id": "fw/whisper-large-v3-pld-fil-norm", "kind": "asr", "languages": ["fil"], "default": False},
    {"id": "fw/whisper-large-v3-pld-hil-norm", "kind": "asr", "languages": ["hil"], "default": False},
    {"id": "halohalo/tts-fil", "kind": "tts", "language": "fil"},
    {"id": "halohalo/tts-hil", "kind": "tts", "language": "hil"},
]


def _verdict(monkeypatch, entries, **params):
    async def fake(*a, **k):
        return entries
    monkeypatch.setattr(transport, "fetch_models", fake)
    ext = make_extension(**params)
    return ext, run(ext._models_verdict())


def test_r5_hil_only_uses_the_listed_hil_model(monkeypatch):
    ext, v = _verdict(monkeypatch, API, language="hil")
    assert v[0] == "ok", v
    assert (ext.config.params["model"], ext.config.params["language"]) == ("fw/whisper-large-v3-pld-hil-norm", "hil")


def test_r5_language_nobody_serves_is_fatal_at_once_naming_the_asr_ids(monkeypatch):
    ext, v = _verdict(monkeypatch, API, language="ceb")
    assert v[:2] == ("bad", "model_not_found"), v
    assert "halohalo/asr-fil" in v[2] and "fw/whisper-large-v3-pld-hil-norm" in v[2] and "tts" not in v[2]


def test_r5_default_model_preferred_among_candidates(monkeypatch):
    entries = [{"id": "halohalo/asr-hil", "kind": "asr", "language": "hil"},
               {"id": "fw/a", "kind": "asr", "languages": ["en", "fil"], "default": False},
               {"id": "fw/b", "kind": "asr", "languages": ["en", "fil"], "default": True}]
    ext, v = _verdict(monkeypatch, entries, language="en-US")
    assert v[0] == "ok" and ext.config.params["model"] == "fw/b", (v, ext.config.params["model"])


def test_r5_no_asr_model_listed_stays_inconclusive(monkeypatch):
    entries = [e for e in API if e["kind"] == "tts"]
    ext, v = _verdict(monkeypatch, entries, language="hil")
    assert v[0] == "inconclusive" and ext.config.params["model"] == "halohalo/asr-hil", v


def test_r5_explicit_model_is_never_replaced(monkeypatch):
    ext, v = _verdict(monkeypatch, API, model="halohalo/asr-hil", language="hil")
    assert v[0] == "missing" and ext.config.params["model"] == "halohalo/asr-hil", v


def test_r5_listed_halohalo_language_model_is_kept(monkeypatch):
    ext, v = _verdict(monkeypatch, API, language="fil")
    assert v[0] == "ok" and ext.config.params["model"] == "halohalo/asr-fil", v

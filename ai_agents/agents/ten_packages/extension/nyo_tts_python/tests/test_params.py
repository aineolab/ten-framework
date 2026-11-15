import json
import pathlib

import pytest

from nyo_tts_python.config import NYOTTSConfig

from .helpers import make_config

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_defaults():
    cfg = NYOTTSConfig(params={"api_key": "k", "base_url": "wss://nyo.example.test"})
    cfg.update_params()
    cfg.validate_config()
    assert cfg.base_url == "wss://nyo.example.test"
    assert cfg.tls_verify is True and cfg.allow_insecure is False
    assert cfg.sample_rate == 24000
    assert cfg.language == "fil" or cfg.language == ""  # model default halohalo/tts-fil


def test_params_are_hoisted_and_consumed():
    cfg = make_config(model="halohalo/tts-ceb", voice="maria", sample_rate=16000)
    assert (cfg.model, cfg.voice, cfg.sample_rate) == ("halohalo/tts-ceb", "maria", 16000)
    assert cfg.language == "ceb"
    assert "model" not in cfg.params


@pytest.mark.parametrize("url", [None, "", "   "])
def test_missing_base_url_rejected(url):
    """No default host: without base_url nothing may connect anywhere."""
    params = {"api_key": "k"} if url is None else {"api_key": "k", "base_url": url}
    cfg = NYOTTSConfig(params=params)
    cfg.update_params()
    assert cfg.base_url.strip() == ""
    with pytest.raises(ValueError, match="base_url is required"):
        cfg.validate_config()


def test_missing_api_key_rejected():
    cfg = NYOTTSConfig(params={"api_key": "", "base_url": "wss://nyo.example.test"})
    cfg.update_params()
    with pytest.raises(ValueError, match="API key"):
        cfg.validate_config()


@pytest.mark.parametrize("url", ["http://x", "host:9100"])
def test_bad_url_rejected(url):
    with pytest.raises(ValueError, match="base_url"):
        make_config(base_url=url)


def test_bad_model_rejected():
    with pytest.raises(ValueError, match="model"):
        make_config(model="orpheus")


def test_bad_sample_rate_rejected():
    with pytest.raises(ValueError, match="sample_rate"):
        make_config(sample_rate=12345)


def test_api_key_masked_in_to_str():
    cfg = make_config(api_key="super-secret-key")
    assert "super-secret-key" not in cfg.to_str()
    assert "super-secret-key" in cfg.to_str(sensitive_handling=False)


def test_property_and_manifest_consistent():
    prop = json.loads((ROOT / "property.json").read_text())
    man = json.loads((ROOT / "manifest.json").read_text())
    declared = man["api"]["property"]["properties"]["params"]["properties"]
    assert set(prop["params"]) <= set(declared)
    assert prop["params"]["api_key"] == "${env:NYO_API_KEY|}"
    assert man["api"]["interface"][0]["import_uri"].endswith("tts-interface.json")
    assert {d["name"]: d["version"] for d in man["dependencies"]} == {
        "ten_runtime_python": "0.11", "ten_ai_base": "0.7"}

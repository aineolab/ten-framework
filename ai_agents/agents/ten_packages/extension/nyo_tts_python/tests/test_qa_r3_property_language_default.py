"""QA round 3: property.json ships params.language="fil". TEN deep-merges a graph node's property into the
extension's property.json (core/src/ten_utils/value/value_merge.c: objects merge recursively), so a graph that sets
only params.model (as the README snippet does with halohalo/tts-ceb) keeps language="fil", and the gateway's
route() lets ?language= override the model suffix: every non-fil model is synthesised by the Filipino engine.
Fixed: property.json ships language "". The second test (optional) checks the effective hil config on the real stack at NYO_QA_REAL (default ws://127.0.0.1:9150)."""
import json
import os
import socket
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from nyo_tts_python.config import NYOTTSConfig
from nyo_tts_python.tts_client import NYOTTSClient

from .helpers import make_extension, run, text_input, wait_for

EXT_DIR = Path(__file__).resolve().parents[1]


def _deep_merge(dst, src):  # what ten_value_object_merge_with_clone does
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _effective(graph_property: dict) -> NYOTTSConfig:
    prop = json.loads((EXT_DIR / "property.json").read_text())
    merged = _deep_merge(prop, graph_property)
    merged["params"]["api_key"] = "k"
    merged["params"]["base_url"] = "wss://api.nyolab.ai"
    cfg = NYOTTSConfig.model_validate(merged)
    cfg.update_params()
    cfg.validate_config()
    return cfg


@pytest.mark.parametrize("model", ["halohalo/tts-hil", "halohalo/tts-ceb"])
def test_qa_r3_graph_model_without_language_routes_to_model_language(model):
    cfg = _effective({"params": {"api_key": "x", "base_url": "wss://api.nyolab.ai", "model": model}})
    q = parse_qs(urlparse(NYOTTSClient(cfg, None).build_url()).query)
    want = model.rsplit("-", 1)[1]
    assert cfg.language == want, (
        f"graph sets model={model} only; effective language={cfg.language!r} (from property.json), "
        f"so the gateway is asked for {q}")


REAL = os.environ.get("NYO_QA_REAL", "ws://127.0.0.1:9150")
KEY = Path(os.environ.get("NYO_QA_REAL_KEY_FILE", "/nonexistent/nyo-qa-key"))


def _up() -> bool:
    u = urlparse(REAL)
    try:
        socket.create_connection((u.hostname, u.port), 0.3).close()
        return KEY.exists()
    except OSError:
        return False


@pytest.mark.skipif(not _up(), reason="real stack not running")
def test_qa_r3_real_stack_hil_graph_gets_hil_engine():
    async def go():
        # the language a graph that sets model only gets after the deep merge with property.json
        lang = _effective({"params": {"model": "halohalo/tts-hil"}}).language
        ext, rec = make_extension(REAL, api_key=KEY.read_text().strip(), model="halohalo/tts-hil", language=lang)
        try:
            await ext.request_tts(text_input("r", "Maayong aga.", end=True))
            await wait_for(lambda: rec.of("audio_end"), timeout=40, what="audio_end")
            assert ext._sample_rate == 16000, (
                f"model halohalo/tts-hil + language {lang!r} was served at {ext._sample_rate} Hz: the fil (Orpheus) engine, "
                "not the Hiligaynon one")
        finally:
            ext._stopped = True
            ext._cancel_reconnect_task()
            if ext.client:
                await ext.client.stop()

    run(go())

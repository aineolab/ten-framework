"""QA r3: TEN deep-merges a node's params into property.json; property.json leaves model and language empty."""
import asyncio
import json
from pathlib import Path

import pytest

from nyo_asr_python.config import NYOASRConfig

from .helpers import run
from .test_protocol import session

PROP = json.loads((Path(__file__).resolve().parents[1] / "property.json").read_text())
HIL_FW = "fw/whisper-large-v3-pld-hil-norm"
LISTING = [
    {"id": "halohalo/asr-fil", "kind": "asr", "language": "fil"},
    {"id": "halohalo/asr-hil", "kind": "asr", "language": "hil"},
    {"id": HIL_FW, "kind": "asr", "languages": ["hil"]},
    {"id": "fw/whisper-base", "kind": "asr", "languages": ["en", "fil"]},
]


def merged(node_params):
    params = {**PROP["params"], "api_key": "k", "base_url": "wss://x.test", **node_params}
    cfg = NYOASRConfig(params=params)
    cfg.apply_defaults()
    return cfg.params


@pytest.mark.parametrize("node, model, language", [
    ({}, "halohalo/asr-fil", "fil"),
    ({"model": "halohalo/asr-hil"}, "halohalo/asr-hil", "hil"),
    ({"language": "hil"}, "halohalo/asr-hil", "hil"),
    ({"model": HIL_FW}, HIL_FW, ""),                     # resolved from GET /v1/models at preflight
    ({"model": HIL_FW, "language": "hil"}, HIL_FW, "hil"),
])
def test_resolution_after_the_merge(node, model, language):
    p = merged(node)
    assert (p["model"], p["language"]) == (model, language)


@pytest.mark.parametrize("node, language", [({"model": "halohalo/asr-hil"}, "hil"), ({"language": "hil"}, "hil"),
                                            ({"model": HIL_FW}, "hil")])
def test_hil_node_connects_with_language_hil(node, language):
    async def go():
        p = merged(node)
        async with session(model=p["model"], language=p["language"]) as (srv, ext):
            srv.models = LISTING
            await ext.start_connection()
            assert not ext._fatal and ext.send_asr_error.await_count == 0 and ext.is_connected()
            assert srv.connections[0]["query"]["language"] == language

    run(go())


def test_model_without_a_single_listed_language_omits_language():
    """fw/whisper-base lists en and fil: no guess; the query and config carry no language (gateway default)."""
    async def go():
        p = merged({"model": "fw/whisper-base"})
        async with session(model=p["model"], language=p["language"]) as (srv, ext):
            srv.models = LISTING
            await ext.start_connection()
            assert ext.is_connected() and "language" not in srv.connections[0]["query"]
            await asyncio.sleep(0.05)
            assert "language" not in srv.received("config")[0]

    run(go())


def test_inconclusive_listing_omits_language_instead_of_sending_fil():
    async def go():
        p = merged({"model": HIL_FW})
        async with session(model=p["model"], language=p["language"]) as (srv, ext):
            srv.models = [{"id": "nvidia/nemotron-3.5-transcribe", "kind": "asr"}]
            verdict = await ext._models_verdict()
            assert verdict[0] == "inconclusive" and ext.config.params["language"] == ""
            from nyo_asr_python.recognition import NYOASRRecognition
            url = NYOASRRecognition("k", None, None, ext.config.params, None).build_url()
            assert "language=" not in url

    run(go())

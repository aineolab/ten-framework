"""QA r4: params.interim (README alias of interim_results) after TEN's deep merge with property.json."""
import json
from pathlib import Path

from nyo_asr_python.config import NYOASRConfig


def test_r4_graph_interim_false_survives_property_json_merge():
    base = json.loads((Path(__file__).resolve().parents[1] / "property.json").read_text())
    base["params"].update({"api_key": "k", "base_url": "wss://api.nyolab.ai", "interim": False})  # graph node
    cfg = NYOASRConfig.model_validate(base)
    cfg.apply_defaults()
    assert cfg.params["interim_results"] is False, "graph set interim=false; property.json's interim_results wins"

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field
from ten_ai_base import utils

from . import transport
from .const import (
    DEFAULT_BASE_URL,
    DEFAULT_RECONNECT_WINDOW_MS,
    DEFAULT_REQUEST_TIMEOUT_MS,
    DEFAULT_BOUNDARY_HINT_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEGMENT_IDLE_MS,
    DEFAULT_SEGMENT_MAX_MS,
    VALID_SAMPLE_RATES,
)


def language_from_model(model: str) -> str:
    """The language the gateway routes ``halohalo/tts-<lang>`` to when no ``language`` is given; else ""."""
    prefix = "halohalo/tts-"
    return model[len(prefix):] if isinstance(model, str) and model.startswith(prefix) else ""


_PARAM_FIELDS = (
    "api_key",
    "base_url",
    "allow_insecure",
    "tls_verify",
    "model",
    "voice",
    "language",
    "sample_rate",
    "request_timeout_ms",
    "reconnect_window_ms",
    "segment_idle_ms",
    "segment_max_ms",
    "boundary_hint_ms",
    "enable_words",
    "gc_freeze",
)


class NYOTTSConfig(BaseModel):
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    allow_insecure: bool = False
    tls_verify: bool = True
    model: str = "halohalo/tts-fil"
    voice: str = ""
    language: str = ""
    # Fallback only; the server's audio_start is authoritative.
    sample_rate: int = DEFAULT_SAMPLE_RATE
    request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS
    # FATAL connect_failed only once the gateway has been unreachable for longer than this.
    reconnect_window_ms: int = DEFAULT_RECONNECT_WINDOW_MS
    # no new chunk for this long and no text_input_end: end:true for the open NRSP request (0 = never)
    segment_idle_ms: int = DEFAULT_SEGMENT_IDLE_MS
    # a segment open this long ends at the next sentence-final chunk (0 = never)
    segment_max_ms: int = DEFAULT_SEGMENT_MAX_MS
    # a sentence-final chunk not followed by another within this long gets one " " increment (0 = never)
    boundary_hint_ms: int = DEFAULT_BOUNDARY_HINT_MS
    enable_words: bool = False
    # Opt-in: gc.freeze() is process-wide (it affects every extension in the host).
    gc_freeze: bool = False

    dump: bool = False
    dump_path: str = "/tmp"
    params: dict[str, Any] = Field(default_factory=dict)

    def update_params(self) -> None:
        params = self.params if isinstance(self.params, dict) else {}
        self.params = params
        for name in _PARAM_FIELDS:
            if name in params:
                setattr(self, name, params.pop(name))
        for name in ("voice", "language"):
            if getattr(self, name) is None:
                setattr(self, name, "")
        if not self.language:
            self.language = language_from_model(self.model)

    def validate_config(self) -> None:
        for name in ("model", "voice", "language"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not str(self.base_url or "").strip():
            raise ValueError("base_url is required")
        if not str(self.api_key or "").strip():
            raise ValueError("NYO API key is required (set NYO_API_KEY)")
        transport.check_base_url(self.base_url, self.allow_insecure)
        if "/" not in self.model:
            raise ValueError(f"model must look like 'vendor/name': {self.model!r}")
        if self.sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample_rate: {self.sample_rate}")
        for name in ("request_timeout_ms", "reconnect_window_ms"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("segment_idle_ms", "segment_max_ms", "boundary_hint_ms"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative integer (0 = off)")

    def to_str(self, sensitive_handling: bool = True) -> str:
        if not sensitive_handling:
            return f"{self}"
        config = copy.deepcopy(self)
        if config.api_key:
            config.api_key = utils.encrypt(config.api_key)
        return f"{config}"

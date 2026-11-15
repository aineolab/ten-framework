from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field
from ten_ai_base import utils

from . import transport
from .const import (
    DEFAULT_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT_MS,
    DEFAULT_SAMPLE_RATE,
    VALID_SAMPLE_RATES,
)


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
    enable_words: bool = False
    gc_freeze: bool = True

    dump: bool = False
    dump_path: str = "/tmp"
    params: dict[str, Any] = Field(default_factory=dict)

    def update_params(self) -> None:
        params = self.params if isinstance(self.params, dict) else {}
        self.params = params
        for name in _PARAM_FIELDS:
            if name in params:
                setattr(self, name, params.pop(name))
        if not self.language and self.model.startswith("halohalo/tts-"):
            self.language = self.model.rsplit("-", 1)[-1]

    def validate_config(self) -> None:
        if not str(self.base_url or "").strip():
            raise ValueError("base_url is required")
        if not str(self.api_key or "").strip():
            raise ValueError("NYO API key is required (set NYO_API_KEY)")
        transport.check_base_url(self.base_url, self.allow_insecure)
        if "/" not in self.model:
            raise ValueError(f"model must look like 'vendor/name': {self.model!r}")
        if self.sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample_rate: {self.sample_rate}")
        if self.request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be > 0")

    def to_str(self, sensitive_handling: bool = True) -> str:
        if not sensitive_handling:
            return f"{self}"
        config = copy.deepcopy(self)
        if config.api_key:
            config.api_key = utils.encrypt(config.api_key)
        return f"{config}"

import json
from typing import Any

from pydantic import BaseModel, Field
from ten_ai_base.utils import encrypt

from .const import (
    DEFAULT_BASE_URL,
    DEFAULT_BUFFER_MS,
    DEFAULT_ENDPOINTING_MS,
    DEFAULT_SAMPLE_RATE,
)
from .transport import check_base_url

_SENSITIVE_SUBSTRINGS = ("key", "token", "secret", "signature", "password")

# halohalo self-hosted languages (plan section 2) plus Philippine English.
PH_LANGUAGES = ("bcl", "ceb", "fil", "hil", "ilo", "pam", "pag", "tsg", "war")

NYO_DEFAULT_PARAMS: dict[str, Any] = {
    "base_url": DEFAULT_BASE_URL,
    "language": "fil",
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "interim_results": True,
    "endpointing_ms": DEFAULT_ENDPOINTING_MS,
    "allow_insecure": False,
    "tls_verify": True,
    "buffer_ms": DEFAULT_BUFFER_MS,
    "gc_freeze": True,
}

# TEN-facing BCP-47 tags returned in asr_result.language.
_LANGUAGE_MAP = {
    "zh": "zh-CN",
    "en": "en-US",
    "eng": "en-PH",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "de": "de-DE",
    "fr": "fr-FR",
    "ru": "ru-RU",
    "es": "es-ES",
    "pt": "pt-PT",
    "it": "it-IT",
    "hi": "hi-IN",
    "ar": "ar-AE",
    **{code: f"{code}-PH" for code in PH_LANGUAGES},
}


def _is_sensitive_key(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _SENSITIVE_SUBSTRINGS)


def normalize_language(code: str | None) -> str:
    code = (code or "").strip()
    return _LANGUAGE_MAP.get(code, code)


class NYOASRConfig(BaseModel):
    dump: bool = False
    dump_path: str = "/tmp"
    finalize_timeout_ms: int = 2000
    params: dict[str, Any] = Field(default_factory=dict)

    def apply_defaults(self) -> None:
        params = self.params if isinstance(self.params, dict) else {}
        # `interim` is the interface-spec alias of `interim_results`.
        if "interim" in params and "interim_results" not in params:
            params["interim_results"] = params["interim"]
        for key, value in NYO_DEFAULT_PARAMS.items():
            params.setdefault(key, value)
        if not params.get("model"):
            lang = str(params.get("language") or "")
            params["model"] = (
                f"halohalo/asr-{lang}" if lang in PH_LANGUAGES else "nyo/auto"
            )
        self.params = params

    def validate_config(self) -> None:
        params = self.params
        if not str(params.get("base_url") or "").strip():
            raise ValueError("base_url is required")
        if not str(params.get("api_key") or "").strip():
            raise ValueError("NYO API key is required (set NYO_API_KEY)")
        # Raises TransportError, which carries a machine-readable .code
        # (e.g. "insecure_url") that on_init forwards to vendor_info.
        check_base_url(
            str(params.get("base_url") or ""),
            bool(params.get("allow_insecure", False)),
        )
        if params.get("sample_rate") not in {8000, 16000, 24000, 48000}:
            raise ValueError(
                f"Unsupported sample_rate: {params.get('sample_rate')}"
            )
        endpointing_ms = params.get("endpointing_ms", 0)
        if type(endpointing_ms) is not int or endpointing_ms < 0:
            raise ValueError("endpointing_ms must be an int >= 0")
        if self.finalize_timeout_ms <= 0:
            raise ValueError("finalize_timeout_ms must be > 0")
        if int(params.get("buffer_ms", 0)) < 0:
            raise ValueError("buffer_ms must be >= 0")

    def to_json(self, sensitive_handling: bool = False) -> str:
        config_dict = self.model_dump()
        if sensitive_handling and config_dict["params"]:
            for key, value in list(config_dict["params"].items()):
                if value and _is_sensitive_key(key):
                    config_dict["params"][key] = encrypt(str(value))
        return json.dumps(config_dict)

    @property
    def normalized_language(self) -> str:
        return normalize_language((self.params or {}).get("language", ""))

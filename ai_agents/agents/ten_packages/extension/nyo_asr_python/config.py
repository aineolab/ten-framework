import json
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr
from ten_ai_base.utils import encrypt

from .const import (
    DEFAULT_BASE_URL,
    DEFAULT_BUFFER_MS,
    DEFAULT_ENDPOINTING_MS,
    DEFAULT_FINALIZE_TIMEOUT_MS,
    DEFAULT_RECONNECT_WINDOW_MS,
    DEFAULT_SAMPLE_RATE,
    NEMOTRON_FINALIZE_TIMEOUT_MS,
    NEMOTRON_MODEL,
)
from .transport import check_base_url

_SENSITIVE_SUBSTRINGS = ("key", "token", "secret", "signature", "password")

# Philippine languages with a TEN BCP-47 tag (`<code>-PH`).
PH_LANGUAGES = ("bcl", "ceb", "fil", "hil", "ilo", "pam", "pag", "tsg", "war")
# Language aliases accepted in params.language (matched case-insensitively, after BCP-47 tags are cut to their
# primary subtag: fil-PH -> fil, hil-PH -> hil, en-US -> en). `en-PH` is Philippine English (`eng`).
LANGUAGE_ALIASES = {"tl": "fil", "tgl": "fil", "fil": "fil", "en-ph": "eng", "en_ph": "eng"}
# PCM16 rates /v1/asr accepts (serve/gateway.py ASR_RATES); others are refused at the handshake.
ASR_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

NYO_DEFAULT_PARAMS: dict[str, Any] = {
    "base_url": DEFAULT_BASE_URL,
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "interim_results": True,
    "endpointing_ms": DEFAULT_ENDPOINTING_MS,
    "allow_insecure": False,
    "tls_verify": True,
    "buffer_ms": DEFAULT_BUFFER_MS,
    "reconnect_window_ms": DEFAULT_RECONNECT_WINDOW_MS,
    "gc_freeze": False,
}

# Both `model` and `language` empty (property.json ships them empty, so a graph node that sets one of them is not
# combined with a default for the other by TEN's property deep-merge).
DEFAULT_MODEL = "halohalo/asr-fil"
DEFAULT_LANGUAGE = "fil"
HALOHALO_ASR_PREFIX = "halohalo/asr-"

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


def canonical_language(code: str | None) -> str:
    """params.language as the gateway expects it: lowercase ISO code, aliases resolved (fil-PH / tl -> fil)."""
    c = str(code or "").strip().lower()
    if not c:
        return ""
    if c in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[c]
    primary = c.replace("_", "-").split("-", 1)[0]
    return LANGUAGE_ALIASES.get(primary, primary)


def normalize_language(code: str | None) -> str:
    code = (code or "").strip()
    return _LANGUAGE_MAP.get(code, code)


class NYOASRConfig(BaseModel):
    dump: bool = False
    dump_path: str = "/tmp"
    finalize_timeout_ms: int = DEFAULT_FINALIZE_TIMEOUT_MS
    params: dict[str, Any] = Field(default_factory=dict)
    # True when params.model was derived from params.language (halohalo/asr-<lang>): the preflight may then
    # replace it with a listed ASR model that declares that language (extension._models_verdict).
    _model_from_language: bool = PrivateAttr(default=False)
    # Whether the user set finalize_timeout_ms, recorded before apply_defaults assigns it (pydantic then counts the
    # field as set), so the preflight's Nemotron default (extension._models_verdict) still applies.
    _finalize_user_set: bool | None = PrivateAttr(default=None)

    @property
    def finalize_timeout_user_set(self) -> bool:
        if self._finalize_user_set is None:
            return "finalize_timeout_ms" in self.model_fields_set
        return self._finalize_user_set

    def apply_defaults(self) -> None:
        params = self.params if isinstance(self.params, dict) else {}
        # `interim` is the interface-spec alias of `interim_results`.
        if "interim" in params and "interim_results" not in params:
            params["interim_results"] = params["interim"]
        for key, value in NYO_DEFAULT_PARAMS.items():
            params.setdefault(key, value)
        # model / language resolution (each may be empty):
        #   both empty    -> halohalo/asr-fil + fil
        #   only language -> halohalo/asr-<lang> (GET /v1/models then says whether the gateway serves it)
        #   only model    -> the halohalo/asr-<lang> suffix, else the model's single language in GET /v1/models
        #                    (extension preflight); still unknown -> omitted, the gateway uses the model's own
        model = str(params.get("model") or "").strip()
        lang = canonical_language(params.get("language"))
        if not model and not lang:
            model, lang = DEFAULT_MODEL, DEFAULT_LANGUAGE
        elif not model:
            # a language the gateway does not serve is reported by the preflight as model_not_found, naming the
            # served ASR ids (halohalo/asr-* is the namespace GET /v1/models lists for every served language)
            model = f"{HALOHALO_ASR_PREFIX}{lang}"
            self._model_from_language = True
        elif not lang and model.startswith(HALOHALO_ASR_PREFIX):
            lang = model[len(HALOHALO_ASR_PREFIX):]
        params["model"], params["language"] = model, lang
        self.params = params
        # Unless the user set finalize_timeout_ms, wait longer than the gateway's own finalize timer so its
        # finalize_end (and the real final) comes first: 3000 ms for streaming models, 30 s for a Nemotron
        # final (a batch decode).
        if self._finalize_user_set is None:
            self._finalize_user_set = "finalize_timeout_ms" in self.model_fields_set
        if not self._finalize_user_set:
            self.finalize_timeout_ms = (NEMOTRON_FINALIZE_TIMEOUT_MS if params.get("model") == NEMOTRON_MODEL
                                        else DEFAULT_FINALIZE_TIMEOUT_MS)

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
        sample_rate = params.get("sample_rate")
        if type(sample_rate) is not int or sample_rate not in ASR_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample_rate: {sample_rate!r} "
                f"(one of {', '.join(map(str, ASR_SAMPLE_RATES))})"
            )
        endpointing_ms = params.get("endpointing_ms", 0)
        if type(endpointing_ms) is not int or endpointing_ms < 0:
            raise ValueError("endpointing_ms must be an int >= 0")
        if self.finalize_timeout_ms <= 0:
            raise ValueError("finalize_timeout_ms must be > 0")
        if int(params.get("buffer_ms", 0)) < 0:
            raise ValueError("buffer_ms must be >= 0")
        window = params.get("reconnect_window_ms")
        if type(window) is not int or window < 0:
            raise ValueError("reconnect_window_ms must be an int >= 0")

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

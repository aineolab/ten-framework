"""Transport helpers shared by the nyo ASR/TTS Python addons.

This module is copied verbatim between ten/nyo_asr_python and
ten/nyo_tts_python, so it must only depend on the standard library and
must not import from the TEN runtime or from the enclosing package.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class TransportError(ValueError):
    """Transport/config failure with a stable machine-readable code (a ValueError, so config validation raises it)."""

    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


def is_loopback_host(host: str | None) -> bool:
    """True for "localhost", any 127.0.0.0/8 address, or "::1"."""
    if not host:
        return False
    h = host.strip()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    if h.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def check_base_url(url: str, allow_insecure: bool) -> None:
    """Validate a ws(s):// base URL, refusing plaintext to remote hosts."""
    parts = urlsplit(url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise TransportError(
            "invalid_url",
            f"invalid base_url {url!r}: expected ws:// or wss:// with a host",
        )
    if parts.scheme == "ws" and not allow_insecure:
        host = parts.hostname
        if not is_loopback_host(host):
            raise TransportError(
                "insecure_url",
                f"refusing base_url ws:// to non-local host {host!r}: "
                "use wss:// or set params.allow_insecure=true",
            )


def ssl_context(url: str, tls_verify: bool) -> ssl.SSLContext | None:
    """Return an SSL context for wss:// URLs, None for ws://."""
    if urlsplit(url).scheme != "wss":
        return None
    ctx = ssl.create_default_context()
    if not tls_verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def http_base(url: str) -> str:
    """Map ws:// -> http:// and wss:// -> https://, stripping trailing '/'."""
    parts = urlsplit(url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return f"{scheme}://{parts.netloc}{parts.path}".rstrip("/")


def _find_tls_verify_error(e: BaseException) -> BaseException | None:
    cur: BaseException | None = e
    for _ in range(6):
        if cur is None:
            return None
        if isinstance(cur, ssl.SSLCertVerificationError):
            return cur
        if isinstance(cur, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(cur):
            return cur
        reason = getattr(cur, "reason", None)  # urllib.error.URLError wraps the SSL error here
        if isinstance(reason, BaseException) and reason is not cur:
            cur = reason
            continue
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return None


def is_tls_verify_error(e: BaseException) -> bool:
    """True if e or its cause/context chain is a TLS verification failure."""
    return _find_tls_verify_error(e) is not None


async def fetch_models(
    base_url: str, api_key: str, tls_verify: bool, timeout: float = 5.0
) -> list[dict]:
    """GET {base}/v1/models and return the model entries (dicts with at least "id").

    The API key is only sent in the Authorization header and is never
    included in error messages or logs.
    """
    url = f"{http_base(base_url)}/v1/models"
    context = ssl_context(base_url, tls_verify)

    def _get() -> dict:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        body = await asyncio.to_thread(_get)
    except urllib.error.HTTPError as e:
        status = e.code
        if status in (401, 403):
            raise TransportError(
                "unauthorized",
                f"GET /v1/models -> {status}: invalid or missing API key",
                status,
            ) from e
        raise TransportError(
            "models_http_error", f"GET /v1/models -> {status}", status
        ) from e
    except (urllib.error.URLError, OSError) as e:
        tls_err = _find_tls_verify_error(e)
        if tls_err is not None:
            host = urlsplit(base_url).hostname
            reason = getattr(tls_err, "verify_message", None) or str(tls_err)
            raise TransportError(
                "tls_error",
                f"TLS certificate verification failed for {host}: {reason}; "
                "fix the certificate or set params.tls_verify=false",
            ) from e
        raise TransportError("models_unreachable", str(e)) from e
    except ValueError as e:  # not JSON: some other service on that port
        raise TransportError("models_http_error", f"GET /v1/models: invalid JSON ({e})") from e

    data = body.get("data", []) if isinstance(body, dict) else []
    return [m for m in data if isinstance(m, dict) and "id" in m]


def check_model(entries: list[dict], model: str, kind: str, language: str | None,
                base_url: str) -> tuple[str, str] | None:
    """Match the configured model against GET /v1/models entries. None if usable, else (code, message).

    Checks the id, the entry's `kind` (asr/tts, when listed) and, when the entry lists `languages`, the
    configured language. The message names the ids of the right kind, so a user can pick one.
    """
    where = f"{http_base(base_url)}/v1/models"
    same_kind = sorted(e["id"] for e in entries if e.get("kind") in (None, kind))
    entry = next((e for e in entries if e["id"] == model), None)
    if entry is None or entry.get("kind") not in (None, kind):
        what = "is not served" if entry is None else f"is a {entry.get('kind')} model, not {kind}"
        return "model_not_found", f"{kind} model {model!r} {what} by {where}; {kind} models: {', '.join(same_kind)}"
    langs = entry.get("languages")
    if language and isinstance(langs, list) and langs and language not in langs:
        return ("language_not_supported",
                f"model {model!r} does not serve language {language!r} (it lists {', '.join(map(str, langs))})")
    return None

"""Shared builders for the NYO ASR unit tests (no TEN runtime app needed)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from nyo_asr_python.config import NYOASRConfig
from nyo_asr_python.extension import NYOASRExtension


def make_config(**params) -> NYOASRConfig:
    cfg = NYOASRConfig(
        finalize_timeout_ms=params.pop("finalize_timeout_ms", 50),
        params={"api_key": "nyo-test-key", "base_url": "wss://nyo.example.test", **params},
    )
    cfg.apply_defaults()
    return cfg


def make_extension(**params) -> NYOASRExtension:
    ext = NYOASRExtension("nyo_asr_python")
    ext.ten_env = MagicMock()
    ext.config = make_config(**params)
    ext.send_asr_result = AsyncMock()
    ext.send_asr_error = AsyncMock()
    ext.send_asr_finalize_end = AsyncMock()
    ext.send_connect_delay_metrics = AsyncMock()
    ext.send_vendor_metrics = AsyncMock()
    ext.on_connected = AsyncMock()
    ext.on_disconnected = AsyncMock()
    return ext


def run(coro):
    return asyncio.run(coro)

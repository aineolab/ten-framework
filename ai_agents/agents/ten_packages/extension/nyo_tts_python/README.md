# nyo_tts_python

TEN TTS extension for the NYO speech plane (NRSP v1), built on `AsyncTTS2BaseExtension`
(ten_ai_base 0.7) in WebSocket mode, modelled on `deepgram_tts`. One socket carries many
sequential requests; text increments stream up, `[u32 request_seq][u32 audio_offset_ms][PCM16]`
frames stream down. Full guide: `docs/ten.md` in nyo-serve.

## What is different from the other WS TTS addons

* **Cancel without reconnect.** `tts_flush` -> `cancel_tts()` sends NRSP `flush`, marks the
  request's `request_seq` stale and emits `tts_audio_end(INTERRUPTED)` immediately. Frames that
  were already in flight carry the old seq and are dropped by header check. The socket is never
  recycled (deepgram/xai reconnect, 100-300 ms).
* **Sample rate comes from the server** (`audio_start.sample_rate`): 24000 for Orpheus + SNAC,
  16000 for the SpeechT5 fallback. `params.sample_rate` is only the pre-first-request default.
* **Frame timestamps are the server's `audio_offset_ms`**, so TEN's subtitle/timeline checks see the
  true position; gaps >20 ms are logged and counted.
* A watchdog fails a request that stays silent for `request_timeout_ms` after `text_input_end`.

## Properties

| property | default | notes |
|---|---|---|
| `params.api_key` | `${env:NYO_API_KEY\|}` | Empty or HTTP 401/403 -> FATAL (-1000). |
| `params.base_url` | required, no default (`${env:NYO_BASE_URL\|}`) | addon appends `/v1/tts`. `ws://` only to loopback unless `allow_insecure`. |
| `params.allow_insecure` | `false` | `true` permits `ws://` to a non-local host; otherwise FATAL `insecure_url` at init. |
| `params.tls_verify` | `true` | verify the `wss://` certificate; a bad one is FATAL `tls_error`. |
| `params.model` | `halohalo/tts-fil` | any TTS id the gateway's `GET /v1/models` lists; checked before the first connect (FATAL `model_not_found`). |
| `params.voice` | `""` | forwarded as `?voice=`. Empty = the model's default speaker. See [Speakers](#speakers). |
| `params.language` | from model | forwarded as `?language=`. |
| `params.sample_rate` | `24000` | fallback only: the stream's rate is the model's (`audio_start.sample_rate`); NRSP has no rate request and the addon does not resample. So `tts_guarder` `basic_audio_setting` passes only when its two configs name models with different rates. |
| `params.request_timeout_ms` | `30000` | inactivity limit once `text_input_end` was sent. |
| `params.enable_words` | `false` | NRSP `alignment` -> `tts_text_result` (subtitles). |
| `params.gc_freeze` | `true` | `gc.collect(); gc.freeze()` at the end of `on_init`, logged at info. It removes the gen-2 GC passes over the TEN + pydantic init-time heap from the audio path (measured in a TEN host: max per-frame cost 4067 us -> 125 us, a 4-8 ms stall otherwise, bench/results/ten.json). It is process-wide: it also freezes the init-time heap of every other extension in the same host. `false` leaves the host's GC alone. |
| `dump`, `dump_path` | `false`, `/tmp` | per-request PCM `nyo_dump_<request_id>.pcm`. |

## Speakers

A model may offer several speakers; the gateway lists them in `GET /v1/models` (`voices`, `default_voice`) and in
`GET /v1/audio/voices`. Set `params.voice` to one of those ids, or leave it empty for the default. A voice the
gateway does not list is refused when the socket opens (HTTP 400, `unsupported_voice`), which this addon reports
as a connection error.

The voice belongs to the session: the gateway fixes it when the socket opens and keeps it for every request on
that socket. `update_configs` with a new `voice` therefore opens a fresh socket: right away when idle, otherwise
after the request in progress.

Orpheus Filipino (`halohalo/tts-fil`) offers ten speakers, ids `FIL_0051`, `FIL_0059`, `FIL_0078` (default),
`FIL_0079`, `FIL_0081`, `FIL_0084`, `FIL_0107`, `FIL_0113`, `FIL_0115` and `FIL_0135`. A gateway in front of
another one must also allow the ids (`SAPIN_TTS_VOICES` on older releases).

## Graph snippet (agents/examples/voice-assistant)

```json
{
  "type": "extension",
  "name": "tts",
  "addon": "nyo_tts_python",
  "extension_group": "tts",
  "property": {
    "dump": false,
    "params": {
      "api_key": "${env:NYO_API_KEY}",
      "base_url": "${env:NYO_BASE_URL}",
      "model": "halohalo/tts-ceb",
      "voice": "FIL_0078"
    }
  }
}
```

Add `{"path": "../../../ten_packages/extension/nyo_tts_python"}` to `tenapp/manifest.json`.

## Errors

Exactly one FATAL `tts_error` per extension instance; afterwards the socket is closed, nothing reconnects, and every
request ends with `tts_audio_end(ERROR)` without another error. `vendor_info.vendor` is always `nyo`; an engine's
name goes into the message. `request_event_interval_ms` counts from the request's first text (it includes TTFB).

## Tests

The TEN guarders need `pytest<9` (their `conftest.py` imports `_pytest.config.Notset`, removed in pytest 9).

```bash
cd ten/nyo_tts_python && PYTHONPATH=<runtime lib:interface:ten_ai_base/interface:ten> pytest tests -q
python3 ten/fake_nrsp_server.py --port 9100 &
NYO_API_KEY=fake-key task tts-guarder-test EXTENSION=nyo_tts_python
```

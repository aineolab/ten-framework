# nyo_asr_python

TEN ASR extension for the NYO speech plane (NRSP v1). One WebSocket per session to
`speechd` (or a local `sapinsapin-serve` core) carries 16 kHz PCM16 up and
`result` / `finalize_end` / `error` / `usage` messages down. Vendor and halohalo
model routing happens server-side; TEN sees one addon and one key.

Built on `AsyncASRBaseExtension` (ten_ai_base 0.7), modelled on `xai_asr_python`.
Full guide, message mapping table and known gaps: `docs/ten.md` in nyo-serve.

## Properties

| property | default | notes |
|---|---|---|
| `params.api_key` | `${env:NYO_API_KEY\|}` | Bearer token. Empty -> FATAL (-1000) on init. |
| `params.base_url` | required, no default (`${env:NYO_BASE_URL\|}`) | the addon appends `/v1/asr`. `ws://` only to loopback unless `allow_insecure`. |
| `params.allow_insecure` | `false` | `true` permits `ws://` to a non-local host. Otherwise that is FATAL `insecure_url` at init. |
| `params.tls_verify` | `true` | verify the `wss://` certificate; a bad one is FATAL `tls_error`. |
| `params.model` | `halohalo/asr-fil` (`property.json`) | any ASR id the gateway's `GET /v1/models` lists (see [Models on `api.nyolab.ai`](#models-on-apinyolabai)). `halohalo/asr-<lang>` (also what an empty `model` derives from `language`) is an alias for the gateway's *current default* model for that language, which a hot swap changes; name an explicit `fw/<registry id>` to pin one. For Hiligaynon on `api.nyolab.ai` name `fw/whisper-large-v3-pld-hil-norm`. Checked on start: not listed, or not an ASR model -> FATAL `model_not_found`; a `language` the model's `languages` does not list -> FATAL `language_not_supported`. |
| `params.language` | `fil` | `bcl ceb fil hil ilo pam pag tsg war eng en zh es ...`; returned to TEN as BCP-47 (`fil-PH`, `en-US`). |
| `params.sample_rate` | `16000` | 8000/16000/24000/48000. |
| `params.interim_results` (`params.interim`) | `true` | maps to NRSP `config.interim`. |
| `params.endpointing_ms` | `1000` | an int (a bool is rejected). Server-side VAD cuts an utterance after this much silence and sends one final for it, so graphs that never send `asr_finalize` (voice-assistant) still get finals. `0` = TEN drives `asr_finalize`. Server `finalize_end` ids the addon never sent (`auto-N`) are ignored. |
| `params.buffer_ms` | `2000` | "Keep" window replayed after a reconnect. |
| `params.hotwords` | - | list of strings, forwarded in `config`. The NYO gateway and asr_fw do not apply hotwords yet, so the addon logs a warning at init when they are set. |
| `params.gc_freeze` | `true` | `gc.collect(); gc.freeze()` at the end of `on_init`, logged at info. It removes the gen-2 GC passes over the TEN + pydantic init-time heap from the audio path (measured in a TEN host: max per-frame cost 4067 us -> 125 us, a 4-8 ms stall otherwise, bench/results/ten.json). It is process-wide: it also freezes the init-time heap of every other extension in the same host. `false` leaves the host's GC alone. |
| `finalize_timeout_ms` | `2000` | max wait for `finalize_end`; `asr_finalize_end` is always sent. |
| `dump`, `dump_path` | `false`, `/tmp` | writes the PCM sent upstream to `nyo_asr_in.pcm`. |

## Models on `api.nyolab.ai`

`GET /v1/models` is the authority; the extension reads it on start. Checked 2026-11-20 (gateway release
`gateway-aws-4`, from its `/healthz` and published model table):

| Model | What |
|---|---|
| `fw/flipvox-whisper-base-fil` (alias `halohalo/asr-fil`) | Filipino speech to text, the default |
| `fw/whisper-large-v3-pld-fil-norm` | Filipino speech to text, large, slower |
| `fw/whisper-large-v3-pld-hil-norm` | Hiligaynon speech to text |
| `halohalo/tts-fil` | Filipino text to speech (Orpheus, 24 kHz, ten speakers) |
| `halohalo/tts-hil` | Hiligaynon text to speech (SpeechT5, 16 kHz, one speaker) |

## Graph snippet (agents/examples/voice-assistant)

Replace the `stt` node; connections stay as they are. This one is Hiligaynon; for Filipino use
`"model": "halohalo/asr-fil", "language": "fil"`.

```json
{
  "type": "extension",
  "name": "stt",
  "addon": "nyo_asr_python",
  "extension_group": "stt",
  "property": {
    "params": {
      "api_key": "${env:NYO_API_KEY}",
      "base_url": "${env:NYO_BASE_URL}",
      "model": "fw/whisper-large-v3-pld-hil-norm",
      "language": "hil",
      "endpointing_ms": 1000
    }
  }
}
```

and add `{"path": "../../../ten_packages/extension/nyo_asr_python"}` to `tenapp/manifest.json`.

## Behaviour in one paragraph

Frames go out as they arrive; while the socket is down the base class keeps up to
`buffer_ms` of audio and the addon replays it, in order, before any live frame once the
socket is back. Unexpected closes reconnect with 0.5/1/2/4/4... s backoff (10 attempts,
NON_FATAL errors, then FATAL). HTTP 401/403 and server `fatal:true` errors are FATAL and stop
retrying. A gateway whose engine is at its session cap answers `busy`: HTTP 503 with `Retry-After`
on the upgrade, which the addon reports as NON_FATAL and retries with the same backoff, or a non-fatal
error frame on an open socket, which it reports as NON_FATAL and the session continues. Server timestamps are mapped through TEN's `AudioTimeline`, so `start_ms` stays correct
across reconnects and dropped audio. Every result carries `metadata.asr_info.backend` and
`metadata.asr_info.model` (the model that served it, so a hot swap is visible); `vendor_metrics`
carries the gateway's `server_ttfw_ms`/`server_ttlw_ms` and the serving `model`.

## Metrics

Vendor metrics (`send_vendor_metrics`): the gateway's `server_ttfw_ms` / `server_ttlw_ms` when it sends them, else
the addon's own `client_ttfw_ms` (first audio frame of a turn -> first result with text) and `client_ttlw_ms`
(`finalize` sent -> its final, matched by `metadata.finalize_id`); a server value always wins for that turn. Auto
(endpointed) finals have no `finalize` to time from, so they carry TTFW only. Also `model` on every change,
`audio_ms` / `credits` from `usage`.

## Errors

Exactly one FATAL `asr_error` per extension instance: after it the addon closes the socket, stops sending audio and
never reconnects; later server errors are logged only. `vendor_info.vendor` is always `nyo`; when the gateway reports
an error raised by an engine (e.g. `halohalo`), the engine is named in the message (`[halohalo] ...`). The
`language` of `asr_result` is the server's when it reports one, else the configured one.

## Tests

The TEN guarders need `pytest<9` (their `conftest.py` imports `_pytest.config.Notset`, removed in pytest 9); the
unit tests run on pytest 9.

```bash
# unit + protocol tests (fake NRSP server in-process)
cd ten/nyo_asr_python && PYTHONPATH=<ten_runtime_python/lib:interface:ten_ai_base/interface:ten> pytest tests -q
# TEN guarder against the fake server
python3 ten/fake_nrsp_server.py --port 9100 &
NYO_API_KEY=fake-key task asr-guarder-test EXTENSION=nyo_asr_python   # from ai_agents/, after copying this dir into ten_packages/extension
```

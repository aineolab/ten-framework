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
| `params.model` | `""` (`property.json`) | any ASR id the gateway's `GET /v1/models` lists (see [Models on `api.nyolab.ai`](#models-on-apinyolabai)). `property.json` leaves `model` and `language` empty because TEN deep-merges a graph node's `params` into it: a node that sets only one of them must not inherit a default for the other. Resolution: both empty -> `halohalo/asr-fil` + `fil`; only `language` -> `halohalo/asr-<lang>` (below; if the gateway does not list it, a listed ASR model that declares the language, with the 32 s Nemotron `finalize_timeout_ms` default when that is an `nvidia/nemotron` model and none is set); only `model` -> the language from a `halohalo/asr-<lang>` suffix, else the single language the model declares in `GET /v1/models` (at the preflight); if that is still unknown (several languages, or no conclusive listing) `language` is left out of the handshake and `config`, and the gateway uses the model's own. `halohalo/asr-<lang>` is an alias for the gateway's *current default* model for that language, which a hot swap changes; name an explicit `fw/<registry id>` to pin one. For Hiligaynon on `api.nyolab.ai` name `fw/whisper-large-v3-pld-hil-norm` (its language `hil` is then taken from the listing). Checked against `GET /v1/models` on start, on every reconnect until it passes, and after a 400/404 handshake. The gateway lists only engines that are ready now, so a model it does not list may be a typo or an engine that is down: one NON_FATAL `asr_error` (`vendor_info.code` `model_not_found`, rate-limited like other reconnect errors), audio keeps buffering, the addon keeps retrying, and if the model is still missing when `reconnect_window_ms` runs out the single FATAL carries `model_not_found`. A listing with no ASR model of that id namespace (engine swap, boot) or no 200 answer is not counted as missing. FATAL at once: the model is listed but as a TTS model (`model_not_found`) or without the configured language in its `languages` list / `language` string (`language_not_supported`). |
| `params.language` | `""` (`property.json`; `fil` when `model` is empty too) | an ISO code (`fil`, `hil`, `ceb`, `eng` = Philippine English, `en` ...) or a BCP-47 tag / alias, normalised case-insensitively before use: `fil-PH` / `tl` -> `fil`, `hil-PH` -> `hil`, `en-US` -> `en`, `en-PH` -> `eng`. Returned to TEN as BCP-47 (`fil-PH`, `hil-PH`, `en-US`); when it is unknown, the server's per-result `language` is returned. An empty `model` is derived as `halohalo/asr-<lang>` for every language; which ids exist is what `GET /v1/models` lists (e.g. `halohalo/asr-fil` on `api.nyolab.ai`). The gateway lists `halohalo/asr-<lang>` for each configured language whenever its ASR engine is up, so when it lists some `halohalo/` ASR id but not ours the preflight uses a listed ASR model that declares the language (the gateway's `default` one first, logged at info; `language: "hil"` alone -> `fw/whisper-large-v3-pld-hil-norm` on `api.nyolab.ai`), and if none does it is FATAL `model_not_found` at once, the message naming the served ASR ids. A listing with no `halohalo/` ASR id (engine swap, boot) stays inconclusive and is retried. An explicitly configured `model` is never replaced. |
| `params.sample_rate` | `16000` | an int: 8000/16000/24000/44100/48000 (what `serve/` accepts; it resamples to 16 kHz). |
| `params.interim_results` (`params.interim`) | `true` | maps to NRSP `config.interim`. Either key is accepted (`interim_results` wins if a node sets both); `property.json` ships neither, so the one a graph node sets is the one used after TEN's deep merge. |
| `params.endpointing_ms` | `1000` | an int (a bool is rejected). Server-side VAD cuts an utterance after this much silence and sends one final for it, so graphs that never send `asr_finalize` (voice-assistant) still get finals. `0` = TEN drives `asr_finalize`. Server `finalize_end` ids the addon never sent (`auto-N`) are ignored. |
| `params.buffer_ms` | `2000` | "Keep" window replayed after a reconnect. |
| `params.reconnect_window_ms` | `120000` | reconnect budget of one outage: retries (jittered backoff 0.5/1/2/4/5/5... s, `Retry-After` of a 429/503 honoured up to 15 s) until the outage has lasted this long, then one FATAL. An outage ends only once a connection proves stable (it delivered a result or `finalize_end`, or stayed up 10 s), so a gateway that accepts and kills every socket also ends in one FATAL. |
| `params.hotwords` | - | list of strings, forwarded in `config`. The NYO gateway and asr_fw do not apply hotwords yet, so the addon logs a warning at init when they are set. |
| `params.gc_freeze` | `false` | opt-in. `true`: `gc.collect(); gc.freeze()` at the end of `on_init`, logged at info. It removes the gen-2 GC passes over the TEN + pydantic init-time heap from the audio path (measured in a TEN host: max per-frame cost 4067 us -> 125 us, a 4-8 ms stall otherwise, bench/results/ten.json). It is process-wide: it also freezes the init-time heap of every other extension in the same host, so it is off unless the host owner enables it. |
| `finalize_timeout_ms` | `3500` (`32000` for `nvidia/nemotron-3.5-transcribe`) when not set | max wait for `finalize_end`; an explicit value always wins. The defaults sit above the gateway's own finalize timers (3000 ms streaming, 30 s Nemotron batch decode) so its `finalize_end` and any late final come first. `asr_finalize_end` is always sent, exactly once per `asr_finalize`, with that finalize's id (overlapping finalizes each get their own; a final counts for the finalize its `metadata.finalize_id` names, else the oldest one still waiting). A finalize that ends without a final promotes only its own turn's interim: the one that was latest when it was sent, or one starting before the audio sent by then, never a later turn's (interims are tracked per result id; a final ends only its own utterance's interim, and a promotion only the interims inside its own span, so an earlier turn whose final is still due keeps its interim). `property.json` leaves it unset so the model default applies. |
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

Replace the `stt` node; connections stay as they are. This one is Hiligaynon (setting only the `model` is
enough: its language comes from `GET /v1/models`); for Filipino use `"model": "halohalo/asr-fil", "language": "fil"`,
or set neither.

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
socket is back (if the socket dies mid-replay, the frames not yet sent stay buffered for the next connection). One connect attempt runs at a time: a `trigger_connect` during an outage joins the running reconnect, one after
`on_stop` is ignored (an `on_stop` during the preflight or the handshake closes that socket, and `on_deinit` closes any socket still open), and callbacks from a socket that is no longer the current one (superseded, or closed by stop or
a FATAL) are ignored, so it cannot disturb the live session. Unexpected closes reconnect with jittered 0.5/1/2/4/5/5... s backoff (connect timeout
20 s, as the gateway queues opens up to 15 s during an engine swap) until the outage outlasts
`reconnect_window_ms`, then FATAL; an outage only ends once a connection proves stable (a result or
`finalize_end`, or 10 s up). The first NON_FATAL reconnect error of an outage (connect failure or reconnectable
server error) is reported, later ones at most every 10 s. HTTP 401/403 and TLS failures are FATAL at once; a 400/404
handshake re-checks `GET /v1/models`: FATAL only when the model is listed without our language (or not as ASR);
a model missing from the listing (and a server `unknown_model` error, and any handshake 404) is NON_FATAL `model_not_found` and retried,
then FATAL `model_not_found` if the window runs out. Server `fatal:true` errors
`engine_restarting` (engine swap, close 1012), `engine_unavailable` and `slow_consumer` end the socket, not the
service: they are reported NON_FATAL (rate-limited as above) and reconnected. `idle_timeout` (the gateway's idle timer counts audio only)
is not reported to the graph: the addon logs it and reconnects on the next audio frame. Every other `fatal:true`
error is FATAL. An `asr_finalize` that arrives while the Keep buffer is being replayed waits for the replay (it never
overtakes audio captured before it); one that arrives while disconnected promotes the last interim once and is
answered at once. An unexpected socket close (not `on_stop`, not a FATAL) promotes every turn still pending, each one's last interim, once
and in start order, as finals marked `asr_info.promoted_on_close` (also when the close ends a finalize wait; a finalize
timeout's promotion is marked `promoted_on_timeout`); no late-final
suppression is armed for it, because the next socket is a new gateway session that never re-decodes audio sent on
the old one. A `finalize_end` that arrives without a final (engine lost/re-opened, gateway finalize timeout)
promotes the last interim once, like a client-side timeout (a final the server endpointed, `auto-N`, answers no
`asr_finalize`; a `finalize_end` followed at once by a close is still a normal end). The re-opened engine's re-decode of
that promoted turn is dropped; a result naming another `asr_finalize`, or running past the promoted span (by more than
`endpointing_ms` + 0.5 s), is a later turn and is kept, and so is the re-decode of an earlier turn that ends before the
span. Suppression lasts until a final of a later turn arrives (the next turn's interims do not end it: they can come
before the promoted turn's replayed final). A gateway whose engine is at its session cap answers `busy`: HTTP 503 with `Retry-After`
on the upgrade, which the addon reports as NON_FATAL and retries inside `reconnect_window_ms`, or a non-fatal
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
never reconnects; later server errors are logged only. Reconnectable server errors and connect failures inside
`reconnect_window_ms` are NON_FATAL (1000; a rejected handshake carries its HTTP status, e.g. `503`, as `vendor_info.code`; a 404 carries `model_not_found`); `idle_timeout` is no error at all (see Behaviour). `vendor_info.vendor` is always `nyo`; when the gateway reports
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

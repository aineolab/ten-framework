# NYO Voice Assistant

The standard voice assistant (Agora RTC → STT → LLM → TTS → Agora RTC) with speech served by a
self-hosted NYO speech gateway (NRSP v1): `nyo_asr_python` for STT and
`nyo_tts_python` for TTS. It has two graphs:

| Graph | Language | Speech to text | Text to speech |
|---|---|---|---|
| `nyo_voice_assistant` (starts by default) | Filipino (`fil`) | `halohalo/asr-fil`, the default Filipino model (on `api.nyolab.ai`: `fw/flipvox-whisper-base-fil`, a fine-tuned Whisper base) | `halohalo/tts-fil` (Orpheus, 24 kHz, ten speakers) |
| `nyo_voice_assistant_hil` | Hiligaynon (`hil`) | `fw/whisper-large-v3-pld-hil-norm` | `halohalo/tts-hil` (SpeechT5, 16 kHz) |

The larger Filipino model `fw/whisper-large-v3-pld-fil-norm` is also served on `api.nyolab.ai`. It is slower and
not served by every NYO server: an agent configured with a model the answering server lacks fails at start
(`model_not_found`). The Hiligaynon graph tells the LLM to answer in Hiligaynon through its `prompt`.

```
agora_rtc → streamid_adapter → stt (nyo_asr_python) → main_control → llm (openai_llm2_python)
          ← tts (nyo_tts_python) ←───────────────────────────────────┘
```

**Using a coding agent?** Point it at [AGENTS.md](AGENTS.md). [nyo_check.py](nyo_check.py) checks your key and the
service before you build anything.

## Prerequisites

| Variable | Required | Meaning |
|---|---|---|
| `AGORA_APP_ID` | yes | Agora App ID |
| `AGORA_APP_CERTIFICATE` | no | Agora App Certificate |
| `NYO_API_KEY` | yes | Bearer key for the NYO gateway. Missing → FATAL `-1000` at init |
| `NYO_BASE_URL` | yes | Service address: `wss://api.nyolab.ai`, prefilled in `ai_agents/.env.example`. Change it only for a private or staging deployment |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | yes | Any OpenAI-compatible LLM |

`ws://` is accepted only for loopback (`127.0.0.1`, `localhost`, `::1`), for example through
`ssh -L 9200:127.0.0.1:9200 <gateway host>` with `NYO_BASE_URL=ws://127.0.0.1:9200`. A `ws://` URL to any other
host fails at start with a FATAL `insecure_url` error unless `allow_insecure: true`. TLS certificates are
verified (`tls_verify: true`).

## Why `endpointing_ms` is set

This graph never sends `asr_finalize`: `main_control` reacts to final `asr_result`s. The STT node therefore sets
`endpointing_ms: 1000`, so the gateway's voice-activity detection cuts each utterance after 1 s of silence and
sends exactly one final per utterance. With `endpointing_ms: 0` the graph would get interim results only and the
LLM would never be called.

## Models: change them in `property.json` only

On start each extension calls `GET <gateway>/v1/models` and fails fast (FATAL, `vendor_info.code =
"model_not_found"`, with the list of available ids) if `model` is not served. Any id that endpoint lists works:

- ASR: `fw/<registry id>` (on `api.nyolab.ai`: `fw/flipvox-whisper-base-fil`, `fw/whisper-large-v3-pld-fil-norm`,
  `fw/whisper-large-v3-pld-hil-norm`) or `halohalo/asr-<lang>` (the gateway's default model for that language,
  for example `halohalo/asr-fil`).
- TTS: `halohalo/tts-<lang>` (on `api.nyolab.ai`: `halohalo/tts-fil`, `halohalo/tts-hil`).

When an engine is at its session cap the gateway answers `busy`. Both extensions report it as a non-fatal error;
a refused connection is retried and the agent keeps running.

Swapping the model behind the gateway (hot swap) needs no change here; the model that served each result is
reported in `asr_result.metadata.asr_info.model` and in the `vendor_metrics` metric.

## Run

```bash
cd agents/examples/nyo-voice-assistant
task install
task run
```

Open the playground at http://localhost:3000 and pick the `nyo_voice_assistant` graph, or
`nyo_voice_assistant_hil` for Hiligaynon.

## Metrics

`nyo_asr_python` reports TTFW/TTLW (base class), plus `vendor_metrics` with the gateway-measured
`server_ttfw_ms`/`server_ttlw_ms` and the serving `model`. `nyo_tts_python` reports TTFB (base class) and a
`metrics` message per request with `server_ttfb_ms`, `server_ttlw_ms` and `model`.

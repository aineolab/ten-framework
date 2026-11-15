# NYO Voice Assistant

The standard voice assistant (Agora RTC → STT → LLM → TTS → Agora RTC) with speech served by a
self-hosted NYO speech gateway (NRSP v1): `nyo_asr_python` for STT and
`nyo_tts_python` for TTS. It is tuned for Filipino (`fil`) with the fine-tuned Whisper model
`fw/whisper-large-v3-pld-fil-norm`.

```
agora_rtc → streamid_adapter → stt (nyo_asr_python) → main_control → llm (openai_llm2_python)
          ← tts (nyo_tts_python) ←───────────────────────────────────┘
```

## Prerequisites

| Variable | Required | Meaning |
|---|---|---|
| `AGORA_APP_ID` | yes | Agora App ID |
| `AGORA_APP_CERTIFICATE` | no | Agora App Certificate |
| `NYO_API_KEY` | yes | Bearer key for the NYO gateway. Missing → FATAL `-1000` at init |
| `NYO_BASE_URL` | yes | Gateway URL, for example `wss://gateway.example.net`. There is no default |
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

- ASR: `fw/<registry id>` (faster-whisper models, for example `fw/whisper-large-v3-pld-fil-norm`) or
  `halohalo/asr-<lang>` (the gateway's default model for that language).
- TTS: `halohalo/tts-<lang>`.

Swapping the model behind the gateway (hot swap) needs no change here; the model that served each result is
reported in `asr_result.metadata.asr_info.model` and in the `vendor_metrics` metric.

## Run

```bash
cd agents/examples/nyo-voice-assistant
task install
task run
```

Open the playground at http://localhost:3000 and pick the `nyo_voice_assistant` graph.

## Metrics

`nyo_asr_python` reports TTFW/TTLW (base class), plus `vendor_metrics` with the gateway-measured
`server_ttfw_ms`/`server_ttlw_ms` and the serving `model`. `nyo_tts_python` reports TTFB (base class) and a
`metrics` message per request with `server_ttfb_ms`, `server_ttlw_ms` and `model`.
